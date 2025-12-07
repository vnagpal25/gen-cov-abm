import torch
import torch.nn.functional as F
import re

from AgentTorch.agent_torch.core.substep import SubstepTransition
from AgentTorch.agent_torch.core.helpers import get_by_path
from ..genome_mlps import ProgressMLP


class SEIRMProgression(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = torch.device(self.config["simulation_metadata"]["device"])
        self.num_timesteps = self.config["simulation_metadata"]["num_steps_per_episode"]

        self.SUSCEPTIBLE_VAR = self.config["simulation_metadata"]["SUSCEPTIBLE_VAR"]
        self.EXPOSED_VAR = self.config["simulation_metadata"]["EXPOSED_VAR"]
        self.INFECTED_VAR = self.config["simulation_metadata"]["INFECTED_VAR"]
        self.RECOVERED_VAR = self.config["simulation_metadata"]["RECOVERED_VAR"]
        self.MORTALITY_VAR = self.config["simulation_metadata"]["MORTALITY_VAR"]
        self.RESET_VAR = -1 * self.RECOVERED_VAR

        self.STAGE_SAME_VAR = 0
        self.STAGE_UPDATE_VAR = 1

        self.calibration_mode = self.config["simulation_metadata"]["calibration"]

        self.INFINITY_TIME = (
            self.config["simulation_metadata"]["num_steps_per_episode"] + 20
        )
        self.INFECTED_TO_RECOVERED_TIME = self.config["simulation_metadata"][
            "INFECTED_TO_RECOVERED_TIME"
        ]

        self.genome_mlp = ProgressMLP(input_dim=1280, hidden_dim=512, output_dim=1).to(
            self.device
        )

    def _compute_genome_modifier(self, protein_features):
        """
        Compute genome progression modifier using MLP.
        Agents with zero vectors (no genomic info) get a default modifier of 1.0.

        protein_features: tensor of shape (num_agents, embedding_dim)
        Returns: tensor of shape (num_agents, 1) with modifiers
        """
        # check which agents have zero vectors (no genomic information)
        # sum across embedding dimension; zero vectors will have sum = 0
        has_genomic_info = (protein_features.abs().sum(dim=1) > 0).float()

        # initialize all modifiers to 1.0 (no modification)
        genome_modifier = torch.ones(
            protein_features.shape[0], 1, device=self.device
        )

        # Only run MLP for agents with genomic information
        agents_with_info = has_genomic_info > 0

        if agents_with_info.any():
            mlp_output = self.genome_mlp(protein_features[agents_with_info])
            # Assign output to corresponding positions
            genome_modifier[agents_with_info] = mlp_output

        return genome_modifier

    def _generate_one_hot_tensor(self, timestep, num_timesteps):
        one_hot_tensor = F.one_hot(torch.tensor(timestep), num_classes=num_timesteps)
        return one_hot_tensor.to(self.device).float()

    def update_daily_deaths(
        self, t, daily_dead, current_stages, current_transition_times
    ):
        # recovered or dead agents
        recovered_and_dead_mask = (current_stages == self.INFECTED_VAR) * (
            current_transition_times <= t
        )

        new_death_recovered_today = (
            current_stages * recovered_and_dead_mask / self.INFECTED_VAR
        )

        if self.calibration_mode:
            num_dead_today = new_death_recovered_today.sum() * self.calibrate_M.to(
                self.device
            )
        else:
            num_dead_today = new_death_recovered_today.sum() * self.learnable_args["M"]

        candidate_indices = torch.where(recovered_and_dead_mask.squeeze(1))[0]

        if num_dead_today > 0 and len(candidate_indices) > 0:
            num_deaths = int(num_dead_today.item())
            # Randomly select agents to die
            if num_dead_today < len(candidate_indices):
                dead_indices = candidate_indices[
                    torch.randperm(len(candidate_indices))[:num_deaths]
                ]

            else:
                dead_indices = candidate_indices

            # Update stages for dead agents
            current_stages[dead_indices] = self.MORTALITY_VAR
            # Update stages for recovered agents (those not selected to die)
            recovered_indices = candidate_indices[
                ~torch.isin(candidate_indices, dead_indices)
            ]
            if len(recovered_indices) > 0:
                current_stages[recovered_indices] = self.RECOVERED_VAR

        daily_dead = (
            daily_dead
            + self._generate_one_hot_tensor(t, self.num_timesteps) * num_dead_today[0]
        )
        return current_stages, daily_dead

    def update_current_stages(self, t, current_stages, current_transition_times):
        transit_agents = (current_transition_times <= t) * self.STAGE_UPDATE_VAR
        stage_transition = (current_stages == self.EXPOSED_VAR) * transit_agents + (
            current_stages == self.INFECTED_VAR
        ) * transit_agents

        new_stages = current_stages + stage_transition
        return new_stages

    def update_next_transition_times(self, t, current_stages, current_transition_times):
        new_transition_times = torch.clone(current_transition_times).to(
            current_transition_times.device
        )
        curr_stages = torch.clone(current_stages).to(current_stages.device)

        new_transition_times[
            (curr_stages == self.INFECTED_VAR) * (current_transition_times == t)
        ] = self.INFINITY_TIME
        new_transition_times[
            (curr_stages == self.EXPOSED_VAR) * (current_transition_times == t)
        ] = (t + self.INFECTED_TO_RECOVERED_TIME)

        del current_transition_times
        del current_stages

        return new_transition_times

    def forward(self, state, action):
        """Update stage and transition times for already infected agents"""
        input_variables = self.input_variables
        t = state["current_step"]

        current_stages = state["agents"]["citizens"]["disease_stage"]
        agents_next_stage_times = state["agents"]["citizens"]["next_stage_time"]
        daily_deaths = state["environment"]["daily_deaths"]
        protein_features = state["agents"]["citizens"]["protein_features"]

        # Compute genome progression modifier using MLP
        # (skips agents with zero vectors)
        
        genome_modifier = self._compute_genome_modifier(protein_features)

        # apply genome modifier to the DURATION (delta_t) between stages, not absolute time
        # calculatre duration from current time to next stage transition
        delta_t = agents_next_stage_times - t
        
        # Apply modifier to duration: modifier < 1 speeds up, modifier > 1 slows down
        # speed up progression as necessary by genome
        # TODO modifier always less than 1 so will always speed up the
        # infection
        # perhaps scale to [0, 2] perhaps 0 < x < 1 -> speed up
        #                         1 < x < 2 -> slow down
        genome_adjusted_delta_t = delta_t * genome_modifier
        
        # calculate new absolute transition times
        genome_adjusted_next_stage_times = t + genome_adjusted_delta_t

        # new_stages = self.update_current_stages(
        #     t, current_stages, agents_next_stage_times
        # )
        new_stages = self.update_current_stages(
            t, current_stages, genome_adjusted_next_stage_times
        )

        # new_transition_times = self.update_next_transition_times(
        #     t, current_stages, agents_next_stage_times
        # )
        new_transition_times = self.update_next_transition_times(
            t, current_stages, genome_adjusted_next_stage_times
        )

        # new_stages, new_daily_deaths = self.update_daily_deaths(
        #     t, daily_deaths, current_stages, agents_next_stage_times
        # )
        new_stages, new_daily_deaths = self.update_daily_deaths(
            t, daily_deaths, current_stages, genome_adjusted_next_stage_times
        )
        return {
            self.output_variables[0]: new_stages,
            self.output_variables[1]: new_transition_times,
            self.output_variables[2]: new_daily_deaths,
        }
