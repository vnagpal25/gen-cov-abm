import torch
from torch_geometric.data import Data
import torch.nn.functional as F
import re
import random
import math
import pandas as pd
import csv
import pdb
import os
import pickle
import glob
import numpy as np

from AgentTorch.agent_torch.core.substep import SubstepTransitionMessagePassing
from AgentTorch.agent_torch.core.helpers import get_by_path
from AgentTorch.agent_torch.core.helpers import set_by_path
from AgentTorch.agent_torch.core.distributions import StraightThroughBernoulli
from ..genome_mlps import TransmitMLP


class NewTransmission(SubstepTransitionMessagePassing):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

        self.device = torch.device(self.config["simulation_metadata"]["device"])
        self.SUSCEPTIBLE_VAR = self.config["simulation_metadata"]["SUSCEPTIBLE_VAR"]
        self.EXPOSED_VAR = self.config["simulation_metadata"]["EXPOSED_VAR"]
        self.RECOVERED_VAR = self.config["simulation_metadata"]["RECOVERED_VAR"]
        self.INFECTED_VAR = self.config["simulation_metadata"]["INFECTED_VAR"]
        self.MORTALITY_VAR = self.config["simulation_metadata"]["MORTALITY_VAR"]

        self.num_timesteps = self.config["simulation_metadata"]["num_steps_per_episode"]
        self.num_weeks = self.config["simulation_metadata"]["NUM_WEEKS"]

        self.STAGE_UPDATE_VAR = 1
        self.INFINITY_TIME = self.config["simulation_metadata"]["INFINITY_TIME"]
        self.EXPOSED_TO_INFECTED_TIME = self.config["simulation_metadata"][
            "EXPOSED_TO_INFECTED_TIME"
        ]
        self.INFECTED_TO_RECOVERED_TIME = self.config["simulation_metadata"][
            "INFECTED_TO_RECOVERED_TIME"
        ]

        self.mode = self.config["simulation_metadata"]["EXECUTION_MODE"]
        self.st_bernoulli = StraightThroughBernoulli.apply

        self.calibration_mode = self.config["simulation_metadata"]["calibration"]

        self.genome_mlp = TransmitMLP(input_dim=1280, hidden_dim=512, output_dim=1).to(
            self.device
        )

        # load county-specific genome embeddings
        self.county_embeddings = self._load_county_embeddings()

        # self.social_distancing_schedule = self.generate_social_distancing_schedule(
        #     initial_factor=1.0, lambda_=0.01, total_steps=self.num_timesteps
        # ).to(self.device)

    def _load_county_embeddings(self):
        """
        Load genome embeddings for the county from the embeddings directory.
        Returns a tensor of shape (num_embeddings, embedding_dim) or None if not found.
        """
        # extract county code from population_dir
        population_dir = self.config["simulation_metadata"].get("population_dir", "")

        # pop_dir/pop25001' -> '25001'
        county_code = None
        match = re.search(r"pop(\d+)", population_dir)
        if match:
            county_code = match.group(1)

        if not county_code:
            raise Exception(
                f"Error: Could not extract county code from population_dir: {population_dir}"
            )

        # TODO fix this hack
        # look for embeddings directory
        # try both relative to workspace and relative to population_dir
        possible_paths = [
            os.path.join("embeddings", county_code),
            os.path.join(
                os.path.dirname(population_dir), "..", "embeddings", county_code
            ),
        ]

        embeddings_dir = None
        for path in possible_paths:
            if os.path.exists(path):
                embeddings_dir = path
                break

        if not embeddings_dir:
            print(f"")
            raise Exception(
                f"Error: Embeddings directory not found for county {county_code}\nSearched paths: {possible_paths}"
            )

        # load all .npy and .pickle files from the embeddings directory
        embedding_files = glob.glob(os.path.join(embeddings_dir, "*.npy"))

        if not embedding_files:
            raise Exception(f"Warning: No embedding files found in {embeddings_dir}")

        embeddings_list = []
        for file_path in embedding_files:
            try:

                data = np.load(file_path)

                # If 2D array, add all rows; if 1D, treat as single embedding
                if data.ndim == 1:
                    embeddings_list.append(data)
                elif data.ndim == 2:
                    embeddings_list.extend(data)

            except Exception as e:
                raise Exception(f"Warning: Failed to load {file_path}: {e}")

        if not embeddings_list:
            raise Exception(
                f"Warning: No valid embeddings loaded from {embeddings_dir}"
            )

        # convert embeddings to tensor
        embeddings_tensor = torch.tensor(
            np.array(embeddings_list), dtype=torch.float32
        ).to(self.device)

        return embeddings_tensor

    def _initialize_protein_features(self, state, infected_agents_mask):
        """
        Initialize protein features for agents at t=0.
        Infected agents get random embeddings from county, others stay as zeros.

        state: starting simulation state
        infected_agents_mask: boolean tensor indicating infected agents
        """
        protein_features = state["agents"]["citizens"]["protein_features"]

        # indices of infected agents
        infected_indices = torch.where(infected_agents_mask.squeeze())[0]
        num_infected = len(infected_indices)

        if num_infected:
            # randomly sample embeddings for infected agents (with replacement)
            random_indices = torch.randint(
                0, len(self.county_embeddings), (num_infected,), device=self.device
            )
            sampled_embeddings = self.county_embeddings[random_indices]

            # assign sampled embeddings to infected agents
            protein_features[infected_indices] = sampled_embeddings

        return state

    def _update_protein_features_for_newly_infected(
        self, state, newly_exposed_mask, all_edgelist, current_stages
    ):
        """
        update protein features for newly infected agents by averaging
        infected neighbors' embeddings.
        If no infected neighbors exist, randomly sample from county embeddings.

        state: simulation state
        newly_exposed_mask: boolean tensor of newly exposed agents
        all_edgelist: edge list from adjacency matrix
        current_stages: disease stages of all agents
        """
        protein_features = state["agents"]["citizens"]["protein_features"]

        # indices of newly exposed agents
        newly_exposed_indices = torch.where(newly_exposed_mask.squeeze())[0]

        if len(newly_exposed_indices) == 0:
            return state

        # identify exposed agents (E or I stage)
        infected_mask = torch.logical_and(
            current_stages > self.SUSCEPTIBLE_VAR, current_stages < self.RECOVERED_VAR
        ).squeeze()

        # for each newly exposed agent, find their infected neighbors
        for agent_idx in newly_exposed_indices:
            # find edges where this agent is the target (receiving infection)
            incoming_edges = all_edgelist[1] == agent_idx

            # get those neighbors
            neighbor_indices = all_edgelist[0][incoming_edges]

            # filter to only exposed neighbors
            infected_neighbors = neighbor_indices[infected_mask[neighbor_indices]]

            if infected_neighbors:
                # Average the embeddings of infected neighbors
                neighbor_embeddings = protein_features[infected_neighbors]
                avg_embedding = neighbor_embeddings.mean(dim=0)
                protein_features[agent_idx] = avg_embedding
            else:
                # No infected neighbors, sample randomly from county
                random_idx = torch.randint(
                    0, len(self.county_embeddings), (1,), device=self.device
                )
                protein_features[agent_idx] = self.county_embeddings[random_idx]

        return state

    def _compute_genome_modifier(self, protein_features):
        """
        Compute genome transmission modifier using MLP.
        Agents with zero vectors (no genomic info) get a default modifier of 1.0.

        returns (num_agents, 1) tensor with modifiers
        """
        # check which agents have zero vectors (no genomic information)
        # sum across embedding dimension; zero vectors will have sum = 0
        has_genomic_info = (protein_features.abs().sum(dim=1) > 0).float()

        # Initialize all modifiers to 1.0 (no modification)
        genome_modifier = torch.ones(protein_features.shape[0], 1, device=self.device)

        # Only run MLP for agents with genomic information
        agents_with_info = has_genomic_info > 0

        if agents_with_info.any():
            # Run MLP only on agents with genomic info
            mlp_output = self.genome_mlp(protein_features[agents_with_info])
            # Assign MLP output to corresponding positions
            genome_modifier[agents_with_info] = mlp_output

        return genome_modifier

    def _lam(
        self,
        x_i,
        x_j,
        edge_attr,
        t,
        R,
        SFSusceptibility,
        SFInfector,
        lam_gamma_integrals,
    ):
        S_A_s = SFSusceptibility[x_i[:, 0].long()]
        A_s_i = SFInfector[x_j[:, 1].long()]
        B_n = edge_attr[1, :]
        integrals = torch.zeros_like(B_n)
        infected_idx = x_j[:, 2].bool()
        infected_times = t - x_j[infected_idx, 3] - 1
        infected_times = infected_times.clamp(
            min=0, max=lam_gamma_integrals.size(0) - 1
        )

        integrals[infected_idx] = lam_gamma_integrals[infected_times.long()]
        edge_network_numbers = edge_attr[0, :]

        I_bar = torch.gather(x_i[:, 4], 0, edge_network_numbers.long()).view(-1)

        will_isolate = x_i[:, 6]  # is the susceptible agent isolating? check x_i vs x_j
        not_isolated = 1 - will_isolate

        # between 0 and 1
        genome_modifier = x_i[:, 7]

        if self.mode == "llm":
            base_lambda = (
                R * S_A_s * A_s_i * B_n * integrals / I_bar
            )  # not_isolated*R*S_A_s*A_s_i*B_n*integrals/I_bar * 1/2
        else:
            base_lambda = R * S_A_s * A_s_i * B_n * integrals / I_bar

        genome_informed_lambda = genome_modifier * base_lambda

        return genome_informed_lambda.view(-1, 1)

    def message(
        self,
        x_i,
        x_j,
        edge_attr,
        t,
        R,
        SFSusceptibility,
        SFInfector,
        lam_gamma_integrals,
    ):
        return self._lam(
            x_i, x_j, edge_attr, t, R, SFSusceptibility, SFInfector, lam_gamma_integrals
        )

    def update_stages(
        self, t, current_stages, agents_next_stage_times, newly_exposed_today
    ):
        transition_to_infected = self.INFECTED_VAR * (
            agents_next_stage_times <= t
        ) + self.EXPOSED_VAR * (agents_next_stage_times > t)
        transition_to_mortality_or_recovered = self.RECOVERED_VAR * (
            agents_next_stage_times <= t
        ) + self.INFECTED_VAR * (
            agents_next_stage_times > t
        )  # can be stochastic --> recovered or mortality

        # Stage progression for agents NOT newly exposed today'''
        # if S -> stay S; if E/I -> see if time to transition has arrived; if R/M -> stay R/M
        stage_progression = (
            (current_stages == self.SUSCEPTIBLE_VAR) * self.SUSCEPTIBLE_VAR
            + (current_stages == self.RECOVERED_VAR) * self.RECOVERED_VAR
            + (current_stages == self.MORTALITY_VAR) * self.MORTALITY_VAR
            + (current_stages == self.EXPOSED_VAR) * transition_to_infected
            + (current_stages == self.INFECTED_VAR) * self.INFECTED_VAR
        )

        # update curr stage - if exposed at current step t or not
        current_stages = newly_exposed_today * self.EXPOSED_VAR + stage_progression
        return current_stages

    def update_transition_times(
        self, t, agents_next_stage_times, newly_exposed_today, current_stages
    ):
        """Note: not differentiable"""
        """ update time """
        exposed_to_infected_time = self.EXPOSED_TO_INFECTED_TIME
        infected_to_recovered_time = self.INFECTED_TO_RECOVERED_TIME
        # for non-exposed
        # if S, R, M -> set to default value; if E/I -> update time if your transition time arrived in the current time
        new_transition_times = torch.clone(agents_next_stage_times)
        curr_stages = torch.clone(current_stages).long()
        # new_transition_times[(curr_stages==self.INFECTED_VAR)*(agents_next_stage_times == t)] = self.INFINITY_TIME
        new_transition_times[
            (curr_stages == self.EXPOSED_VAR) * (agents_next_stage_times == t)
        ] = (t + infected_to_recovered_time)
        return (
            newly_exposed_today * (t + 1 + exposed_to_infected_time)
            + (1 - newly_exposed_today) * new_transition_times
        )

    def _generate_one_hot_tensor(self, timestep, num_timesteps):
        timestep_tensor = torch.tensor([timestep])
        one_hot_tensor = F.one_hot(timestep_tensor, num_classes=num_timesteps)
        one_hot_tensor = one_hot_tensor.view(
            1, -1
        )  # Ensure 2D shape (1, num_timesteps)
        return one_hot_tensor.to(self.device)[0]

    def update_infected_times(self, t, agents_infected_times, newly_exposed_today):
        """Note: not differentiable"""
        updated_infected_times = torch.clone(agents_infected_times).to(
            agents_infected_times.device
        )

        updated_infected_times[newly_exposed_today.bool()] = t

        return updated_infected_times

    def recover_random_agents(self, current_stages, num_recoveries=20):

        susceptible_indices = torch.where(current_stages == self.SUSCEPTIBLE_VAR)[0]

        num_recoveries = min(num_recoveries, len(susceptible_indices))

        if num_recoveries > 0:
            recover_indices = torch.randperm(len(susceptible_indices))[:num_recoveries]
            susceptible_to_recover = susceptible_indices[recover_indices]

            updated_stages = current_stages
            updated_stages[susceptible_to_recover] = self.RECOVERED_VAR
        else:
            updated_stages = current_stages

        return updated_stages

    def get_stage_proportions(self, t, current_stages):
        total_people = len(current_stages)
        stage_names = ["susceptible", "exposed", "infected", "recovered", "dead"]

        stage_counts = torch.stack(
            [
                (current_stages == self.SUSCEPTIBLE_VAR).sum(),
                (current_stages == self.EXPOSED_VAR).sum(),
                (current_stages == 2).sum(),
                (current_stages == self.RECOVERED_VAR).sum(),
                (current_stages == 4).sum(),
            ]
        )

        proportions = stage_counts.float() / total_people

        for i, proportion in enumerate(proportions):
            print(f"Proportion of people in {stage_names[i]}: {proportion.item():.4f}")

        print("------------")

        csv_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "washtenaw_info.csv"
        )
        csv_path = os.path.abspath(csv_path)

        if t == 0:
            # Overwrite the file and write header
            with open(csv_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    ["t", "susceptible", "exposed", "infected", "recovered", "dead"]
                )

        with open(csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([t] + [p.item() for p in proportions])

        return proportions

    def modify_initial_infected(self, current_stages, proportion):

        for num in range(len(current_stages)):
            if current_stages[num][0] == 1:
                current_stages[num][0] = 0

        total_numbers = len(current_stages)
        num_ones_to_insert = int(total_numbers * proportion)

        indices = list(range(total_numbers))
        ones_indices = random.sample(indices, num_ones_to_insert)

        for idx in ones_indices:
            current_stages[idx][0] = 2.0

        return current_stages

    def generate_social_distancing_schedule(
        self, initial_factor=1.0, lambda_=0.01, total_steps=28
    ):
        social_distancing_schedule = []
        current_factor = initial_factor

        for t in range(total_steps):
            current_factor = current_factor * math.exp(-lambda_)
            social_distancing_schedule.append(current_factor)

        return torch.tensor(social_distancing_schedule, dtype=torch.float32)

    def get_infected_time(self, agents_stages):

        agents_infected_time = (500) * torch.ones_like(
            agents_stages
        )  # init all values to infinity time
        agents_infected_time[agents_stages == 1] = (
            -1
        )  # set previously infected agents to -1
        agents_infected_time[agents_stages == 2] = (
            -1 * self.EXPOSED_TO_INFECTED_TIME
        )  # -1*exposed_to_infected_time

        return agents_infected_time.float()

    def get_mean_agent_interactions(self, agents_ages):
        ADULT_LOWER_INDEX, ADULT_UPPER_INDEX = (
            1,
            4,
        )  # ('U19', '20t29', '30t39', '40t49', '50t64', '65A')

        agents_mean_interactions = 0 * torch.ones(
            size=agents_ages.shape
        )  # shape: (num_agents)
        mean_int_ran_mu = torch.tensor([2, 3, 4]).float()  # child, adult, elderly

        child_agents = (agents_ages < ADULT_LOWER_INDEX).view(-1)
        adult_agents = torch.logical_and(
            agents_ages >= ADULT_LOWER_INDEX, agents_ages <= ADULT_UPPER_INDEX
        ).view(-1)
        elderly_agents = (agents_ages > ADULT_UPPER_INDEX).view(-1)

        agents_mean_interactions[child_agents.bool(), 0] = mean_int_ran_mu[0]
        agents_mean_interactions[adult_agents.bool(), 0] = mean_int_ran_mu[1]
        agents_mean_interactions[elderly_agents.bool(), 0] = mean_int_ran_mu[2]

        return agents_mean_interactions

    def get_next_stage_time(
        self,
        t,
        agents_stages,
        exposed_to_infected_times=5,
        infected_to_recovered_times=20,
    ):
        agents_next_stage_time = (500) * torch.ones_like(
            agents_stages
        )  # init all values to infinity time
        # agents_next_stage_time[agents_stages == 1] = exposed_to_infected_times
        # agents_next_stage_time[agents_stages == 2] = (
        #     infected_to_recovered_times  # infected_to_recovered time
        # )

        agents_next_stage_time[
            (agents_stages == self.INFECTED_VAR) * (agents_stages == t)
        ] = self.INFINITY_TIME
        agents_next_stage_time[
            (agents_stages == self.RECOVERED_VAR) * (agents_stages == t)
        ] = (
            t + infected_to_recovered_times
        )  # they go back to susceptible

        return agents_next_stage_time.float()

    def update_initial_times(
        self, agents_next_stage_times, agents_infected_time, agents_stages
    ):
        infected_to_recovered_time = self.INFECTED_TO_RECOVERED_TIME
        exposed_to_infected_time = self.EXPOSED_TO_INFECTED_TIME

        agents_infected_time[agents_stages == self.EXPOSED_VAR] = -1
        agents_infected_time[agents_stages == self.INFECTED_VAR] = (
            -1 * self.EXPOSED_TO_INFECTED_TIME
        )
        agents_next_stage_times[agents_stages == self.EXPOSED_VAR] = (
            exposed_to_infected_time
        )
        agents_next_stage_times[agents_stages == self.INFECTED_VAR] = (
            infected_to_recovered_time
        )

        return agents_infected_time, agents_next_stage_times

    # # For pickle files
    # def update_adjacency_matrix(self, state, pickle_file_path):
    #     raw_mobility_data = pd.read_pickle(pickle_file_path)

    #     mobility_data = torch.tensor(raw_mobility_data.iloc[:, :2].values, dtype=torch.long).to(self.device)

    #     source_nodes = mobility_data[:, 0]
    #     target_nodes = mobility_data[:, 1]

    #     edge_list = torch.stack((source_nodes, target_nodes), dim=0)

    #     edge_attr = torch.ones(2, edge_list.size(1)).to(self.device)

    #     adjacency_matrix_path = ["network", "agent_agent", "infection_network", "adjacency_matrix"]
    #     adjacency_matrix = (edge_list, edge_attr)

    #     return set_by_path(state, adjacency_matrix_path, adjacency_matrix)

    # For single csv file
    def update_adjacency_matrix(self, state, csv_file_path):
        raw_mobility_data = pd.read_csv(csv_file_path, header=None).values

        mobility_data = torch.tensor(raw_mobility_data).to(self.device)

        source_nodes = mobility_data[:, 0]
        target_nodes = mobility_data[:, 1]

        edge_list = torch.stack((source_nodes, target_nodes), dim=0).to(self.device)

        edge_attr = torch.ones(2, edge_list.size(1)).to(self.device)

        adjacency_matrix_path = [
            "network",
            "agent_agent",
            "infection_network",
            "adjacency_matrix",
        ]
        adjacency_matrix = (edge_list, edge_attr)

        return set_by_path(state, adjacency_matrix_path, adjacency_matrix)

    def forward(self, state, action=None):
        input_variables = self.input_variables
        t = int(state["current_step"])

        # Set initial infection rate

        # Change mobility data depending on timestep
        # infection_network_file = f"{self.config['simulation_metadata']['population_dir']}/mobility_networks/0.pkl"
        infection_network_file = f"{self.config['simulation_metadata']['population_dir']}/mobility_networks/0.csv"
        # print(f"Loading infection network file for timestep {t}: {infection_network_file}")
        state = self.update_adjacency_matrix(state, infection_network_file)

        # social_distancing_factor = self.social_distancing_schedule[t]
        time_step_one_hot = self._generate_one_hot_tensor(t, self.num_timesteps)

        week_id = int(t / 7)
        week_one_hot = self._generate_one_hot_tensor(week_id, self.num_weeks)

        if self.calibration_mode:
            R_tensor = self.calibrate_R2.to(self.device)
        else:
            R_tensor = self.learnable_args["R2"]  # tensor of size NUM_WEEK

        # R_0_value = self.config['simulation_metadata']['R_0_VALUE']
        # R_tensor = torch.full((self.num_weeks, 1), R_tensor.item(), requires_grad=True)
        # if (t == 0):
        #     print(R_tensor)

        R = (R_tensor.T * week_one_hot).sum()

        current_stages = state["agents"]["citizens"]["disease_stage"]
        agents_ages = get_by_path(state, re.split("/", input_variables["age"]))
        agents_next_stage_times = state["agents"]["citizens"]["next_stage_time"]
        agents_infected_time = state["agents"]["citizens"]["infected_time"]
        initial_infection_rate = self.config["simulation_metadata"][
            "INITIAL_INFECTION_RATE"
        ]

        # initial timestep
        if t == 0:
            current_stages = self.modify_initial_infected(
                current_stages, initial_infection_rate
            )
            agents_infected_time, agents_next_stage_times = self.update_initial_times(
                agents_next_stage_times, agents_infected_time, current_stages
            )
            # boolean mask for infected agents
            infected_mask = current_stages == self.INFECTED_VAR

            # for the infected agents, initialize their protein embeddings
            state = self._initialize_protein_features(state, infected_mask)

        SFSusceptibility = get_by_path(
            state, re.split("/", input_variables["SFSusceptibility"])
        )
        SFInfector = get_by_path(state, re.split("/", input_variables["SFInfector"]))
        all_lam_gamma = get_by_path(
            state, re.split("/", input_variables["lam_gamma_integrals"])
        )

        agents_mean_interactions_split = self.get_mean_agent_interactions(agents_ages)

        # current_transition_times = self.get_next_stage_time(t, current_stages, self.EXPOSED_TO_INFECTED_TIME, 8)

        all_edgelist, all_edgeattr = get_by_path(
            state, re.split("/", input_variables["adjacency_matrix"])
        )

        daily_infected = get_by_path(
            state, re.split("/", input_variables["daily_infected"])
        )

        agents_infected_index = torch.logical_and(
            current_stages > self.SUSCEPTIBLE_VAR, current_stages < self.RECOVERED_VAR
        )

        will_isolate = action["citizens"]["isolation_decision"]

        # Compute genome transmission modifier using MLP
        # (skips agents with zero vectors)
        protein_features = state["agents"]["citizens"]["protein_features"]
        genome_transmission_modifier = self._compute_genome_modifier(protein_features)

        all_node_attr = (
            torch.stack(
                (
                    agents_ages.to(self.device),  # 0
                    current_stages.detach(),  # 1
                    agents_infected_index,  # 2
                    agents_infected_time,  # 3
                    agents_mean_interactions_split.to(
                        self.device
                    ),  # 4 *agents_mean_interactions_split,
                    torch.unsqueeze(
                        torch.arange(self.config["simulation_metadata"]["num_agents"]),
                        1,
                    ).to(
                        self.device
                    ),  # 5
                    will_isolate,  # 6
                    genome_transmission_modifier,  # 7
                )
            )
            .transpose(0, 1)
            .squeeze()
        )  # .t() # 7

        num_nodes = int(all_edgelist.max().item() + 1)
        agents_data = Data(
            all_node_attr,
            edge_index=all_edgelist,
            edge_attr=all_edgeattr,
            t=t,
            num_nodes=num_nodes,
        )

        new_transmission = self.propagate(
            agents_data.edge_index,
            x=agents_data.x,
            edge_attr=agents_data.edge_attr,
            t=agents_data.t,
            R=R,
            SFSusceptibility=SFSusceptibility,
            SFInfector=SFInfector,
            lam_gamma_integrals=all_lam_gamma.squeeze(),
        )

        prob_not_infected = torch.exp(-1 * new_transmission)
        # prob_infected = will_isolate*(1 - prob_not_infected)
        probs = torch.hstack((1 - prob_not_infected, prob_not_infected))

        # Gumbel softmax logic
        potentially_exposed_today = self.st_bernoulli(probs)[:, 0].to(
            self.device
        )  # using straight-through bernoulli
        # potentially_exposed_today = potentially_exposed_today * (
        #     1.0 - will_isolate.squeeze()
        # )

        # nvidia_smi.nvmlInit()
        # deviceCount = nvidia_smi.nvmlDeviceGetCount()

        # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        # util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        # mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        # print(f"Mem Free: {mem.free/1024**2:5.2f}MB / {mem.total/1024**2:5.2f}MB | gpu-util: {util.gpu/100.0:3.1%} | gpu-mem: {util.memory/100.0:3.1%} |")

        newly_exposed_today = (
            current_stages == self.SUSCEPTIBLE_VAR
        ).squeeze() * potentially_exposed_today

        daily_infected = daily_infected + newly_exposed_today.sum() * time_step_one_hot

        daily_infected = daily_infected.squeeze(0)

        # Update protein features for newly infected agents
        # (average infected neighbors' embeddings via message passing)
        newly_exposed_mask = newly_exposed_today.unsqueeze(1)
        state = self._update_protein_features_for_newly_infected(
            state, newly_exposed_mask, all_edgelist, current_stages
        )

        updated_stages = self.update_stages(
            t, current_stages, agents_next_stage_times, newly_exposed_mask
        )
        updated_next_stage_times = self.update_transition_times(
            t, agents_next_stage_times, newly_exposed_mask, current_stages
        )

        updated_infected_times = self.update_infected_times(
            t, agents_infected_time, newly_exposed_mask
        )

        # num_vaccines = self.calibrate_num_vaccines.to(self.device)
        # print(num_vaccines)
        # print(R_tensor)

        # updated_stages = self.recover_random_agents(updated_stages, int(num_vaccines[0].item()))

        # self.get_stage_proportions(t, updated_stages)

        return {
            self.output_variables[0]: updated_stages,
            self.output_variables[1]: updated_next_stage_times,
            self.output_variables[2]: updated_infected_times,
            self.output_variables[3]: daily_infected,
        }
