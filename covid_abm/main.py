import argparse
import torch
from tqdm import trange
from .simulator import get_registry, get_runner
from agent_torch.core.helpers import read_config

# --------------------------
#  Parse arguments
# --------------------------

parser = argparse.ArgumentParser(
    description="AgentTorch: million-scale, differentiable agent-based models"
)

parser.add_argument(
    "-c",
    "--config",
    default="yamls/config.yaml",
    help="config file with simulation parameters",
)

# NEW optional arguments
parser.add_argument("--state", type=str, help="Override the state")
parser.add_argument("--county", type=str, help="Override the county FIPS code")

args = parser.parse_args()

# --------------------------
#  Load config
# --------------------------

config_file = args.config
config = read_config(config_file)

# Apply overrides ONLY if the user provided them
if args.state is not None:
    config["simulation_metadata"]["state"] = args.state

if args.county is not None:
    # Adjust directory automatically based on your population structure
    config["simulation_metadata"]["county"] = args.county
    config["simulation_metadata"]["population_dir"] = (
        f"/home/namishah/alex/gen-cov-abm/populations/pop{args.county}"
    )

registry = get_registry()
runner = get_runner(config, registry)

# --------------------------
#  Run simulation
# --------------------------

device = torch.device(runner.config["simulation_metadata"]["device"])
num_episodes = runner.config["simulation_metadata"]["num_episodes"]
num_steps_per_episode = runner.config["simulation_metadata"]["num_steps_per_episode"]

print(":: preparing simulation...")

runner.init()

for episode in trange(num_episodes, desc=":: running episodes"):
    runner.step(num_steps_per_episode)
    runner.reset()

print(":: finished execution")
