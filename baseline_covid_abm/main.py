import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import trange
from .simulator import get_registry, get_runner

from agent_torch.core.helpers import read_config

parser = argparse.ArgumentParser(
    description="AgentTorch: million-scale, differentiable agent-based models"
)
parser.add_argument(
    "-c",
    "--config",
    default="covid_abm/yamls/config.yaml",
    help="config file with simulation parameters",
)
parser.add_argument(
    "--ground_truth_file",
    default="data/daily_county_data/25001_daily_data.csv",
    help="CSV file containing ground truth time series for metrics (set empty to skip)",
)
parser.add_argument(
    "--truth_column",
    default="cases",
    help="Column in the ground truth file to compare against (e.g., cases or deaths)",
)
parser.add_argument(
    "--truth_offset",
    type=int,
    default=0,
    help="Row offset (in days) into the ground truth file to align with the simulation start",
)
parser.add_argument(
    "--truth_window",
    type=int,
    default=None,
    help="Optional override for the number of days to compare against (defaults to num_steps_per_episode)",
)
parser.add_argument(
    "--calib_params",
    default=None,
    help="Path to calibrated parameter tensor (.pt) to run in calibration mode.",
)

args = parser.parse_args()
config_file = args.config

config = read_config(config_file)
calib_tensors = None
if args.calib_params:
    import torch

    calib_tensors = torch.load(args.calib_params, map_location="cpu")
    config["simulation_metadata"]["calibration"] = True
print("\n================= DEBUG CONFIG =================")
print("Loaded from:", config_file)
import yaml
print(yaml.dump(config))
print("================================================\n")

registry = get_registry()
runner = get_runner(config, registry)

device = torch.device(runner.config["simulation_metadata"]["device"])
num_episodes = runner.config["simulation_metadata"]["num_episodes"]
num_steps_per_episode = runner.config["simulation_metadata"]["num_steps_per_episode"]
truth_window = args.truth_window or num_steps_per_episode

truth_values = None
if args.ground_truth_file:
    try:
        gt_df = pd.read_csv(args.ground_truth_file)
        if args.truth_column not in gt_df.columns:
            raise ValueError(
                f"Column '{args.truth_column}' not found in {args.ground_truth_file}"
            )
        truth_values = gt_df[args.truth_column].astype(float).to_numpy()
    except FileNotFoundError:
        print(
            f"Ground truth file '{args.ground_truth_file}' not found. "
            "Metrics will be skipped.",
            flush=True,
        )

print(":: preparing simulation...")

runner.init()

def apply_calibration():
    if calib_tensors is None:
        return
    r2_tensor = calib_tensors["calibrate_R2"].to(device)
    new_trans = runner.initializer.transition_function["0"]["new_transmission"]
    new_trans.calibrate_R2 = r2_tensor
    seirm = runner.initializer.transition_function["1"]["seirm_progression"]
    if "calibrate_M" in calib_tensors:
        seirm.calibrate_M = calib_tensors["calibrate_M"].to(device)
    else:
        seirm.calibrate_M = torch.tensor([0.12], device=device)

apply_calibration()

for episode in trange(num_episodes, desc=":: running episodes"):
    for _ in range(num_steps_per_episode):
        runner.step(1)

    preds = (
        runner.state["environment"]["daily_infected"]
        .detach()
        .cpu()
        .numpy()
        .flatten()
    )

    if truth_values is not None:
        start_idx = args.truth_offset + episode * num_steps_per_episode
        end_idx = start_idx + truth_window
        if end_idx > len(truth_values):
            raise ValueError(
                f"Ground truth series ({len(truth_values)} rows) is too short for "
                f"episode {episode}. Needed up to index {end_idx}."
            )
        gt_slice = truth_values[start_idx:end_idx]
        nd = np.sum(np.abs(preds - gt_slice)) / (np.sum(np.abs(gt_slice)) + 1e-8)
        rmse = np.sqrt(np.mean((preds - gt_slice) ** 2))
        mae = np.mean(np.abs(preds - gt_slice))
        print(
            f"Episode {episode} metrics -> ND: {nd:.4f}, RMSE: {rmse:.4f}, "
            f"MAE: {mae:.4f}",
            flush=True,
        )
    else:
        print(f"Episode {episode} metrics skipped (no ground truth provided).")

    runner.reset()
    apply_calibration()

print(":: finished execution")