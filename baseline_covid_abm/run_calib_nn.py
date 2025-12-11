import argparse
import copy
from pathlib import Path

import pandas as pd
import torch
from epiweeks import Week
from torch import nn

torch.autograd.set_detect_anomaly(True)

from agent_torch.core.helpers import read_config
from covid_abm.simulator import get_registry, get_runner
from covid_abm.calibration.calibnn import CalibNN
from covid_abm.calibration.utils.data import (
    get_dataloader,
    NN_INPUT_WEEKS,
)
from covid_abm.calibration.utils.feature import Feature
from covid_abm.calibration.utils.neighborhood import Neighborhood
from covid_abm.utils.windows import WindowSpec, derive_windows


COUNTIES = [
    "25001",
    "25003",
    "25005",
    "25009",
    "25013",
    "25021",
    "25023",
    "25025",
    "25027",
]


def infer_population_size(pop_prefix: str) -> int:
    age_pickle = Path(f"{pop_prefix}/age.pickle")
    if not age_pickle.exists():
        raise FileNotFoundError(f"{age_pickle} not found.")
    series = pd.read_pickle(age_pickle)
    if not isinstance(series, pd.Series):
        raise ValueError(f"{age_pickle} did not contain a pandas Series.")
    return len(series)


def prepare_config(
    base_config: dict, county: str, population_suffix: str, start_week: int
) -> dict:
    cfg = copy.deepcopy(base_config)
    pop_prefix = f"populations/pop{county}_{population_suffix}"
    if not Path(pop_prefix).exists():
        raise FileNotFoundError(
            f"Population directory {pop_prefix} not found. Run the sampler first."
        )
    pop_size = infer_population_size(pop_prefix)
    cfg["simulation_metadata"]["population_dir"] = pop_prefix
    cfg["simulation_metadata"]["age_group_file"] = f"{pop_prefix}/age.pickle"
    cfg["simulation_metadata"]["disease_stage_file"] = f"{pop_prefix}/disease_stages.csv"
    cfg["simulation_metadata"][
        "infection_network_file"
    ] = f"{pop_prefix}/mobility_networks/0.csv"
    cfg["simulation_metadata"]["mapping_path"] = f"{pop_prefix}/mapping.json"
    cfg["simulation_metadata"]["num_agents"] = pop_size
    cfg["simulation_metadata"]["calibration"] = True
    cfg["simulation_metadata"]["START_WEEK"] = start_week

    try:
        age_prop = cfg["state"]["agents"]["citizens"]["properties"]["age"]
        age_prop["initialization_function"]["arguments"]["file_path"]["value"] = cfg[
            "simulation_metadata"
        ]["age_group_file"]
    except KeyError:
        pass

    try:
        disease_prop = cfg["state"]["agents"]["citizens"]["properties"]["disease_stage"]
        disease_prop["initialization_function"]["arguments"]["file_path"]["value"] = cfg[
            "simulation_metadata"
        ]["disease_stage_file"]
    except KeyError:
        pass

    try:
        mi_args = cfg["environment"]["mean_interactions"]["initialization_function"][
            "arguments"
        ]["file_path"]
        mi_args["value"] = cfg["simulation_metadata"]["age_group_file"]
    except KeyError:
        pass

    try:
        cfg["network"]["agent_agent"]["infection_network"]["arguments"]["file_path"] = cfg[
            "simulation_metadata"
        ]["infection_network_file"]
    except KeyError:
        pass

    return cfg


def get_neighborhood_by_fips(fips: str) -> Neighborhood:
    for neighborhood in Neighborhood:
        if neighborhood.nta_id == fips:
            return neighborhood
    raise ValueError(f"No Neighborhood entry for FIPS {fips}")


def load_features(county: str, start_week: int, num_weeks: int, device: torch.device):
    neighborhood = get_neighborhood_by_fips(county)
    epiweek_start = Week.fromstring(str(start_week))
    feature_list = [
        Feature.RETAIL_CHANGE,
        Feature.GROCERY_CHANGE,
        Feature.PARKS_CHANGE,
        Feature.TRANSIT_CHANGE,
        Feature.WORK_CHANGE,
        Feature.RESIDENTIAL_CHANGE,
        Feature.CASES,
        Feature.CASES_4WK_AVG,
    ]
    dataloader = get_dataloader(neighborhood, epiweek_start, num_weeks, feature_list)
    metadata, features = next(iter(dataloader))
    return metadata.to(device), features.to(device)


def load_ground_truth(county: str, offset: int, steps: int, column: str) -> torch.Tensor:
    daily_path = Path("data/daily_county_data") / f"{county}_daily_data.csv"
    df = pd.read_csv(daily_path)
    series = df[column].astype(float)
    slice_arr = series.iloc[offset : offset + steps].to_numpy()
    return torch.tensor(slice_arr, dtype=torch.float32)


def assign_calibration_tensors(runner, r2_values: torch.Tensor, mortality: torch.Tensor):
    new_trans = runner.initializer.transition_function["0"]["new_transmission"]
    seirm = runner.initializer.transition_function["1"]["seirm_progression"]
    new_trans.calibrate_R2 = r2_values.unsqueeze(1)
    seirm.calibrate_M = mortality


def train_county_window(
    county: str,
    window: WindowSpec,
    base_config: dict,
    truth_column: str,
    epochs: int,
    lr: float,
    device: torch.device,
    output_dir: Path,
    population_suffix: str,
):
    num_steps = base_config["simulation_metadata"]["num_steps_per_episode"]
    num_weeks = base_config["simulation_metadata"]["NUM_WEEKS"]

    cfg = prepare_config(base_config, county, population_suffix, window.start_week)
    registry = get_registry()
    runner = get_runner(cfg, registry)
    runner.init()

    metadata, features = load_features(county, window.start_week, num_weeks, device)
    gt = load_ground_truth(county, window.offset, num_steps, truth_column).to(device)

    model = CalibNN(
        metas_train_dim=len(Neighborhood),
        X_train_dim=features.shape[-1],
        device=device,
        training_weeks=num_weeks,
        out_dim=1,
        scale_output="abm-covid",
    ).to(device)
    mortality_param = nn.Parameter(torch.tensor([0.12], dtype=torch.float32, device=device))

    optimizer = torch.optim.Adam(
        list(model.parameters()) + [mortality_param],
        lr=lr,
    )
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        params = model(features, metadata).squeeze(-1)
        runner.reset()
        assign_calibration_tensors(runner, params, mortality_param)
        runner.step(num_steps)
        preds = (
            runner.state_trajectory[-1][-1]["environment"]["daily_infected"]
            .squeeze()
            .to(device)
        )
        target = gt.to(preds.dtype)
        loss = loss_fn(preds, target)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(
                f"[Train] County {county} window {window.label} "
                f"epoch {epoch+1}/{epochs} loss={loss.item():.4f}"
            )

    runner.reset()
    assign_calibration_tensors(runner, params.detach(), mortality_param.detach())
    runner.step(num_steps)
    final_preds = runner.state_trajectory[-1][-1]["environment"]["daily_infected"].squeeze()
    runner.reset()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{county}_{window.label}.pt"
    torch.save(
        {
            "calibrate_R2": params.detach().cpu().unsqueeze(1),
            "calibrate_M": mortality_param.detach().cpu(),
            "county": county,
            "window": window.label,
        },
        output_path,
    )
    print(f"[SAVE] Stored calibration params at {output_path}")

    return final_preds, gt.cpu()


def main():
    parser = argparse.ArgumentParser(description="Train CalibNN for MA counties/windows.")
    parser.add_argument(
        "--base_config",
        default="covid_abm/yamls/config_base.yaml",
        help="Template config path.",
    )
    parser.add_argument(
        "--truth_column",
        default="cases",
        choices=["cases", "deaths"],
        help="Ground truth column to match.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per window.")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate.")
    parser.add_argument(
        "--output_dir",
        default="Results/calib_params",
        help="Directory to store calibrated parameter tensors.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Computation device (cpu or cuda). Note: runner currently uses CPU tensors.",
    )
    parser.add_argument(
        "--start_date",
        default="2020-06-01",
        help="Inclusive start date for calibration windows (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end_date",
        default="2021-07-31",
        help="Inclusive end date for calibration windows (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--population_suffix",
        default="sample30000",
        help="Population directory suffix (e.g., sample30000).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    base_config = read_config(args.base_config)
    output_dir = Path(args.output_dir)
    window_days = base_config["simulation_metadata"]["num_steps_per_episode"]

    daily_dir = Path("data/daily_county_data")
    base_daily = None
    for county in COUNTIES:
        candidate = daily_dir / f"{county}_daily_data.csv"
        if candidate.exists():
            base_daily = candidate
            break
    if base_daily is None:
        raise FileNotFoundError("Could not find any daily county data files.")
    windows = derive_windows(base_daily, args.start_date, args.end_date, window_days)
    if not windows:
        raise RuntimeError("No calibration windows generated for the requested range.")

    for county in COUNTIES:
        for window in windows:
            print(f"\n=== Training CalibNN for county {county}, window {window.label} ===")
            train_county_window(
                county=county,
                window=window,
                base_config=base_config,
                truth_column=args.truth_column,
                epochs=args.epochs,
                lr=args.lr,
                device=device,
                output_dir=output_dir,
                population_suffix=args.population_suffix,
            )


if __name__ == "__main__":
    main()
