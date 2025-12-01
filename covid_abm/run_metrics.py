import argparse
import copy
import os
import re
import subprocess
import tempfile
from pathlib import Path

import yaml


COUNTIES = [
    "25001",
    "25003",
    "25005",
    "25009",
    "25013",
    "25015",
    "25021",
    "25023",
    "25025",
    "25027",
]

WINDOWS = [
    {"label": "202027-202031", "offset": 30, "start_week": 202027},
    {"label": "202031-202035", "offset": 65, "start_week": 202031},
]

METRIC_PATTERN = re.compile(
    r"ND:\s*([0-9.+-eE]+),\s*RMSE:\s*([0-9.+-eE]+),\s*MAE:\s*([0-9.+-eE]+)"
)


def load_config(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_temp_config(config: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="covid_config_", delete=False
    )
    yaml.safe_dump(config, tmp)
    tmp.close()
    return Path(tmp.name)


def run_simulation(
    config_path: Path,
    ground_truth: Path,
    offset: int,
    truth_column: str,
    calib_path: Path | None = None,
):
    cmd = [
        "python",
        "-m",
        "covid_abm.main",
        "-c",
        str(config_path),
        "--ground_truth_file",
        str(ground_truth),
        "--truth_offset",
        str(offset),
        "--truth_column",
        truth_column,
    ]
    if calib_path is not None:
        cmd.extend(["--calib_params", str(calib_path)])
    result = subprocess.run(cmd, capture_output=True, text=True)
    stdout = result.stdout
    stderr = result.stderr
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}).\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )
    match = METRIC_PATTERN.search(stdout)
    if not match:
        raise RuntimeError(f"Could not parse metrics from output:\n{stdout}")
    nd, rmse, mae = map(float, match.groups())
    return nd, rmse, mae, stdout


def prepare_config(base_config: dict, county: str) -> dict:
    cfg = copy.deepcopy(base_config)
    pop_prefix = f"populations/pop{county}_sample5000"
    cfg["simulation_metadata"]["population_dir"] = pop_prefix
    cfg["simulation_metadata"]["age_group_file"] = f"{pop_prefix}/age.pickle"
    cfg["simulation_metadata"]["disease_stage_file"] = f"{pop_prefix}/disease_stages.csv"
    cfg["simulation_metadata"][
        "infection_network_file"
    ] = f"{pop_prefix}/mobility_networks/0.csv"
    cfg["simulation_metadata"]["mapping_path"] = f"{pop_prefix}/mapping.json"
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


def main():
    parser = argparse.ArgumentParser(
        description="Run covid_abm.main across multiple counties/time windows."
    )
    parser.add_argument(
        "--base_config",
        default="covid_abm/yamls/config_base.yaml",
        help="Template config to copy per county.",
    )
    parser.add_argument(
        "--truth_column",
        default="cases",
        help="Ground truth column to compare (cases or deaths).",
    )
    parser.add_argument(
        "--output_csv",
        default="Results/metrics_summary.csv",
        help="Path to write aggregated metrics CSV.",
    )
    parser.add_argument(
        "--use_calib",
        action="store_true",
        help="If set, load calibrated parameter tensors from --calib_dir.",
    )
    parser.add_argument(
        "--calib_dir",
        default="Results/calib_params",
        help="Directory containing {county}_{window}.pt calibration tensors.",
    )
    args = parser.parse_args()

    base_config_path = Path(args.base_config)
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config {base_config_path} not found.")

    base_config = load_config(base_config_path)
    results = []

    calib_dir = Path(args.calib_dir)

    for county in COUNTIES:
        gt_file = Path(f"data/daily_county_data/{county}_daily_data.csv")
        if not gt_file.exists():
            print(f"[WARN] Ground truth file missing for {county}, skipping.")
            continue
        cfg = prepare_config(base_config, county)
        tmp_cfg_path = write_temp_config(cfg)
        try:
            for window in WINDOWS:
                try:
                    calib_path = None
                    if args.use_calib:
                        calib_path = calib_dir / f"{county}_{window['label']}.pt"
                        if not calib_path.exists():
                            print(
                                f"[WARN] Missing calibration file {calib_path}, skipping calibrated run."
                            )
                            continue
                    nd, rmse, mae, stdout = run_simulation(
                        tmp_cfg_path,
                        gt_file,
                        window["offset"],
                        args.truth_column,
                        calib_path=calib_path,
                    )
                    results.append(
                        {
                            "county": county,
                            "window": window["label"],
                            "truth_offset": window["offset"],
                            "ND": nd,
                            "RMSE": rmse,
                            "MAE": mae,
                        }
                    )
                    print(
                        f"[OK] County {county}, window {window['label']}: "
                        f"ND={nd:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}"
                    )
                except Exception as run_err:
                    print(
                        f"[ERR] County {county}, window {window['label']} failed: {run_err}"
                    )
        finally:
            os.remove(tmp_cfg_path)

    if not results:
        print("No results collected.")
        return

    import csv

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["county", "window", "truth_offset", "ND", "RMSE", "MAE"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved metrics to {output_path}")

    from collections import defaultdict

    aggregates = defaultdict(lambda: {"ND": [], "RMSE": [], "MAE": []})
    for row in results:
        agg = aggregates[row["window"]]
        agg["ND"].append(row["ND"])
        agg["RMSE"].append(row["RMSE"])
        agg["MAE"].append(row["MAE"])

    print("\nAggregate metrics by window:")
    for window in WINDOWS:
        label = window["label"]
        agg = aggregates.get(label)
        if not agg:
            print(f"- {label}: no data")
            continue
        nd_avg = sum(agg["ND"]) / len(agg["ND"])
        rmse_avg = sum(agg["RMSE"]) / len(agg["RMSE"])
        mae_avg = sum(agg["MAE"]) / len(agg["MAE"])
        print(
            f"- {label}: ND={nd_avg:.4f}, RMSE={rmse_avg:.4f}, MAE={mae_avg:.4f} "
            f"(n={len(agg['ND'])})"
        )


if __name__ == "__main__":
    main()

