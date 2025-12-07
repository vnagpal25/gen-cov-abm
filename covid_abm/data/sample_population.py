from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


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


def load_series(path: Path) -> pd.Series:
    series = pd.read_pickle(path)
    if not isinstance(series, pd.Series):
        raise ValueError(f"{path} did not contain a pandas Series.")
    return series


def sample_indices(pop_size: int, target_size: int, seed: int) -> np.ndarray:
    if target_size > pop_size:
        raise ValueError(
            f"Requested sample of {target_size} exceeds population size {pop_size}."
        )
    rng = np.random.default_rng(seed)
    sampled = rng.choice(pop_size, size=target_size, replace=False)
    sampled.sort()
    return sampled


def write_age_files(
    src_csv: Path,
    src_pickle: Path,
    dest_dir: Path,
    indices: np.ndarray,
) -> None:
    ages = pd.read_csv(src_csv)
    sampled_ages = ages.iloc[indices].reset_index(drop=True)
    sampled_ages.to_csv(dest_dir / "age.csv", index=False)

    age_bins = load_series(src_pickle)
    sampled_bins = age_bins.iloc[indices].reset_index(drop=True)
    sampled_bins.to_pickle(dest_dir / "age.pickle")


def write_disease_stages(src_csv: Path, dest_dir: Path, indices: np.ndarray) -> None:
    stages = pd.read_csv(src_csv)
    sampled_stages = stages.iloc[indices].reset_index(drop=True)
    sampled_stages.to_csv(dest_dir / "disease_stages.csv", index=False)


def sample_edges(
    edge_path: Path,
    dest_path: Path,
    selected_indices: np.ndarray,
) -> None:
    selected_set = set(int(i) for i in selected_indices.tolist())
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices.tolist())}

    chunks = []
    for chunk in pd.read_csv(
        edge_path,
        header=None,
        names=["src", "dst"],
        dtype=np.int64,
        chunksize=500_000,
    ):
        mask = chunk["src"].isin(selected_set) & chunk["dst"].isin(selected_set)
        if not mask.any():
            continue
        filtered = chunk.loc[mask].copy()
        filtered["src"] = filtered["src"].map(index_map)
        filtered["dst"] = filtered["dst"].map(index_map)
        chunks.append(filtered)

    if chunks:
        sampled = pd.concat(chunks, ignore_index=True)
    else:
        sampled = pd.DataFrame(columns=["src", "dst"], dtype=np.int64)

    sampled.to_csv(dest_path, header=False, index=False)


def sample_population(
    county: str,
    base_dir: Path,
    suffix: str,
    target_size: int,
    seed: int,
    overwrite: bool,
) -> Path:
    src_dir = base_dir / f"pop{county}"
    if not src_dir.exists():
        raise FileNotFoundError(f"Missing source population directory {src_dir}")

    dest_dir = base_dir / f"pop{county}_{suffix}"
    if dest_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{dest_dir} already exists. Pass --overwrite to rebuild the sample."
            )
        shutil.rmtree(dest_dir)
    (dest_dir / "mobility_networks").mkdir(parents=True, exist_ok=True)

    age_csv = src_dir / "age.csv"
    stage_csv = src_dir / "disease_stages.csv"
    age_pickle = src_dir / "age.pickle"
    edges_csv = src_dir / "mobility_networks" / "0.csv"

    age_bins = load_series(age_pickle)
    pop_size = len(age_bins)
    indices = sample_indices(pop_size, target_size, seed)

    write_age_files(age_csv, age_pickle, dest_dir, indices)
    write_disease_stages(stage_csv, dest_dir, indices)
    shutil.copy(src_dir / "mapping.json", dest_dir / "mapping.json")

    sample_edges(edges_csv, dest_dir / "mobility_networks" / "0.csv", indices)
    return dest_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create downsampled MA county populations."
    )
    parser.add_argument(
        "--counties",
        nargs="*",
        default=COUNTIES,
        help="FIPS codes to process (default: all MA counties).",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=30000,
        help="Number of agents to keep per county.",
    )
    parser.add_argument(
        "--suffix",
        default="sample30000",
        help="Suffix for the sampled population directories.",
    )
    parser.add_argument(
        "--base_dir",
        default="populations",
        help="Root directory containing pop{FIPS} folders.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base random seed for sampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing sampled populations.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    for idx, county in enumerate(args.counties):
        seed = args.seed + idx
        dest = sample_population(
            county=county,
            base_dir=base_dir,
            suffix=args.suffix,
            target_size=args.target_size,
            seed=seed,
            overwrite=args.overwrite,
        )
        print(f"[OK] {county}: wrote sample to {dest}")


if __name__ == "__main__":
    main()

