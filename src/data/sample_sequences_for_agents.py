"""
Sample genomic sequences for agents by county.

This script:
1. Loads and filters genomic sequences for Massachusetts counties
2. Loads ORF1a sequence embeddings
3. Normalizes embeddings to unit vectors
4. Saves individual embeddings by county
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# County name to FIPS code mapping
COUNTY_CODE_MAP = {
    "Barnstable County": 25001,
    "Suffolk County MA": 25025,
    "Bristol County": 25005,
    "Norfolk County": 25021,
    "Plymouth County": 25023,
    "Worcester County": 25027,
    "Essex County MA": 25009,
    "Hampden County": 25013,
    "Berkshire County": 25003,
    "Hampshire County": 25015,
}


def load_and_filter_sequences(sequences_csv_path: Path) -> pd.DataFrame:
    # load sequences from csv and filter to massachusetts
    df = pd.read_csv(sequences_csv_path)

    # Filter for Massachusetts and specified counties
    df = df[
        (df["division_exposure"] == "Massachusetts")
        & (df["location"].isin(COUNTY_CODE_MAP.keys()))
    ]

    # summary stats
    print(f"{len(df)} genomes in Massachusetts")
    print("\nSequences by county:")
    print(df["location"].value_counts())

    return df


def load_sequence_ids(sequence_ids_path: Path) -> tuple[list[str], dict[str, int]]:
    # load sequence ids and return an index mapping
    with open(sequence_ids_path, "r") as f:
        sequence_ids = [line.strip() for line in f]

    id_to_index = {seq_id: idx for idx, seq_id in enumerate(sequence_ids)}

    return sequence_ids, id_to_index


def extract_genome_info(
    df: pd.DataFrame, id_to_index: dict[str, int]
) -> tuple[list[int], list[str], list[str]]:
    # return genome indices, names, and county names as seperated lists
    genome_indices = []
    genome_names = []
    counties = []

    for _, row in df.iterrows():
        genome_indices.append(id_to_index[row["name"]])
        genome_names.append(row["name"])
        counties.append(row["location"])

    return genome_indices, genome_names, counties


def load_and_filter_embeddings(
    embeddings_path: Path, genome_indices: list[int]
) -> np.ndarray:
    # load embeddings and filter to only include ones in MA
    all_embeddings = np.load(embeddings_path)
    selected_embeddings = all_embeddings[genome_indices]

    print(
        f"Loaded {len(selected_embeddings)} selected embeddings of dimension {selected_embeddings.shape[1]}"
    )

    return selected_embeddings


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    # take unit norm of all vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms

    return normalized_embeddings


def save_embeddings_by_county(
    genome_names: list[str],
    embeddings: np.ndarray,
    counties: list[str],
    output_dir: Path,
    verbose: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    county_counts = {}

    for genome_name, embedding, county in zip(genome_names, embeddings, counties):
        # Get county FIPS code
        county_code = COUNTY_CODE_MAP[county]

        # Create county directory if it doesn't exist
        county_dir = output_dir / str(county_code)
        county_dir.mkdir(exist_ok=True)

        # Save individual embedding file
        filename = genome_name.replace("/", "_").replace(" ", "_")
        filepath = county_dir / f"{filename}.npy"
        np.save(filepath, embedding)

        # Track counts
        county_counts[county] = county_counts.get(county, 0) + 1

        if verbose:
            print(f"For {county}, saved to {filepath}")

    print("\nSummary of saved embeddings by county:")
    for county, count in sorted(county_counts.items()):
        print(f"  {county}: {count} embeddings")


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    data_dir = project_root / "data" / "genomic_data"

    # Step 1: Load and filter sequences
    print("Step 1: Loading and filtering sequences...")
    df = load_and_filter_sequences(data_dir / "all_sequences.csv")

    # Step 2: Load sequence IDs and create mapping
    print("Step 2: Loading sequence IDs...")
    sequence_ids, id_to_index = load_sequence_ids(data_dir / "orf1a_sequence_ids.txt")
    print(f"Loaded {len(sequence_ids)} sequence IDs")

    # Step 3: Extract genome information
    print("Step 3: Extracting genome information...")
    genome_indices, genome_names, counties = extract_genome_info(df, id_to_index)

    # Step 4: Load and filter embeddings
    print("Step 4: Loading embeddings...")
    selected_embeddings = load_and_filter_embeddings(
        data_dir / "orf1a_sequence_embeddings.npy", genome_indices
    )

    # Step 5: Normalize embeddings
    print("Step 5: Normalizing embeddings...")
    normalized_embeddings = normalize_embeddings(selected_embeddings)

    # Step 6: Save embeddings by county
    print(f"Step 6: Saving embeddings to 'embeddings' folder...")
    save_embeddings_by_county(
        genome_names,
        normalized_embeddings,
        counties,
        Path("embeddings"),
    )

    print("Processing complete")


if __name__ == "__main__":
    main()
