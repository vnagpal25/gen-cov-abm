"""
Generate ESM-2 embeddings specified proteins.

This script loads protein sequences from ma_sequences.csv and generates
embeddings using the ESM-2 (650M parameter) model. The embeddings are
saved in multiple formats for downstream analysis.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
# Add src directory to path for imports
SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SRC_DIR))


import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import esm

from utils.path_utils import get_data_dir


def remove_stop_codon(sequence):
    """Remove stop codon (*) from end of sequence if present."""
    if sequence[-1] == "*":
        return sequence[:-1]
    return sequence


def generate_embeddings(
    seq_df, seq_column, model, batch_converter, device, batch_size=32
):
    """
    Generate embeddings for sequences in a specified column.

    Args:
        seq_df: DataFrame containing sequences
        seq_column: Name of the column containing sequences
        model: ESM-2 model
        batch_converter: Batch converter from alphabet
        device: Torch device (cpu/cuda/mps)
        batch_size: Number of sequences to process per batch

    Returns:
        embeddings_array: NumPy array of embeddings
        ids: List of sequence IDs
    """
    all_embeddings = []
    all_ids = []

    # Clean sequences
    seq_df_clean = seq_df.copy()
    seq_df_clean[seq_column] = seq_df_clean[seq_column].apply(remove_stop_codon)

    print(f"Processing {len(seq_df_clean)} sequences from '{seq_column}' column...")

    for i in tqdm(range(0, len(seq_df_clean), batch_size)):
        batch_df = seq_df_clean.iloc[i : i + batch_size]

        # Prepare batch data
        data = [(row["name"], row[seq_column]) for _, row in batch_df.iterrows()]

        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        # Get embeddings
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]

        # Average across residues (excluding special tokens)
        sequence_repr = token_representations.mean(1)

        # Store results
        all_embeddings.append(sequence_repr.cpu().numpy())
        all_ids.extend(batch_labels)

    # Concatenate all embeddings
    embeddings_array = np.vstack(all_embeddings)
    print(f"Final embeddings shape for '{seq_column}': {embeddings_array.shape}")

    return embeddings_array, all_ids


def save_embeddings(embeddings, ids, output_dir, prefix):
    # Save embeddings in multiple formats.

    # Save as numpy array
    npy_path = output_dir / f"{prefix}_embeddings.npy"
    np.save(npy_path, embeddings)
    print(f"Saved to {npy_path}")

    # Save IDs
    ids_path = output_dir / f"{prefix}_ids.txt"
    with open(ids_path, "w") as f:
        f.write("\n".join(ids))
    print(f"Saved IDs to {ids_path}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate ESM-2 embeddings for protein sequences"
    )
    parser.add_argument(
        "proteins",
        nargs="+",
        help="List of protein column names to generate embeddings for (e.g., n_sequence s_sequence)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    # Load sequence data
    seq_file_path = get_data_dir() / "genomic_data" / "all_sequences.csv"
    seq_df = pd.read_csv(seq_file_path)
    print(f"Loaded {len(seq_df)} sequences from: {seq_file_path}")

    # Load ESM-2 model
    print("Loading ESM-2 model...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # Disable dropout

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    print(f"Model loaded successfully on device: {device}")

    # Output directory
    output_dir = get_data_dir() / "genomic_data"

    # Generate embeddings for each protein in the list
    for protein_name in args.proteins:
        print(f"Processing protein: {protein_name}")

        embeddings, ids = generate_embeddings(
            seq_df, protein_name, model, batch_converter, device, args.batch_size
        )

        # Save embeddings
        save_embeddings(embeddings, ids, output_dir, protein_name)

    print("All embeddings generated successfully!")


if __name__ == "__main__":
    main()
