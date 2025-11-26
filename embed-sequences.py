"""
Generate ESM-2 embeddings for N-sequence and S-sequence data.

This script loads protein sequences from ma_sequences.csv and generates
embeddings using the ESM-2 (650M parameter) model. The embeddings are
saved in multiple formats for downstream analysis.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
print(f"Project root: {ROOT}")

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import esm

from src.utils.path_utils import get_data_dir


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
    """
    Save embeddings in multiple formats.

    Args:
        embeddings: NumPy array of embeddings
        ids: List of sequence IDs
        output_dir: Directory to save files
        prefix: Prefix for output files (e.g., 'n_sequence', 's_sequence')
    """
    print(f"\nSaving {prefix} embeddings...")

    # Save as numpy array
    npy_path = output_dir / f"{prefix}_embeddings.npy"
    np.save(npy_path, embeddings)
    print(f"Saved to {npy_path}")

    # Save IDs
    ids_path = output_dir / f"{prefix}_ids.txt"
    with open(ids_path, "w") as f:
        f.write("\n".join(ids))
    print(f"Saved IDs to {ids_path}")

    # Save as CSV
    csv_path = output_dir / f"{prefix}_embeddings.csv"
    embedding_df = pd.DataFrame(embeddings, index=ids)
    embedding_df.to_csv(csv_path)
    print(f"Saved to {csv_path}")


def main():
    # Load sequence data
    seq_file_path = get_data_dir() / "ma_sequences.csv"
    print(f"Reading from: {seq_file_path}")
    seq_df = pd.read_csv(seq_file_path)
    print(f"Loaded {len(seq_df)} sequences")

    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # Disable dropout

    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    print("Model loaded successfully")
    print(f"Using device: {device}")

    # Generate embeddings
    batch_size = 32  # Adjust based on GPU memory

    print("Generating N-sequence embeddings...")
    n_embeddings, n_ids = generate_embeddings(
        seq_df, "n_sequence", model, batch_converter, device, batch_size
    )

    print("Generating S-sequence embeddings...")
    s_embeddings, s_ids = generate_embeddings(
        seq_df, "s_sequence", model, batch_converter, device, batch_size
    )

    # Save embeddings
    print("\n[4/4] Saving embeddings...")
    output_dir = get_data_dir()

    save_embeddings(n_embeddings, n_ids, output_dir, "n_sequence")
    save_embeddings(s_embeddings, s_ids, output_dir, "s_sequence")


if __name__ == "__main__":
    main()
