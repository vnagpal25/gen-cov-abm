import argparse
import csv
import math
import random
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.types import Device
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import wandb
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)


class CrossAttentionFusion(nn.Module):
    """
    cross-attention to fuse multiple protein embeddings.

    Takes embeddings from multiple proteins (S, N, ORF1a) and fuses them using
    cross-attention with learnable query vectors.
    """

    def __init__(self, embedding_dim: int, num_queries: int = 8, num_heads: int = 8):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_queries = num_queries
        self.num_heads = num_heads

        # Input layer normalization (normalize embeddings before fusion)
        self.input_norm = nn.LayerNorm(embedding_dim)

        # Learnable query vectors
        self.queries = nn.Parameter(torch.randn(num_queries, embedding_dim))

        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, batch_first=True
        )

        # Layer normalization (after attention)
        self.norm = nn.LayerNorm(embedding_dim)

        # Pooling to get final fused embedding
        self.pool = nn.Linear(num_queries * embedding_dim, embedding_dim)

    def forward(self, protein_embeddings: List[torch.Tensor]) -> torch.Tensor:
        batch_size = protein_embeddings[0].size(0)

        # Normalize each protein embedding before fusion
        normalized_embeddings = [self.input_norm(emb) for emb in protein_embeddings]

        # Stack protein embeddings to create key-value pairs
        # Shape: (B, num_proteins, D)
        kv = torch.stack(normalized_embeddings, dim=1)

        # Expand queries for batch
        # Shape: (B, num_queries, D)
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply cross-attention
        # queries: (B, num_queries, D)
        # kv: (B, num_proteins, D) used as both keys and values
        attn_output, _ = self.cross_attention(queries, kv, kv)

        # Apply layer norm
        attn_output = self.norm(attn_output)

        # Flatten and pool to get final embedding
        # Shape: (B, num_queries * D) -> (B, D)
        attn_output = attn_output.reshape(batch_size, -1)
        return attn_output.squeeze(1)

        # fused = self.pool(attn_output)

        # return fused


class GatedFusion(nn.Module):
    def __init__(self, embedding_dim: int, num_inputs: int = 3, hidden_dim: int = 512):
        super().__init__()

        self.num_inputs = num_inputs
        self.embedding_dim = embedding_dim

        # Linear transforms for each protein
        self.transforms = nn.ModuleList(
            [nn.Linear(embedding_dim, hidden_dim) for _ in range(num_inputs)]
        )

        # Gate that outputs 1 weight per input modality
        self.gate_layer = nn.Linear(hidden_dim * num_inputs, num_inputs)

        # Final mixing projection
        self.output_layer = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, protein_embeddings: List[torch.Tensor]):
        """
        protein_embeddings: [emb_s, emb_n, emb_orf1a]
        each shape: (B, D)
        """

        B = protein_embeddings[0].size(0)

        # Step 1: project each modality independently
        projected = [
            F.relu(self.transforms[i](emb)) for i, emb in enumerate(protein_embeddings)
        ]
        # list of (B, hidden_dim)

        # Step 2: gating network
        concat = torch.cat(projected, dim=-1)  # (B, hidden_dim * num_inputs)

        gate_logits = self.gate_layer(concat)  # (B, num_inputs)
        gate = torch.softmax(gate_logits, dim=-1)  # normalized weights

        # Step 3: weighted sum over modalities
        fused = torch.zeros(B, projected[0].size(1), device=projected[0].device)
        for i, proj in enumerate(projected):
            fused += gate[:, i].unsqueeze(1) * proj

        # Step 4: project back to embedding_dim
        fused = self.output_layer(fused)  # (B, embedding_dim)

        return fused


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ClassificationHead(nn.Module):
    """Linear classification head on top of projection head"""

    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class CladeClassificationModel(nn.Module):
    """Combined model: CrossAttentionFusion + ProjectionHead + ClassificationHead"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_classes: int,
        num_queries: int = 8,
        num_heads: int = 8,
    ):
        super().__init__()
        # self.fusion = CrossAttentionFusion(input_dim, num_queries, num_heads)
        self.fusion = GatedFusion(embedding_dim=input_dim, num_inputs=3, hidden_dim=512)

        self.projection = ProjectionHead(input_dim, hidden_dim, out_dim)
        self.classifier = ClassificationHead(out_dim, num_classes)

    def forward(self, protein_embeddings: List[torch.Tensor], return_embeddings=False):
        """
        Forward pass through the model.

        Args:
            protein_embeddings: List of input embeddings for each protein [S, N, ORF1a]
                               Each tensor has shape (B, D)
            return_embeddings: If True, return only projection output (no classification)

        Returns:
            If return_embeddings=True: projected embeddings
            If return_embeddings=False: classification logits
        """
        # Fuse protein embeddings using cross-attention
        fused = self.fusion(protein_embeddings)

        # Project fused embedding
        embeddings = self.projection(fused)

        if return_embeddings:
            return embeddings
        return self.classifier(embeddings)


# -------------------------
# Data helpers
# -------------------------
class EmbeddingDataset(Dataset):
    """
    Holds precomputed per-genome embeddings for all proteins and clade labels.

    expected inputs:
      ids: list of genome ids
      embeddings_s: dict id->tensor (D) for S protein
      embeddings_n: dict id->tensor (D) for N protein
      embeddings_orf1a: dict id->tensor (D) for ORF1a protein
      clade_labels: dict id->int (clade label)
    """

    def __init__(
        self,
        ids: List[str],
        embeddings_s: Dict[str, torch.Tensor],
        embeddings_n: Dict[str, torch.Tensor],
        embeddings_orf1a: Dict[str, torch.Tensor],
        clade_labels: Dict[str, int],
    ):
        self.ids = ids
        self.embeddings_s = embeddings_s
        self.embeddings_n = embeddings_n
        self.embeddings_orf1a = embeddings_orf1a
        self.clade_labels = clade_labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        gid = self.ids[idx]
        return {
            "id": gid,
            "embedding_s": self.embeddings_s[gid].float(),
            "embedding_n": self.embeddings_n[gid].float(),
            "embedding_orf1a": self.embeddings_orf1a[gid].float(),
            "clade_label": self.clade_labels[gid],
            "idx": idx,
        }


def validate(
    model: CladeClassificationModel,
    dataloader: DataLoader,
    criterion: nn.Module,
):
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            emb_s = batch["embedding_s"].to(DEVICE)
            emb_n = batch["embedding_n"].to(DEVICE)
            emb_orf1a = batch["embedding_orf1a"].to(DEVICE)
            labels = batch["clade_label"].to(DEVICE)

            # Forward pass with all protein embeddings
            logits = model([emb_s, emb_n, emb_orf1a], return_embeddings=False)
            loss = criterion(logits, labels)

            # Get predictions
            predictions = torch.argmax(logits, dim=1)

            # Accumulate
            total_loss += loss.item() * emb_s.size(0)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_samples += emb_s.size(0)

    # Compute metrics
    avg_loss = total_loss / total_samples
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    f1_macro = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
    f1_weighted = f1_score(
        all_labels, all_predictions, average="weighted", zero_division=0
    )

    return avg_loss, accuracy, f1_macro, f1_weighted


def train_classification_model(
    dataset: EmbeddingDataset,
    emb_dim: int,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int = 512,
    out_dim: int = 256,
    num_queries: int = 8,
    num_heads: int = 8,
    val_batches: int = 8,
    wandb_project: str = "phylo-embedding-finetuning",
    wandb_run_name: str = None,
):
    """
    Train classification model for clade prediction using multi-protein fusion.

    Args:
        dataset: EmbeddingDataset containing the data for all proteins
        emb_dim: Dimension of input embeddings
        num_classes: Number of clade classes
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        hidden_dim: Hidden dimension of projection head MLP
        out_dim: Output dimension of projection head (embedding dimension)
        num_queries: Number of learnable queries for cross-attention
        num_heads: Number of attention heads
        val_batches: Number of batches to reserve for validation
        wandb_project: wandb project name
        wandb_run_name: wandb run name (optional)

    Returns:
        Trained model
    """

    # Split dataset into train and validation
    val_size = val_batches * batch_size
    train_size = len(dataset) - val_size

    # Ensure we have enough data
    if val_size >= len(dataset):
        raise ValueError(
            f"Validation size ({val_size}) is larger than dataset size ({len(dataset)})"
        )

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED)
    )

    print(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} validation")

    # Initialize wandb
    wandb.init(
        project=wandb_project,
        name=wandb_run_name or "multi_protein_fusion_classification",
        config={
            "proteins": ["S", "N", "ORF1a"],
            "fusion_method": "cross_attention",
            "num_queries": num_queries,
            "num_heads": num_heads,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "emb_dim": emb_dim,
            "hidden_dim": hidden_dim,
            "out_dim": out_dim,
            "num_classes": num_classes,
            "loss": "cross_entropy",
            "train_size": train_size,
            "val_size": val_size,
            "val_batches": val_batches,
        },
    )

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create model
    model = CladeClassificationModel(
        input_dim=emb_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_classes=num_classes,
        num_queries=num_queries,
        num_heads=num_heads,
    ).to(DEVICE)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Compute balanced class weights from training data
    y_train = []
    for i in range(len(train_dataset)):
        y_train.append(train_dataset[i]["clade_label"])
    y_train = np.array(y_train)

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    print(f"Class weights: {class_weights}")

    # Loss function (cross-entropy with balanced class weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    global_step = 0

    for epoch in range(epochs):
        # Training phase
        model.train()

        total_loss = 0.0
        all_train_predictions = []
        all_train_labels = []
        total_samples = 0

        # Progress bar
        pbar = tqdm(
            train_dataloader, desc=f"Epoch {epoch+1}/{epochs} (Multi-Protein Fusion)"
        )

        for step, batch in enumerate(pbar):
            # Get embeddings for all proteins
            emb_s = batch["embedding_s"].to(DEVICE)  # (B, D)
            emb_n = batch["embedding_n"].to(DEVICE)  # (B, D)
            emb_orf1a = batch["embedding_orf1a"].to(DEVICE)  # (B, D)
            labels = batch["clade_label"].to(DEVICE)  # (B,)

            # get classification logits from fused embeddings
            logits = model([emb_s, emb_n, emb_orf1a], return_embeddings=False)

            # Compute loss
            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute accuracy
            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == labels).sum().item()

            # Update metrics
            total_loss += loss.item() * emb_s.size(0)
            all_train_predictions.extend(predictions.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
            total_samples += emb_s.size(0)
            global_step += 1

            # Log to wandb every 100 steps
            if global_step % 100 == 0:
                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "train_accuracy": correct / emb_s.size(0),
                        "step": global_step,
                    }
                )

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{correct / emb_s.size(0):.4f}",
                    "step": global_step,
                }
            )

        # Epoch training metrics
        avg_train_loss = total_loss / total_samples
        train_accuracy = np.mean(
            np.array(all_train_predictions) == np.array(all_train_labels)
        )
        train_f1_macro = f1_score(
            all_train_labels, all_train_predictions, average="macro", zero_division=0
        )
        train_f1_weighted = f1_score(
            all_train_labels, all_train_predictions, average="weighted", zero_division=0
        )

        # # Validation phase
        # val_loss, val_accuracy, val_f1_macro, val_f1_weighted = validate(
        #     model, val_dataloader, criterion
        # )

        # Log epoch metrics to wandb
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_accuracy": train_accuracy,
                "train_f1_macro": train_f1_macro,
                "train_f1_weighted": train_f1_weighted,
                # "val_loss": val_loss,
                # "val_accuracy": val_accuracy,
                # "val_f1_macro": val_f1_macro,
                # "val_f1_weighted": val_f1_weighted,
                "step": global_step,
            }
        )

        print(
            f"[Epoch {epoch+1}/{epochs}] Multi-Protein Fusion\n"
            f"  Train - Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.4f}, "
            f"F1-Macro: {train_f1_macro:.4f}, F1-Weighted: {train_f1_weighted:.4f}\n"
            # f"  Val   - Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}, "
            # f"F1-Macro: {val_f1_macro:.4f}, F1-Weighted: {val_f1_weighted:.4f}"
        )

    wandb.finish()
    return model


def save_projected_embeddings(
    dataset: EmbeddingDataset,
    model: CladeClassificationModel,
    output_prefix: str = "data/finetuned",
):
    """
    Apply trained fusion + projection head to all embeddings and save them.
    Note: Uses fusion + projection head, NOT the classification layer.

    Args:
        dataset: EmbeddingDataset containing original embeddings for all proteins
        model: Trained CladeClassificationModel
        output_prefix: Prefix for output files (e.g., "data/finetuned")

    Saves:
        {output_prefix}_fused_embeddings.npy: Fused and projected embeddings
        {output_prefix}_ids.txt: Corresponding IDs
    """
    model.eval()

    projected = []
    ids_list = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset[idx]
            gid = item["id"]

            # Get original embeddings for all proteins
            emb_s = item["embedding_s"].to(DEVICE).unsqueeze(0)
            emb_n = item["embedding_n"].to(DEVICE).unsqueeze(0)
            emb_orf1a = item["embedding_orf1a"].to(DEVICE).unsqueeze(0)

            # Fuse and project using trained model
            proj_emb = model([emb_s, emb_n, emb_orf1a], return_embeddings=True)

            # Normalize to unit vector
            proj_emb = F.normalize(proj_emb, dim=-1)

            # Save
            projected.append(proj_emb.cpu().numpy())
            ids_list.append(gid)

    # Stack into array of size num_samples x out_dim
    projected = np.vstack(projected)

    # Save embeddings
    np.save(f"{output_prefix}_fused_embeddings.npy", projected)

    # Save IDs
    with open(f"{output_prefix}_ids.txt", "w") as f:
        for gid in ids_list:
            f.write(f"{gid}\n")

    print(f"Saved fused projected embeddings to {output_prefix}_fused_embeddings.npy")
    print(f"Saved IDs to {output_prefix}_ids.txt")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Data file arguments
    parser.add_argument(
        "--ids-file",
        type=str,
        default="data/ma_distance_matrix_node_names.txt",
        help="Path to file containing genome IDs (one per line)",
    )
    parser.add_argument(
        "--sequences-csv",
        type=str,
        default="data/ma_sequences.csv",
        help="Path to sequences CSV file containing clade information",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing the embeddings files (default: data/)",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--val-batches",
        type=int,
        default=8,
        help="Number of batches to reserve for validation",
    )

    # Model architecture
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension of projection head MLP",
    )
    parser.add_argument(
        "--out-dim",
        type=int,
        default=256,
        help="Output dimension of projection head",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=8,
        help="Number of learnable queries for cross-attention fusion",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads in cross-attention",
    )

    # Output and logging
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="data/finetuned",
        help="Prefix for output files",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="phylo-embedding-finetuning",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Weights & Biases run name (optional)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def load_clade_mapping(
    sequences_csv: str, ids: List[str]
) -> Tuple[Dict[str, int], Dict[int, str], int]:
    """
    Load clade information from sequences CSV and create mapping.

    Args:
        sequences_csv: Path to CSV file with clade information
        ids: List of genome IDs to include

    Returns:
        Tuple of (clade_to_idx, idx_to_clade, num_classes)
        - clade_to_idx: Dict mapping clade name to integer label
        - idx_to_clade: Dict mapping integer label to clade name
        - num_classes: Number of unique clades
    """
    print("Loading clade information from CSV...")

    # Read CSV and build genome_id -> clade mapping
    id_to_clade = {}
    with open(sequences_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            genome_id = row["name"]
            clade = row.get("nextstrain_clade", "")
            if genome_id in ids:
                id_to_clade[genome_id] = clade

    # Get unique clades (excluding empty/unassigned ones)
    unique_clades = set(
        clade for clade in id_to_clade.values() if clade and clade != "unassigned"
    )
    unique_clades = sorted(list(unique_clades))

    # Create bidirectional mapping
    clade_to_idx = {clade: idx for idx, clade in enumerate(unique_clades)}
    idx_to_clade = {idx: clade for clade, idx in clade_to_idx.items()}

    # Add 'unknown' class for unassigned/missing clades
    unknown_idx = len(clade_to_idx)
    clade_to_idx["unknown"] = unknown_idx
    idx_to_clade[unknown_idx] = "unknown"

    num_classes = len(clade_to_idx)

    print(f"Found {num_classes} unique clades (including 'unknown' class)")
    print(f"Clades: {unique_clades + ['unknown']}")

    return clade_to_idx, idx_to_clade, num_classes, id_to_clade


def load_data(args):
    """Load embeddings for all proteins and clade labels from files."""
    print("Loading data for all proteins...")

    # Load genome IDs (names)
    with open(args.ids_file, "r") as f:
        ids = [line.strip() for line in f if line.strip()]

    # Load embeddings for all three proteins
    embeddings_files = {
        "S": f"{args.data_dir}/s_sequence_embeddings.npy",
        "N": f"{args.data_dir}/n_sequence_embeddings.npy",
        "ORF1a": f"{args.data_dir}/orf1a_sequence_embeddings.npy",
    }

    embeddings_dict = {}
    emb_dim = None

    for protein, file_path in embeddings_files.items():
        print(f"Loading {protein} embeddings from {file_path}")
        embeddings_array = np.load(file_path)

        # Check that all proteins have same embedding dimension
        if emb_dim is None:
            emb_dim = embeddings_array.shape[1]
        else:
            if embeddings_array.shape[1] != emb_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: {protein} has dim {embeddings_array.shape[1]}, "
                    f"expected {emb_dim}"
                )

        # Convert to dictionary mapping id to tensorized embedding
        embeddings_dict[protein] = {
            gid: torch.from_numpy(embeddings_array[i]) for i, gid in enumerate(ids)
        }

    # Load clade labels
    clade_to_idx, idx_to_clade, num_classes, id_to_clade = load_clade_mapping(
        args.sequences_csv, ids
    )

    # Create clade label dictionary
    clade_labels = {}
    unknown_count = 0
    for gid in ids:
        clade = id_to_clade.get(gid, "")
        if not clade or clade == "unassigned":
            clade_labels[gid] = clade_to_idx["unknown"]
            unknown_count += 1
        else:
            clade_labels[gid] = clade_to_idx[clade]

    print(f"Genomes with unknown/unassigned clade: {unknown_count}/{len(ids)}")

    # num genomes
    num_samples = len(ids)
    print(f"Loaded {num_samples} samples of dimension {emb_dim} for each protein")

    return (
        ids,
        embeddings_dict["S"],
        embeddings_dict["N"],
        embeddings_dict["ORF1a"],
        clade_labels,
        emb_dim,
        num_classes,
        idx_to_clade,
    )


def main():
    """Main training loop."""
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load data for all proteins
    (
        ids,
        embeddings_s,
        embeddings_n,
        embeddings_orf1a,
        clade_labels,
        emb_dim,
        num_classes,
        idx_to_clade,
    ) = load_data(args)

    # Create dataset with all protein embeddings
    dataset = EmbeddingDataset(
        ids, embeddings_s, embeddings_n, embeddings_orf1a, clade_labels
    )

    print(f"\nTraining multi-protein fusion classification model...")
    print(f"Proteins: S, N, ORF1a")
    print(f"Hyperparameters:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Output dim: {args.out_dim}")
    print(f"  Number of queries: {args.num_queries}")
    print(f"  Number of heads: {args.num_heads}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Validation batches: {args.val_batches}")
    print()

    # Train model
    model = train_classification_model(
        dataset=dataset,
        emb_dim=emb_dim,
        num_classes=num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        num_queries=args.num_queries,
        num_heads=args.num_heads,
        val_batches=args.val_batches,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    # Save fused projected embeddings
    print(f"\nSaving fused projected embeddings...")
    save_projected_embeddings(
        dataset=dataset,
        model=model,
        output_prefix=args.output_prefix,
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
