#!/bin/bash

# Data files
IDS_FILE="data/ma_distance_matrix_node_names.txt"
# DIST_MATRIX_FILE="data/ma_evolutionary_distance_matrix_normalized.npy"
SEQUENCES_CSV="data/all_sequences-v2.csv"
DATA_DIR="data"

# Training hyperparameters
EPOCHS=200
# LEARNING_RATE=5e-5
# NUM_NEGATIVES=15        # K: number of negative samples per positive
# POS_THRESHOLD=0.10      # Phylogenetic distance threshold for positive 
# samples
# NEG_THRESHOLD=0.40      # Phylogenetic distance threshold for negative 
# samples
# BATCH_SIZE=1            # Currently fixed at 1 due to sampler logic
LEARNING_RATE=1e-3
BATCH_SIZE=32           # Batch size for classification training

# Model architecture
HIDDEN_DIM=1280          # Hidden dimension of projection head MLP
OUT_DIM=1280             # Output dimension of projection head
NUM_QUERIES=1            # Number of learnable queries for cross-attention
NUM_HEADS=8              # Number of attention heads

# Output and logging
OUTPUT_PREFIX="data/finetuned"
WANDB_PROJECT="phylo-embedding-finetuning"
SEED=42

echo "Multi-Protein Cross-Attention Fusion with Clade Classification"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "Batch Size: $BATCH_SIZE"
echo "Hidden Dim: $HIDDEN_DIM"
echo "Output Dim: $OUT_DIM"
echo "Num Queries: $NUM_QUERIES"
echo "Num Heads: $NUM_HEADS\n"

# Run the training script
uv run python test-fine-tune.py \
  --ids-file $IDS_FILE \
  --sequences-csv $SEQUENCES_CSV \
  --data-dir $DATA_DIR \
  --epochs $EPOCHS \
  --lr $LEARNING_RATE \
  --batch-size $BATCH_SIZE \
  --hidden-dim $HIDDEN_DIM \
  --out-dim $OUT_DIM \
  --num-queries $NUM_QUERIES \
  --num-heads $NUM_HEADS \
  --output-prefix $OUTPUT_PREFIX \
  --wandb-project $WANDB_PROJECT \
  --seed $SEED \
  --val-batches 0


# --dist-matrix-file $DIST_MATRIX_FILE \
#  --negatives $NUM_NEGATIVES \
#  --pos-thresh $POS_THRESHOLD \
#  --neg-thresh $NEG_THRESHOLD \
