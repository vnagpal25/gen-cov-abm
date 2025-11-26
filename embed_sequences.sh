#!/bin/bash

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=embed_sequences
#SBATCH --account=cse598s002f25_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16000m
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=log.out
#SBATCH --error=error.out

echo "Starting full GPT training..."
echo "=================================="

# Set up environment

source /sw/pkgs/arc/python3.10-anaconda/2023.03/etc/profile.d/conda.sh
conda activate gen_cov_abm

# Training hyperparameters for full model
python embed-sequences.py

echo "Inference completed!"
