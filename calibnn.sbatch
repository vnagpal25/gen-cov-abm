#!/bin/bash

# ===== SLURM options =====
#SBATCH --job-name=calibnn-gpu
#SBATCH --account=cse598s002f25_class
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/calibnn_gpu_%j.out
#SBATCH --error=logs/calibnn_gpu_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

module load python/3.10.4
source ~/miniconda3/etc/profile.d/conda.sh
conda activate covid_clean

cd /home/namishah/alex/gen-cov-abm

python -m covid_abm.run_calib_nn \
    --base_config covid_abm/yamls/config_base.yaml \
    --truth_column cases \
    --epochs 10 \
    --device cuda