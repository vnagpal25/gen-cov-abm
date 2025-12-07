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

source ~/miniconda3/etc/profile.d/conda.sh
conda activate covid_clean
PYTHON=~/miniconda3/envs/covid_clean/bin/python

cd /home/namishah/alex/gen-cov-abm

$PYTHON -m covid_abm.run_calib_nn \
  --base_config covid_abm/yamls/config_base.yaml \
  --truth_column cases \
  --epochs 10 \
  --device cuda \
  --population_suffix sample30000 \
  --start_date 2021-06-01 \
  --end_date 2021-07-05 \
  --output_dir Results/calib_params_jun21

$PYTHON -m covid_abm.run_calib_nn \
  --base_config covid_abm/yamls/config_base.yaml \
  --truth_column cases \
  --epochs 10 \
  --device cuda \
  --population_suffix sample30000 \
  --start_date 2021-06-27 \
  --end_date 2021-07-31 \
  --output_dir Results/calib_params_jul21