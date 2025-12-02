#!/bin/bash
#SBATCH --job-name=metrics
#SBATCH --account=cse598s002f25_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/metrics_%j.out
#SBATCH --error=logs/metrics_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT_ROOT=/home/namishah/alex/gen-cov-abm
PYTHON=/home/namishah/miniconda3/envs/covid_clean/bin/python

cd "$PROJECT_ROOT"
"$PYTHON" -m covid_abm.run_metrics \
  --base_config covid_abm/yamls/config_base.yaml \
  --truth_column cases \
  --use_calib \
  --calib_dir Results/calib_params \
  --output_csv Results/metrics_summary_calib.csv
