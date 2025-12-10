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

# June 1 – July 5 window
$PYTHON -m covid_abm.run_metrics \
  --base_config covid_abm/yamls/config_base.yaml \
  --truth_column cases \
  --population_suffix sample30000 \
  --start_date 2021-06-01 \
  --end_date 2021-07-05 \
  --use_calib \
  --calib_dir Results/calib_params_jun21 \
  --output_csv Results/metrics_summary_jun21.csv

# June 27 – July 31 window
$PYTHON -m covid_abm.run_metrics \
  --base_config covid_abm/yamls/config_base.yaml \
  --truth_column cases \
  --population_suffix sample30000 \
  --start_date 2021-06-27 \
  --end_date 2021-07-31 \
  --use_calib \
  --calib_dir Results/calib_params_jul21 \
  --output_csv Results/metrics_summary_jul21.csv

