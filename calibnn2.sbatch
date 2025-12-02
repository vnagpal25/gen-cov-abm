#!/bin/bash
#SBATCH --job-name=calibnn-ma
#SBATCH --account=cse598s002f25_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/calibnn_%j.out
#SBATCH --error=logs/calibnn_%j.err

module load python/3.10
source ~/.venvs/covid/bin/activate
cd /home/namishah/alex/gen-cov-abm

python -m covid_abm.run_calib_nn \
    --base_config covid_abm/yamls/config_base.yaml \
    --truth_column cases \
    --epochs 10 \
    --lr 0.01