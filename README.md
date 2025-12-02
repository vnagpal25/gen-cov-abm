## Gen-COV-ABM: User’s Manual 

### Overview
This repo contains a differentiable COVID-19 agent-based model (ABM) with a small neural calibration network (Calib-NN). In this project we use it to simulate COVID-19 spread in Massachusetts counties over two 5‑week windows with and without genomics embeddings.

The workflow is:
- **1**: Set up a Python environment and install dependencies.
- **2**: Use the provided county-level data and population files (already included in the repo).
- **3**: Train Calib-NN across all counties and time windows, without genomic embeddings.
- **4**: Run the covid_abm to produce ND/RMSE/MAE metrics for the calibrated baseline.
- **5**: Train Calib-NN across all counties and time windows, with genomic embeddings.
- **6**: Run the gen-cov_abm to produce ND/RMSE/MAE metrics for the calibrated baseline.

### 1. Installation and Environment
These commands assume you are in the project root directory (the folder containing this README and the `covid_abm` package).

- **Create and activate a conda environment** (only needed once):
```bash
conda create -n covid python=3.10
conda activate covid
```

- **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

- **Install the local AgentTorch dependency in editable mode**:
```bash
pip install -e AgentTorch
```

After this, the `covid_abm` package and AgentTorch should be importable, and all scripts below can be run from the project root.

### 2. Data and Configuration
All required inputs for the baseline experiment are already included in the repository:

- **County-level time series and features**:
  - Located under `covid_abm/data/` (for example, `county_data.csv`).
- **Simulation configuration**:
  - Main config file: `covid_abm/yamls/config_base.yaml`
  - This file points to the correct population, age, disease-state, and network files and is already set for:
    - 5,000 agents per run
    - Two 5-week evaluation windows (July 2020 and August 2020)
- **Genomics Data**:


### 3. Training Calib-NN (Baseline Calibration)
Calib-NN is trained to map county-level features (i.e., weekly cases) to time-varying transmission and mortality parameters for each county and time window.

To train Calib-NN on CPU for all counties and both windows, run:

```bash
conda activate covid
python -m covid_abm.run_calib_nn \
    --base_config covid_abm/yamls/config_base.yaml \
    --truth_column cases \
    --epochs 10
```

Each `.pt` file is a small PyTorch checkpoint containing the tensors that will later be injected into the simulator.

### 4. Running the ABM and Computing Metrics
Once Calib-NN has finished training and the parameter files exist in `Results/calib_params/`, you can generate the baseline performance metrics.

To run the calibrated simulations and compute ND, RMSE, and MAE for all counties and windows:

```bash
conda activate covid
python covid_abm/run_metrics.py \
    --base_config covid_abm/yamls/config_base.yaml \
    --truth_column cases \
    --use_calib \
    --calib_dir Results/calib_params \
    --output_csv Results/metrics_summary_calib.csv
```

You can open `Results/metrics_summary_calib.csv` in any spreadsheet to inspect the baseline metrics.


