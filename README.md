## Gen-COV-ABM: User’s Manual 

### Overview
This repo contains a differentiable COVID-19 agent-based model (ABM) with a small neural calibration network (Calib-NN). In this project we use it to simulate COVID-19 spread in Massachusetts counties over two 5‑week windows with and without genomics embeddings.

The workflow is:
- **1**: Set up a Python environment and install dependencies.
- **2**: Download and preprocess data for simulation and use the provided county-level data and population files (already included in the repo).
- **3**: Train Calib-NN across all counties and time windows, without genomic embeddings.
- **4**: Run the baseline_covid_abm to produce ND/RMSE/MAE metrics for the calibrated baseline.
- **x**: Train Calib-NN across all counties and time windows, with genomic embeddings.
- **x**: Run the cov_abm to produce ND/RMSE/MAE metrics for the calibrated baseline.

### 1. Installation and Environment
These commands assume you are in the project root directory (the folder containing this README and the `covid_abm` package).

- **Create and activate a conda environment** (only needed once):
```bash
conda create -n covid_abm_venv python=3.10
conda activate covid_abm_venv
```

- **Install Python dependencies from pyproject.toml**:
```bash 
pip install .
```

- **Install the local AgentTorch dependency in editable mode**:
```bash
git submodule add https://github.com/AgentTorch/AgentTorch.git AgentTorch
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
  - Navigate to the `data` directory and run the bash script `curl_broad_institute_data.sh` to pull the raw data from the Broad Institute Covid-19 dataset
    ```bash
    cd data/genomic_data
    sh curl_broad_institute_data.sh
    ``` 
  - Refer to `notebooks/explore-data.ipynb` for a preliminary data exploration
  - Execute `python src/data/extract_sequences.py` to extract the protein and genome sequences from the phylogenetic tree
  - Execute the following`python src/data/embed_sequences.py` to extract the embeddings of the ORF1a protein sequences using the ESM-2 protein language model
    ```bash
    python embed_sequences.py orf1a_sequence --batch-size 8
    ```
  - Refer to `notebooks/visualize-embeddings.ipynb` for visualizing the embedded sequences by subclade/strain
  - Execute `python src/data/sample_sequences_for_agents.py` to sample genome sequenecs for exposed agents by county level 
  - Optional: We were ultimately unsuccessful in finetuning our embeddings to more closely reflect the phylogenetic tree structure and unique characteristics of Covid-19. However, we have include the files for that efforts in our `src/data` directory: `train_embeddings.sh` and `fine_tune_embeddings-classification.py` 

### 3. Training Calib-NN (Baseline + Genomics Calibration)
Calib-NN is trained to map county-level features (i.e., weekly cases) to time-varying transmission and mortality parameters for each county and time window.

To train Calib-NN on CPU for all counties and both windows, run the `calibnn_baseline.sbatch` file. You can edit the paths here to match your path directory to the folder as well as change compute usage. 

Each `.pt` file is a small PyTorch checkpoint containing the tensors that will later be injected into the simulator.

For genomics-embedded calibration, run the calibnn.sbatch file instead. The results will populate under Results folder this time. 

### 4. Running the ABM and Computing Metrics
Once Calib-NN has finished training and the parameter files exist in `Baseline_Results/calib_params/`, you can generate the baseline performance metrics.

To run the baseline calibrated simulations and compute ND, RMSE, and MAE for all counties and windows, run the `metrics_baseline.sbatch` file which will produce 2 files called metrics_summary + date under the `Baseline_Results` folder. You can open these files in any spreadsheet to inspect the baseline metrics.

For genomics-embedded calibration, run the `metrics.sbatch` file instead. The results will populate under Results folder this time. 

