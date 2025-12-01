# gen-cov-abm: **Genomics-informed COVID-19 forecasting using differentiable agent-based modeling**
This repository contains experiments extending **GradABM** with viral genomic embeddings (Gen-COV-ABM) and the **Massachusetts county-level baseline** needed for calibration and evaluation.


## Massachusetts GradABM Baseline Setup

```bash
conda create -n covid python=3.10
conda activate covid
pip install -r requirements.txt
pip install -e AgentTorch   # run from repo root


conda create -n covid python=3.10 && conda activate covid
pip install -r requirements.txt
pip install -e AgentTorch (from the repo root)

```

## Data Overview
### Population Data
Agent-based demographic and mobility structures per county:
