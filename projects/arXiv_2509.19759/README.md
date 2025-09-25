## Phenomenological constraints on QCD transport with quantified theory uncertainties

This directory contains the numerical implementation used in the paper. It includes plots of results in the paper, and scripts to reproduce the inference from scratch.

**Directory layout:**
- ../../src/ : code for Bayesian inference, with or without theoretical uncertainties 
  - `ModelDiscrepancy.py` : class to compute likelihood and posteriors
  - `sampling_methods.py` : MCMC sampling functions
  - `get_quantiles.py` : functions to compute quantiles for observables using MCMC chains
  - `plot_scripts.py` : plotting helpers (e.g., corner plots)
- projects/arXiv_2509.xxxxx/
  - experimental_data/ : experimental data files
  - emulators/ : trained emulators with simulation file and training method
  - mcmc_data/ : precomputed MCMC chains obtained after Bayesian inference

**Requirements:**
- Conda (miniconda/anaconda)
- Python 3.11 (as pinned in environment.yml)

**Quick start:**
1) Install and activate environment:
    - `environment.yml`: conda environment. Install with `conda env create -f environment.yml`
    - Activate environment: `conda activate md`

2) Plot results: Precomputed chains are in `mcmc_data/`. Open and run the plotting notebooks:
    - `plots.ipynb`

3) Run Bayesian inference from scratch and generate MCMC chains (computationally expensive)
   ```bash
    python3 run_Grad.py
    python3 run_CE.py
    ```
    - Plot results (Step 2)
