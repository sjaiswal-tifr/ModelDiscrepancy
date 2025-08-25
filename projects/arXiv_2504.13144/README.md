## Bayesian model-data comparison incorporating theoretical uncertainties

Code for https://arxiv.org/abs/2504.13144.

This directory contains the numerical implementation used in the paper. It includes ready-to-plot results in the paper, and scripts to reproduce the inference from scratch.

**Directory layout:**
- ../../src/ : code for Bayesian inference, with or without theoretical uncertainties 
  - `ModelDiscrepancy.py` : class to compute likelihood and posteriors
  - `sampling_methods.py` : MCMC sampling functions
  - `get_quantiles.py` : functions to compute quantiles for observables using MCMC chains
  - `plot_scripts.py` : plotting helpers (e.g., corner plots)
- ball_drop/ : ball drop problem implementation
  - experimental_data/ : input data files
  - mcmc_data/ : precomputed MCMC chains obtained after Bayesian inference
- heavy-ion/ : heavy-ion problem implementation
  - experimental_data/ : input data files
  - 2param_model/ : heavy-ion problem with 2 parameters
    - mcmc_data/ : precomputed MCMC chains obtained after Bayesian inference
  - 5param_model/ : heavy-ion problem with 5 parameters 
    - mcmc_data/ : precomputed MCMC chains obtained after Bayesian inference

**Requirements:**
- Conda (miniconda/anaconda)
- Python 3.11 (as pinned in environment.yml.)

**Quick start:**
1) Install and activate environment:
    - `environment.yml`: conda environment. Install with `conda env create -f environment.yml`
    - Activate environment: `conda activate md`

2) Plot results: Precomputed chains are in ball_drop/mcmc_data/, heavy-ion/2param_model/mcmc_data/, and heavy-ion/5param_model/mcmc_data/. Open and run the plotting notebooks:
    - ball_drop/plot_ball_drop.nb
    - heavy-ion/2param_model/plot_shear0p1_esw0p2.nb
    - heavy-ion/5param_model/plot_shear0p1_esw0p2.nb
    - heavy-ion/5param_model/plot_etakink0p1_Tkink0p18_ahigh1_alow-1_esw0p2.nb

3) Run the inference from scratch (this takes time -- run on a cluster). The code is parallelized to use all availabile CPU cores.
    - For ball-drop experiment:
        ```bash
        (cd ball_drop && python3 run_ball_drop.py)
        ```
    - For heavy-ion 2 parameter model:
        ```bash
        (cd heavy-ion/2param_model && python3 run_shear0p1_esw0p2.py)
        ```
    - For heavy-ion 5 parameter model:
        ```bash
        (cd heavy-ion/5param_model && python3 run_shear0p1_esw0p2.py)
        ```
        ```bash
        (cd heavy-ion/5param_model && python3 run_etakink0p1_Tkink0p18_ahigh1_alow-1_esw0p2.py)
        ```
    
    - Plot results (Step 2)
