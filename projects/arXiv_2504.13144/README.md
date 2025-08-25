# Bayesian model-data comparison incorporating theoretical uncertainties

Code for https://arxiv.org/abs/2504.13144.

This directory contains the numerical implementation used in the paper. It includes ready-to-plot results in the paper, and scripts to reproduce the inference from scratch.

Directory layout
    - environment.yml – conda environment (Install: conda env create -f environment.yml)
    - ball_drop/ : ball drop problem implementation
        - experimental_data/ : input data files
        - mcmc_data/ : precomputed MCMC chains for quick plotting
    - heavy-ion/ : heavy-ion problem implementation
        - 2param_model/ : heavy-ion problem with 2 parameters 
        - 5param_model/ : heavy-ion problem with 5 parameters 
        - experimental_data/ : input data files
            - mcmc_data/ : precomputed MCMC chains for quick plotting

        
Requirements:
    - Conda (miniconda/anaconda)
    - Python 3.11 (as pinned in environment.yml.)


Quick start:
1) Plot precomputed results:
   Precomputed chains are in mcmc_data/* for ball_drop, heavy-ion/2param_model, and heavy-ion/2param_model. Open and run the plotting notebooks:
    - ball_drop/plot_*.nb
    - heavy-ion/2param_model/plot_*.nb
    - heavy-ion/5param_model/plot_*.nb


2) Run the inference from scratch:
    - Activate the environment: conda activate md
    - (cd ball_drop && python3 run_ball_drop.py)
    - (cd heavy-ion/2param_model && python3 run_*.py)
    - (cd heavy-ion/5param_model && python3 run_*.py)
    - Plot precomputed results (Step 1)
