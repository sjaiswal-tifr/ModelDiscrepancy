#!/usr/bin/env python3

import os, sys
import psutil
import numpy as np
from joblib import Parallel, delayed
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------

def set_njobs(ncores):
    """
    Get number of availaible cores for parallelizing
    """

    # Try to get the allowed CPU count via affinity (works on Linux/Windows)
    try:
        available_cores = len(os.sched_getaffinity(0))# len(psutil.Process().cpu_affinity())
    except Exception:
        # On systems where affinity is not available (e.g., macOS), use total CPU count
        available_cores = os.cpu_count()

    # Validate that ncores is an integer and at least -1
    if not isinstance(ncores, int) or ncores < -1:
        raise ValueError("ncores must be an integer greater than or equal to -1. Specify '-1' to use all available cores.")
    
    if ncores == -1:
        # Use all available cores
        num_cores = available_cores
    elif ncores > available_cores:
        # Warn if more cores are requested than available and then use available cores
        warnings.warn(
            f"ncores ({ncores}) exceeds available cores ({available_cores}). Using available cores.",
            UserWarning
        )
        num_cores = available_cores
    else:
        num_cores = ncores  

    return num_cores

# --------------------------------------------------------------------------------------------

def realization_model(MDclass, batch_samples, n_realizations):
    """
    Processes batch of MCMC samples and generate n_realizations of model predictions
      Note: model prediction is gaussian for each sample.
      Function computes model means and stds, annd generates n_realizations from the gaussian.
    
    Parameters:
    - MDclass: Instantiated model discrepancy class
    - batch_samples: MCMC samples of shape (batch_size, num_params)
    - n_realizations: Number of realizations per sample
    
    Returns:
    - realizations_flat: numpy array of shape (batch_size * n_realizations, num_observables)
    """

    all_realization = []
    
    for theta in batch_samples:
        theta = np.atleast_1d(theta) # theta is a 1-D array of shape (num_model_parameters,)
        model_mean, model_std = MDclass.model_predict(theta) # both are 1-D arrays of length total_observations

        # now draw n_draws samples from N(model_mean, model_std^2):
        # result has shape (n_realizations, total_observations)
        batch_realizations = np.random.normal(
            loc=model_mean,          # broadcasts to (n_realizations, total_observations)
            scale=model_std,         # same shape
            size=(n_realizations, model_mean.shape[0])
        )

        # collect this sample's draws
        all_realization.append(batch_realizations)
    
    # after the loop, stack into one big array:
    # shape = (batch_size * n_realizations, total_observations)
    return np.vstack(all_realization)

# ----------------------------------------------------------------------------------

def realization_modelPlusGP(MDclass, batch_samples, n_realizations):
    """
    Processes batch of MCMC samples and generate n_realizations from posteror predictive distribution (PPD).
      Note: model + GP prediction is gaussian for each sample. 
      Function computes model + GP means and stds, annd generates n_realizations from the gaussian.
    
    Parameters:
    - MDclass: Instantiated model discrepancy class
    - batch_samples: MCMC samples of shape (batch_size, num_params)
    - n_realizations: Number of realizations from PPD per sample
    
    Returns:
    - realizations_flat: numpy array of shape (batch_size * n_realizations, num_observables)
    """
    
    num_model_param = MDclass.num_model_parameters
    MD_hp_counts = MDclass.MD_hp_counts
    num_observables = MDclass.num_observables
    scaled_exp_data = MDclass.scaled_exp_data
    MD_kernels = MDclass.MD_kernels

    # List for storing all unscaled realizations for all samples
    unscaled_realization_all = []
    
    for thetaphi in batch_samples:
        theta = np.atleast_1d(thetaphi[:num_model_param])
        phi = np.atleast_1d(thetaphi[num_model_param:])

        model_mean, model_std = MDclass.model_predict(theta)

        # Indices for slicing
        obs_start = 0
        phi_start = 0
        # List for storing realizations for each sample
        unscaled_realization_allobs = []
        
        # Loop over each observable
        for i in range(num_observables):
            obs_info        = scaled_exp_data[i]
            n_obs           = obs_info["n_obs"]
            x               = obs_info["x"]
            mean_exp_scaled = obs_info["mean_exp_scaled"]
            std_exp_scaled  = obs_info["std_exp_scaled"]
            offset          = obs_info["offset"]
            scale           = obs_info["scale"]
            
            obs_end = obs_start + n_obs

            # Extract model predictions for this observable (in unscaled space)
            mean_pred_obs = model_mean[obs_start:obs_end]   # unscaled
            std_pred_obs  = model_std[obs_start:obs_end]    # unscaled

            # Scale model mean and std
            mean_pred_obs_scaled = (mean_pred_obs - offset) / scale
            std_pred_obs_scaled  = std_pred_obs / scale

            # The model variance (in scaled space)
            cov_diag_eta = std_pred_obs_scaled**2

            kernel_func = MD_kernels[i]
            hp_count = MD_hp_counts[i]
            phi_end = phi_start + hp_count

            # Build the discrepancy covariance matrix (in scaled space)
            cov_total_zeta = kernel_func(x, x, *phi[phi_start:phi_end])

            # Build the full covariance (in scaled space)
            np.fill_diagonal(cov_total_zeta, cov_total_zeta.diagonal() + cov_diag_eta)

            # The variance in data (in scaled space)
            cov_eps = np.diag(std_exp_scaled**2)

            # Compute posterior predictive mean and covariance (in scaled space)
            S = cov_total_zeta + cov_eps
            mean_zetagiveny = mean_pred_obs_scaled + cov_total_zeta @ np.linalg.solve(S, (mean_exp_scaled-mean_pred_obs_scaled).T)
            covariance_zetagiveny = cov_total_zeta - cov_total_zeta @ np.linalg.solve(S, cov_total_zeta.T)
            
            # Draw n_realizations from the multivariate normal distribution (in scaled space)
            drawn_realization = np.random.multivariate_normal(
                mean=mean_zetagiveny, 
                cov=covariance_zetagiveny, 
                size=n_realizations
            )

            # Unscale the realizations and bring it on original scale
            unscaled_realization = drawn_realization * scale + offset

            # Append realizations to the list to store all realizations for all observables
            unscaled_realization_allobs.append(unscaled_realization)

            # Advance to the next observable
            obs_start = obs_end
            phi_start = phi_end
            
        # now shape of unscaled_realization_allobs is
        # [ (n_realizations, n_obs1), (n_realizations, n_obs2), ... ]
        # we want to stitch these horizontally so each row is all observables:
        # result -> (n_realizations, total_observations)
        sample_realizations = np.hstack(unscaled_realization_allobs)

        # collect for all samples
        unscaled_realization_all.append(sample_realizations)

    # final shape: (batch_size * n_realizations, total_observations)
    return np.vstack(unscaled_realization_all)

# ----------------------------------------------------------------------------------

def quantiles(samples, MDclass, save_filename, whichmodel, n_realizations=100, ncores=-1, quantiles=[50, 2.5, 97.5, 16, 84]):
    """
    Computes quantiles for the given samples by generating realizations in parallel. 
    Quantiles are computed only at experimental data locations.
    
    Parameters:
    - samples: numpy array of shape (total_samples, num_params)
    - MDclass : Instantiated model discrepancy class
    - save_filename: Filename to save quantiles along with directory path
    - whichmodel: allowed 'modelPlusGP' or 'model'. 
    - n_realizations: Number of realizations per sample
    - ncores: Number of parallel jobs (-1 uses all available cores)
    - quantiles: List of quantiles to compute (in percent)
    
    Returns:
    - quantiles_dict: Dictionary with quantile percentages as keys and numpy arrays as values
    """

    start_time = time.time()

    logger.info("Computing quantiles for model + GP...")

    num_cores = set_njobs(ncores)
    
    batch_size = int(np.ceil(len(samples)/num_cores))
    
    # Split samples into batches
    batches = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]
    
    logger.info(f"Total samples: {len(samples)}. Total cores: {num_cores}. Batch size: {batch_size}. Total batches: {len(batches)}")
    
    # Parallel processing of batches
    if whichmodel == 'modelPlusGP':
        results = Parallel(n_jobs=ncores)(
            delayed(realization_modelPlusGP)(MDclass, batch, n_realizations) for batch in batches
        )
    elif whichmodel == 'model':
        results = Parallel(n_jobs=ncores)(
            delayed(realization_model)(MDclass, batch, n_realizations) for batch in batches
        )
    else:
        raise ValueError("During quantile computation: 'whichmodel' must be 'modelPlusGP' or 'model'.")
    
    # Concatenate all realizations
    all_realizations = np.vstack(results)  # Shape: (total_samples * n_realizations, num_observables)
    logger.info(f"All realizations shape: {all_realizations.shape}")
    
    # Compute quantiles
    quantiles_values = np.percentile(all_realizations, quantiles, axis=0)  # Shape: (len(quantiles), num_observables)
    
    end_time = time.time()

    header_str =  "# Quantiles: " + ", ".join(str(q) for q in quantiles)
    np.savetxt(save_filename, quantiles_values, header=header_str)

    logger.info(f"Time taken: {end_time - start_time:.2f} seconds. Quantiles saved in {save_filename}\n")

# ----------------------------------------------------------------------------------
