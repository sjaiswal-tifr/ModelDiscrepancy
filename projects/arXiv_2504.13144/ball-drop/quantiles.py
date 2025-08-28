#!/usr/bin/env python3

import os, sys
import psutil
import numpy as np
from joblib import Parallel, delayed
import time
import warnings
import logging
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
def quantiles_model(samples, x_predict, model_predict_func, save_filename, quantiles=[50, 2.5, 97.5, 16, 84]):
    """
    Calculate quantiles for model outputs (height, velocity, acceleration)
    from MCMC samples.

    Parameters:
        samples: 2D numpy array of shape (num_samples, ndim) containing MCMC samples.
        x_predict: 1D array of time steps at which the model is evaluated.
        model_predict_func : function for model prediction
        quantiles: List of percentiles to compute.

    Returns:
        h_quantiles, v_quantiles, a_quantiles: Each is a numpy array of shape 
        (len(quantiles), len(x_predict)) containing the computed percentiles.
    """
    start_time = time.time()
    
    num_samples = samples.shape[0]
    nots = len(x_predict)

    logger.info("Computing quantiles for model predictions...")
    
    # Preallocate arrays for simulation outputs (each row corresponds to one sample)
    h_mc = np.empty((num_samples, nots))
    v_mc = np.empty((num_samples, nots))
    a_mc = np.empty((num_samples, nots))

    # Loop over samples, calling model_predict_func for each sample
    for i in range(num_samples):
        sol_mc, _ = model_predict_func(samples[i])

        # Assuming sol_mc is a vector of length 3*nots:
        h_mc[i, :] = sol_mc[:nots]
        v_mc[i, :] = sol_mc[nots:2*nots]
        a_mc[i, :] = sol_mc[2*nots:3*nots]

    # Compute percentiles across the sample dimension (axis=0) for each time step.
    h_quantiles = np.percentile(h_mc, quantiles, axis=0)
    v_quantiles = np.percentile(v_mc, quantiles, axis=0)
    a_quantiles = np.percentile(a_mc, quantiles, axis=0)

    # concatenate along columns to get shape (Q, 3*nots)
    quantiles_values = np.concatenate([h_quantiles, v_quantiles, a_quantiles], axis=1)

    # Save with header
    header_str = "# Quantiles: " + ", ".join(str(q) for q in quantiles)
    np.savetxt(save_filename, quantiles_values, header=header_str, fmt="%.8f")

    end_time = time.time()
    logger.info(f"Time taken: {end_time - start_time:.2f} seconds. Quantiles saved in {save_filename}\n")

# ----------------------------------------------------------------------------------

def realization_modelPlusGP(MDclass, x_predict, model_predict_func, batch_samples, n_realizations):
    """
    Processes batch of MCMC samples and generate n_realizations from posteror predictive distribution (PPD).
      Note: model + GP prediction is gaussian for each sample. 
      Function computes model + GP means and stds, annd generates n_realizations from the gaussian.
    
    Parameters:
    - MDclass: Instantiated model discrepancy class
    - x_predict : prediction points
    - model_predict_func : function for model prediction
    - batch_samples: MCMC samples of shape (batch_size, num_params)
    - n_realizations: Number of realizations from PPD per sample
    
    Returns:
    - realizations_flat: numpy array of shape (batch_size * n_realizations, num_observables)
    """
    
    num_model_param  = MDclass.num_model_parameters
    MD_hp_counts     = MDclass.MD_hp_counts         # list of hp counts per observable
    num_observables  = MDclass.num_observables
    scaled_exp_data  = MDclass.scaled_exp_data      # list of dicts, one per observable
    MD_kernels       = MDclass.MD_kernels           # list of kernel callables per observable

    # prediction grid as column vector
    X_star = np.asarray(x_predict).reshape(-1, 1)
    m_predict = X_star.shape[0]  # length of predict mean for each observable

    # Collect all samples' realizations here
    unscaled_realization_all = []

    for thetaphi in batch_samples:
        theta = np.atleast_1d(thetaphi[:num_model_param])
        phi   = np.atleast_1d(thetaphi[num_model_param:])

        # Model predictions on the training grid (stacked over observables)
        model_mean_train, model_std_train = MDclass.model_predict(theta)
        model_mean_train = np.asarray(model_mean_train).reshape(-1)
        model_std_train  = np.asarray(model_std_train).reshape(-1)

        # Model predictions on the prediction grid (stacked over observables)
        model_mean_predict, model_std_predict = model_predict_func(theta)
        model_mean_predict = np.asarray(model_mean_predict).reshape(-1)
        model_std_predict  = np.asarray(model_std_predict).reshape(-1)

        # For slicing model predictions
        obs_start_train = 0
        obs_start_predict = 0
        phi_start  = 0

        # Will hold this sample's realizations per observable (each: n_realizations x n_pred)
        unscaled_realization_allobs = []

        for i in range(num_observables):
            obs_info        = scaled_exp_data[i]
            X_train         = obs_info["x"]
            n_train         = obs_info["n_obs"]
            mean_exp_scaled = obs_info["mean_exp_scaled"]
            std_exp_scaled  = obs_info["std_exp_scaled"]
            offset          = obs_info["offset"]
            scale           = obs_info["scale"]
        
            obs_end_train = obs_start_train + n_train
            obs_end_predict = obs_start_predict + m_predict

            # Slice this observable's model curve on the train grid x_predict
            mean_obs_train = model_mean_train[obs_start_train:obs_end_train]   # unscaled  # (n_train,)
            std_obs_train  = model_std_train[obs_start_train:obs_end_train]    # unscaled  # (n_train,)

            # Scale model mean and std for train
            mean_obs_train_scaled = (mean_obs_train - offset) / scale   # (n_train,)
            std_obs_train_scaled  = std_obs_train / scale               # (n_train,)
            
            # Slice this observable's model curve on the prediction grid x_predict
            mean_obs_predict = model_mean_predict[obs_start_predict:obs_end_predict]   # unscaled  # (m_predict,)
            std_obs_predict  = model_std_predict[obs_start_predict:obs_end_predict]    # unscaled  # (m_predict,)

            # Scale model mean and std for predict
            mean_obs_predict_scaled = (mean_obs_predict - offset) / scale   # (n_pred,)
            std_obs_predict_scaled  = std_obs_predict / scale               # (n_pred,)

            # Kernels & hyperparameters
            kernel_func = MD_kernels[i]
            hp_count = MD_hp_counts[i]
            phi_end = phi_start + hp_count
            phi_slice   = phi[phi_start:phi_end]

            # K matrix is from covariance kernel
            # K11 : train covariance. K12: train-predict coorelation. K22: predict covariance
            K11  = kernel_func(X_train, X_train, *phi_slice)   # (n_train, n_train)
            K12  = kernel_func(X_star,  X_train, *phi_slice)   # (m_predict, n_train)
            K22  = kernel_func(X_star,  X_star,  *phi_slice)   # (m_predict, m_predict)
            K22 = K22.copy()
            np.fill_diagonal(K22, K22.diagonal() + std_obs_predict_scaled**2)
            

            # Total training covariance S = K11 + diag(model_var + data_var)   (in scaled space)
            S_diag = (std_obs_train_scaled**2) + (std_exp_scaled**2)         # (n_train,)
            S = K11 + np.diag(S_diag)

            # Residual (keep it 1D; don't transpose)
            resid = mean_exp_scaled - mean_obs_train_scaled             # (n_train,)
            
            # Posterior predictive mean
            mean_zetagiveny = mean_obs_predict_scaled + K12 @ np.linalg.solve(S, resid)  # (m_pred,)
            
            # Posterior predictive covariance
            SK12T = np.linalg.solve(S, K12.T)                           # (n_train, m_pred)
            covariance_zetagiveny = K22 - K12 @ SK12T                   # (m_pred, m_pred)
            
            # Optional: symmetrize to clean tiny numerical asymmetries
            covariance_zetagiveny = 0.5*(covariance_zetagiveny + covariance_zetagiveny.T)

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
            obs_start_train = obs_end_train
            obs_start_predict = obs_end_predict
            phi_start = phi_end

        # now shape of unscaled_realization_allobs is
        # [ (n_realizations, n_obs1), (n_realizations, n_obs2), … ]
        # we want to stitch these horizontally so each row is all observables:
        # result -> (n_realizations, total_observations)
        sample_realizations = np.hstack(unscaled_realization_allobs)

        # collect for all samples
        unscaled_realization_all.append(sample_realizations)

    # final shape: (batch_size * n_realizations, total_observations)
    return np.vstack(unscaled_realization_all)

# ----------------------------------------------------------------------------------

def quantiles_modelPlusGP(samples, MDclass, x_predict, model_predict_func, save_filename, n_realizations=100, ncores=-1, quantiles=[50, 2.5, 97.5, 16, 84]):
    """
    Computes quantiles for the given samples by generating realizations in parallel.
    
    Parameters:
    - samples: numpy array of shape (total_samples, num_params)
    - MDclass : Instantiated model discrepancy class
    - x_predict : prediction points
    - model_predict_func : function for model prediction
    - save_filename: Filename to save quantiles along with directory path
    - n_realizations: Number of realizations per sample
    - ncores: Number of parallel jobs (-1 uses all available cores)
    - quantiles: List of quantiles to compute (in percent)
    
    Saves quantiles for model + discrpancy GP predictions
    """

    start_time = time.time()

    num_cores = set_njobs(ncores)
    
    batch_size = int(np.ceil(len(samples)/num_cores))
    
    # Split samples into batches
    batches = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]
    
    # Parallel processing of batches
    logger.info("Computing quantiles for model + GP predictions...")
    logger.info(f"Total samples: {len(samples)}. Total cores: {num_cores}. Batch size: {batch_size}. Total batches: {len(batches)}")
    results = Parallel(n_jobs=num_cores)(
        delayed(realization_modelPlusGP)(MDclass, x_predict, model_predict_func, batch, n_realizations) for batch in batches
    )
    
    # Concatenate all realizations
    all_realizations = np.vstack(results)  # Shape: (total_samples * n_realizations, num_observables)
    logger.info(f"All realizations shape: {all_realizations.shape}")
    
    # Compute quantiles
    quantiles_values = np.percentile(all_realizations, quantiles, axis=0)  # Shape: (len(quantiles), num_observables)

    header_str =  "# Quantiles: " + ", ".join(str(q) for q in quantiles)
    np.savetxt(save_filename, quantiles_values, header=header_str, fmt="%.8f")

    end_time = time.time()
    logger.info(f"Time taken: {end_time - start_time:.2f} seconds. Quantiles saved in {save_filename}\n")

# ----------------------------------------------------------------------------------
