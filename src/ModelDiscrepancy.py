#!/usr/bin/env python3
'''
Created January 2025

@author: Sunil Jaiswal (sjaiswal.tifr@gmail.com)

Reference:
    This implementation is based on the methodology proposed in the following paper:
    Sunil Jaiswal et al., "Bayesian model-data comparison incorporating theoretical uncertainties", arXiv: 2504.13144.
    Available at: https://arxiv.org/abs/2504.13144
'''

import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
import logging
logger = logging.getLogger(__name__)

# ===================================================================

class ModelDiscrepancy():
    """
    ModelDiscrepancy class to incorporate theoretical uncertainties, if desired.
    Output data are standardized before computing likelihood.
    
    Parameters:
    -----------
    exp_data (dict): A dictionary containing the experimental data.
        - The first-level keys correspond to observable names.
        - Each observable's value is another dictionary with the following keys:
            - 'x': The independent variable(s). Can be 1-d or 2-d. 
              If 2-d, rows should be coordinates of the observed mean and std.
            - 'mean': The observed mean values at the corresponding 'x'.
            - 'std': The standard deviations of the observed values at 'x'.
        Example: exp_data = {
                                "observable_1": {
                                    "x": [0.1, 0.2, 0.3, 0.4], 
                                    "mean": [1.5, 1.7, 1.6, 1.8],
                                    "std": [0.1, 0.1, 0.15, 0.2]
                                },
                                "observable_2": {...},
                                ...
                            }
    
    model_predict : callable
        A function taking `theta` (1D NumPy array) and returning (model_mean, model_std),
        each a 1D NumPy array of length self.total_observations.

    log_priors_model : list of callables
        A list of log prior functions for each model parameter.

    MD : bool, optional
        Whether to apply a discrepancy model. Default is False.

    MDkernels : dict, optional
        Dictionary of GP kernels and their hyperparameter priors for each observable if MD=True.
        - The first-level keys correspond to observable names. Must be same as in exp_data.
        - Each observable's value is another dictionary with the following keys:
            - 'kernel': Kernel function for discrepancy GP for that observable.
            - 'log_priors': log priors for hyperparameters for kernel as functions in a list.
        Example: MDkernels = {
                            "observable_1": {
                                "kernel": kernel_function, 
                                "log_priors": [log_prior_cbar, log_prior_l, log_prior_r]
                            },
                            "observable_2": {...},
                            ...
                        }
    Raises:
    -------
    ValueError:
        If mismatch in lengths of 'x', 'mean', 'std' for any observable,
        or if `log_priors_model` is not a list of callables,
        or if `MD=True` but `MDkernels` is None,
        or if mismatch in structure of `MDkernels`,
        or if 'kernel' or 'log_priors' is missing for any observable in MDkernels.
    """
    
    def __init__(self, exp_data, model_predict, log_priors_model, MD = False, MDkernels = None):

        logger.info("================= ModelDiscrepancy class instantiated =================")

        # Store references
        self.exp_data = exp_data
        self.model_predict = model_predict
        self.log_priors_model = log_priors_model
        self.MD = MD
        self.MDkernels = MDkernels

        if not isinstance(self.log_priors_model, list):
            raise ValueError("'log_priors_model' must be a list of callable functions.")
        self.num_model_parameters = len(self.log_priors_model)  # Number of model parameters
        
        # Basic structure checks for exp_data
        self.observables = list(self.exp_data.keys())  # Observable names in a list
        self.num_observables = len(self.observables)  # Number of observables
        self.total_observations = 0  # total observations across all observables
        self._validate_exp_data()

        # Validate model with a simple trial parameter vector
        self._validate_model()

        # ----------------------------------------------------

        # If MD, validate kernels and store kernel/log-priors info in list form (to avoid repeated dict lookups)
        self.total_hyperparameters = 0
        self.log_priors_MDGP = []
        self.MD_kernels = []
        self.MD_hp_counts = []
        if self.MD:
            if self.MDkernels is None:
                raise ValueError("MDkernels must be provided when MD=True.")
            self._validate_MDkernels()

            # store kernel/log-priors info in list form (to avoid repeated dict lookups)
            for obs in self.observables:
                kernel_func = self.MDkernels[obs]["kernel"]
                prior_funcs = self.MDkernels[obs]["log_priors"]
                self.MD_kernels.append(kernel_func)
                self.MD_hp_counts.append(len(prior_funcs))

        # ----------------------------------------------------
        
        # Total parameters = model parameters + discrepancy hyperparameters
        self.total_parameters = self.num_model_parameters + self.total_hyperparameters

        # Prepare lists for scaled exp data for observables
        self.scaled_exp_data = []  # list of dicts, each containing scaled info + offsets

        # Standardize the experimental data (y-values) for each observable
        self._standardize_exp_data()

        # ----------------------------------------------------
        
        logger.info("All inputs validated. All checks passed.")
        if self.MD:
            logger.info("MD = True. Theoretical uncertainities will be considered.")
        else:
            logger.info("MD = False. Theoretical uncertainities will not be considered.")
        logger.info(f"Observables: {self.observables}.")
        logger.info(f"Total observations: {self.total_observations}.")
        logger.info(f"Number of model parameters: {self.num_model_parameters}.")
        if self.MD:
            logger.info(f"Total discrepancy GP hyperparameters: {self.total_hyperparameters}.")
        logger.info("=======================================================================\n")

    # ===========================================================================================
            
    def _validate_exp_data(self):
        """
        Ensure that for each observable:
          1) x can be coerced to a 1-D or 2-D float array of coordinates,
          2) mean and std can each be coerced to 1-D float arrays,
          3) the number of coordinates (rows) in x matches len(mean) and len(std),
          4) accumulate the total number of observations across all observables.
        """
        total_obs = 0
        for obs in self.observables:
            data = self.exp_data[obs]
            x, mean, std = data['x'], data['mean'], data['std']

            # 1) x -> ndarray of floats
            try:
                xarr = np.asarray(x, dtype=float)
            except Exception as e:
                raise ValueError(f"For observable '{obs}': could not convert x to float array: {e}")
            if xarr.ndim == 0:
                raise ValueError(f"For observable '{obs}': x must be 1-D or 2-D, got scaler value.")
            if xarr.ndim > 2:
                raise ValueError(f"For observable '{obs}': x must be 1-D or 2-D, got ndim={xarr.ndim}")
    
            # 2) mean -> 1-D float array
            try:
                mean_arr = np.asarray(mean, dtype=float)
            except Exception as e:
                raise ValueError(f"For observable '{obs}': could not convert mean to 1-D float array: {e}")
            if mean_arr.ndim != 1:
                raise ValueError(f"For observable '{obs}': mean must be 1-D, got ndim={mean_arr.ndim}")
    
            # 3) std  -> 1-D float array
            try:
                std_arr = np.asarray(std, dtype=float)
            except Exception as e:
                raise ValueError(f"For observable '{obs}': could not convert std to 1-D float array: {e}")
            if std_arr.ndim != 1:
                raise ValueError(f"For observable '{obs}': std must be 1-D, got ndim={std_arr.ndim}")
    
            # 4) length consistency: ensure the number of coordinates is x matches length of mean & std
            n_coords = xarr.shape[0]
            if not (n_coords == len(mean_arr) == len(std_arr)):
                raise ValueError(
                    f"Length mismatch in observable '{obs}': "
                    f"n_coords={n_coords}, len(mean)={len(mean_arr)}, len(std)={len(std_arr)}"
                )
            
            total_obs += n_coords
    
        self.total_observations = total_obs

    # -------------------------------------------------------------------------------------------

    def _validate_model(self):
        """
        Check the model_predict function using a trial theta vector. Ensures:
          - model_predict runs without error,
          - returns two 1D arrays (model_mean, model_std),
          - both arrays have length == self.total_observations.
        """
        theta_trial = np.zeros(self.num_model_parameters, dtype=float) + 1e-8
        try:
            # Check log priors can handle these trial values
            for i, prior_func in enumerate(self.log_priors_model):
                lp = prior_func(theta_trial[i])
        except Exception as e:
            raise ValueError(f"Error calling log priors for model parameters: {e}")

        # Now test the model_predict
        try:
            model_mean, model_std = self.model_predict(theta_trial)
        except Exception as e:
            raise ValueError(f"Error running model_predict with trial theta={theta_trial}: {e}")

        if model_mean.ndim != 1 or model_std.ndim != 1:
            raise ValueError("model_predict must return two 1D arrays (model_mean, model_std).")

        if len(model_mean) != self.total_observations or len(model_std) != self.total_observations:
            raise ValueError(
                f"model_predict do not match total_observations={self.total_observations}. "
                f"Got lengths: model_mean={len(model_mean)}, model_std={len(model_std)}."
            )

    # -------------------------------------------------------------------------------------------

    def _standardize_exp_data(self):
        """
        For each observable, store:
            - x (unscaled) in shape (n_obs, 1)
            - scaled mean_exp
            - scaled std_exp
            - offset = y_scaler.mean_[0]
            - scale  = y_scaler.scale_[0]
        in a list self.exp_data so we can do fast lookups.
        """
        for obs in self.observables:
            data = self.exp_data[obs]
            # x = np.array(data["x"], dtype=float).reshape(-1, 1)       # shape (n_obs, 1)
            ##--- SJ: making changes for more than 1-d input space
            x = np.asarray(data["x"], dtype=float)  # shape (n_obs, inputspace_dim)
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            ##---
            mean_exp = np.array(data["mean"], dtype=float).reshape(-1, 1)
            std_exp = np.array(data["std"], dtype=float)

            n_obs = x.shape[0]
            # inputspace_dim = x.shape[1]  
            # logger.info(f"For observable '{obs}': input space dimension: {inputspace_dim}.")
            
            # Fit standard scaler for y data
            y_scaler = StandardScaler()
            mean_exp_scaled = y_scaler.fit_transform(mean_exp).ravel()  # shape (n_obs,)

            # For the standard deviation, we just divide by the scale
            offset = float(y_scaler.mean_[0])  # single float
            scale = float(y_scaler.scale_[0])
            std_exp_scaled = std_exp / scale

            # Store it in a list (dict or tuple) for quick access
            self.scaled_exp_data.append({
                "n_obs"           : n_obs,
                "x"               : x,
                "mean_exp_scaled" : mean_exp_scaled,
                "std_exp_scaled"  : std_exp_scaled,
                "offset"          : offset,
                "scale"           : scale,
            })

    # -------------------------------------------------------------------------------------------

    def _validate_MDkernels(self):
        """
        Ensure MDkernels structure matches exp_data and includes 'kernel' + 'log_priors'.
        Accumulate all discrepancy hyperparameters.
        """
        if set(self.MDkernels.keys()) != set(self.observables):
            raise ValueError(
                "The onservable keys in MDkernels must match the keys in exp_data. "
                f"Expected: {self.observables}, got: {list(self.MDkernels.keys())}."
            )

        for obs in self.observables:
            kernel_data = self.MDkernels[obs]
            if "kernel" not in kernel_data:
                raise ValueError(f"Missing 'kernel' in MDkernels for '{obs}'.")
            if "log_priors" not in kernel_data:
                raise ValueError(f"Missing 'log_priors' in MDkernels for '{obs}'.")
                
            lp_list = kernel_data["log_priors"]
            if not isinstance(lp_list, list):
                raise ValueError(
                    f"'log_priors' must be a list of callables for observable '{obs}'. "
                    f"Got: {type(lp_list)}."
                )
            self.log_priors_MDGP.extend(lp_list)

        self.total_hyperparameters = len(self.log_priors_MDGP)

    # -------------------------------------------------------------------------------------------
    
    def _robust_cholesky(self, cov_total, eps=1e-12):
        """
        This function is called when Cholesky decompostion of cov_total fails.
        Attempts to compute Cholesky factorization by first adding jitter at different
        levels (based on 1e-4, 1e-3, 1e-2 times the minimum of the diagonal elements)
        to enforce positive definiteness. If fails, does SVD-based regularization by
        clamping small eigenvalues to ensure the matrix is SPD.

        Parameters:
            cov_total (np.ndarray): The covariance matrix (assumed symmetric).
            cov_diag (np.ndarray): The diagonal elements of cov_total.
            eps (float): The minimum allowed eigenvalue in the SVD-based correction.

        Returns:
            L (np.ndarray): Lower-triangular matrix such that L @ L.T approximates cov_total.
        """
        cov_total = (cov_total + cov_total.T) / 2.0 # Ensure symmetry before proceeding
        cov_diag = np.diag(cov_total)
        min_cov_diag = np.min(cov_diag)
        # Define jitter multipliers to try
        jitter_multipliers = [1e-5, 1e-4, 1e-3, 1e-2]

        for mult in jitter_multipliers:
            jitter = mult * min_cov_diag
            try:
                L = np.linalg.cholesky(cov_total + jitter * np.eye(cov_total.shape[0]))
                logger.info(f"Cholesky succeeded with jitter = {jitter:.1e} ({mult:.1e} x min[cov_diagonal]).")
                return L
            except np.linalg.LinAlgError:
                pass  # try the next jitter level
    
        # If none of the jitter values work, fall back to SVD-based correction.
        logger.info("Adaptive jitter failed; using SVD-based correction by clamping small eigenvalues..")
        w, V = np.linalg.eigh(cov_total)
        # Try increasing eps adaptively until the matrix becomes SPD.
        epsilon = eps
        max_eps = 1e-6  # Maximum allowed epsilon (can be adjusted)
        while True:
            w_clamped = np.clip(w, epsilon, None)
            # Reconstruct the covariance matrix with clamped eigenvalues
            cov_fixed = (V * w_clamped) @ V.T
            # Ensure symmetry
            cov_fixed = (cov_fixed + cov_fixed.T) / 2.0
            try:
                L = np.linalg.cholesky(cov_fixed)
                logger.info(f"SVD-based correction succeeded with epsilon = {epsilon:.1e}")
                return L
            except np.linalg.LinAlgError:
                epsilon *= 10
                if epsilon > max_eps:
                    raise ValueError("SVD-based correction failed: matrix is not positive definite even after clamping eigenvalues.")
                
                
    
    # ===========================================================================================

    def log_likelihood(self, theta, phi):
        """
        Calculate the log-likelihood under the assumption that each observable 
        is independent and can be handled in a block-diagonal manner.

        Parameters
        ----------
        theta : np.ndarray (1D)
            Model parameters (unscaled).
        phi : np.ndarray (1D)
            Discrepancy hyperparameters if MD=True, else empty.

        Returns
        -------
        float
            The total log-likelihood over all observables.
        """
        # Get model predictions (unscaled)
        model_mean, model_std = self.model_predict(theta)

        # Initialize log-likelihood accumulator
        log_likelihood_value = 0.0
        
        # Indices for slicing
        obs_start = 0
        phi_start = 0

        # Loop over each observable
        for i in range(self.num_observables):
            obs_info        = self.scaled_exp_data[i]
            n_obs           = obs_info["n_obs"]
            x               = obs_info["x"]
            mean_exp_scaled = obs_info["mean_exp_scaled"]
            std_exp_scaled  = obs_info["std_exp_scaled"]
            offset          = obs_info["offset"]
            scale           = obs_info["scale"]
            
            obs_end = obs_start + n_obs

            # Extract model predictions for this observable
            mean_pred_obs = model_mean[obs_start:obs_end]   # unscaled
            std_pred_obs  = model_std[obs_start:obs_end]    # unscaled

            # Scale model mean and std
            mean_pred_obs_scaled = (mean_pred_obs - offset) / scale
            std_pred_obs_scaled  = std_pred_obs / scale

            # The base diagonal = experimental variance + model variance (in scaled space)
            cov_diag = std_exp_scaled**2 + std_pred_obs_scaled**2

            # Residual
            residual = mean_exp_scaled - mean_pred_obs_scaled
            
            if self.MD:
                # We have a GP kernel for this observable
                kernel_func = self.MD_kernels[i]
                hp_count = self.MD_hp_counts[i]
                phi_end = phi_start + hp_count

                # Build the discrepancy covariance matrix
                try:
                    cov_total = kernel_func(x, x, *phi[phi_start:phi_end])
                except Exception as e:
                    raise ValueError(f"Error in GP kernel for obs index={i}: {e}")

                # Add cov_diag to the diagonals
                np.fill_diagonal(cov_total, cov_total.diagonal() + cov_diag)
                
                # Cholesky factor
                try:
                    L = np.linalg.cholesky(cov_total)
                except np.linalg.LinAlgError:
                    try:
                        # try with added small jitter to diagonals of the matrix
                        min_cov_diag = np.min(np.diag(cov_total))
                        jitter = 1e-6 * min_cov_diag
                        L = np.linalg.cholesky(cov_total + jitter * np.eye(cov_total.shape[0]))
                    except np.linalg.LinAlgError:
                        # fallback to robust_cholesky
                        logger.info("In Likelihood computation: Cholesky decomposition failed; applying jitter/SVD-based regularization..")
                        L = self._robust_cholesky(cov_total=cov_total, eps=1e-12)

                # Solve L y = residual -> y = L^-1 residual
                y = np.linalg.solve(L, residual)
                md2 = y @ y  # Mahalanobis distance^2

                # log(det(K)) from Cholesky
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                
                # Gaussian log-likelihood for this block
                log_likelihood_value += -0.5 * (md2 + log_det + n_obs * np.log(2.0 * np.pi))

                # md2 = mahalanobis(mean_exp_scaled, mean_pred_obs_scaled, np.linalg.inv(cov_total))**2
                # log_likelihood_value += -0.5 * (md2 + np.log(np.linalg.det(cov_total)) + n_obs * np.log(2.0 * np.pi) )
                
                # Advance hyperparameter phi to next observable
                phi_start = phi_end

            else:
                # No discrepancy => covariance is purely diagonal
                # Then the likelihood is an N-dimensional normal with diagonal cov_diag
                # Mahalanobis distance^2 in diagonal: sum( (r^2 / var_i) )
                md2 = np.sum((residual**2) / cov_diag)
                log_det = np.sum(np.log(cov_diag))
                log_likelihood_value += -0.5 * (md2 + log_det + n_obs * np.log(2.0 * np.pi))

            obs_start = obs_end

        return log_likelihood_value

    # -------------------------------------------------------------------------------------------

    def log_posterior(self, theta_and_phi):
        """
        Calculate the log posterior = sum of log priors + log likelihood.

        Parameters
        ----------
        theta_and_phi : np.ndarray (1D)
            Combined array of model params (theta) and discrepancy params (phi) if MD=True.

        Returns
        -------
        float
            The log posterior value.
        """
        theta_and_phi=np.atleast_1d(theta_and_phi)
        # if theta_and_phi.ndim != 1:
        #     raise ValueError("theta_and_phi must be a 1D array.")

        if len(theta_and_phi) != self.total_parameters:
            raise ValueError(
                f"Expected {self.total_parameters} parameters total, got {len(theta_and_phi)}."
            )

        # Split out theta and phi
        theta = theta_and_phi[: self.num_model_parameters]
        phi = theta_and_phi[self.num_model_parameters :]

        # ----------------------------------------
        # Calculate log-prior for model parameters
        # ----------------------------------------
        log_prior_val = 0.0
        for i, prior_func in enumerate(self.log_priors_model):
            lp = prior_func(theta[i])
            if lp <= -1e5:
                return -1e30
            log_prior_val += lp

        # ------------------------------------------------------
        # If MD is enabled, accumulate log-prior for phi as well
        # ------------------------------------------------------
        if self.MD:
            for i, prior_func in enumerate(self.log_priors_MDGP):
                lp = prior_func(phi[i])
                if lp <= -1e5:
                    return -1e30
                log_prior_val += lp

        # ------------------------
        # Calculate log-likelihood
        # ------------------------
        log_likelihood_val = self.log_likelihood(theta, phi)

        return log_prior_val + log_likelihood_val
        
# ===========================================================================================
