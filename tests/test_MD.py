#!/usr/bin/env python3

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sampling_methods import pocomc_sampling
from ModelDiscrepancy import ModelDiscrepancy

import numpy as np
from scipy.stats import beta
from sklearn.gaussian_process.kernels import DotProduct, RBF

# =========================================
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
)
# =========================================

def true_func(x):
    val =  x / ( 1.0 + x / 10.0 )
    return val, np.random.uniform(1e-3, 1e-6, size=np.shape(val))

def model(x, theta):
    val =  theta[0] * x
    return val, np.random.uniform(1e-5, 1e-8, size=np.shape(val))

# Generate nump points between 0 and 2
nump = 8
x_obs = np.round(np.linspace(0, 2.0, nump) + np.insert(np.random.uniform(1e-3, 1e-2, nump-1), 0, 0.0), 4)

# Create the dictionary for experimental observable
exp_dict = {} 
truth_mean, truth_std = true_func(x_obs)
exp_dict['func'] = {
    "x": x_obs.tolist(),
    "mean": truth_mean.tolist(),
    "std": truth_std.tolist()
}

# Model predictions
true_param_mean, true_param_std = 1.0, 0.0
model_predict = lambda theta: model(x=x_obs, theta=theta)
# ===========================================

def MD_kernel(X1, X2, cbar, l, r):
    """
    Compute the covariance matrix using a custom kernel:
    K(X1, X2) = cbar^2 * (X1 @ X2.T)^r * exp(-||X1 - X2||^2 / (2 * l^2)).
    """
    dot_product_term = np.dot(X1, X2.T) ** r if r > 0 else 1.0
    rbf_kernel = RBF(length_scale=l)
    rbf_term = rbf_kernel(X1, X2)
    return cbar**2 * dot_product_term * rbf_term

# Define function for general beta priors 
def log_prior_param(param, param_min, param_max):
    """
    Log prior for parameter based on a Beta distribution.
    """
    scale = param_max - param_min  # Rescaling factor
    if param_min < param < param_max:
        # Rescale param to the range [param_min, param_max]
        param_rescaled = (param - param_min) / scale
        prior = beta.pdf(param_rescaled, 1.1, 1.1) / scale
        return np.log(prior)
    else:
        return -np.inf
# ===========================================

model_bounds = np.array([[ 0.7 , 1.2 ]])  # truth = 1

# Define log_prior functions for model parameter
log_prior_theta = [lambda param: log_prior_param(param, param_min=model_bounds[0][0], param_max=model_bounds[0][1])]
# ===========================================

HP_bounds = np.array([[ 0.0 ,  5.0 ],   # for cbar
                      [ 0.0 ,  5.0 ],   # for l
                      [ 0.0 ,  5.0 ]])  # for r

# Define priors for GP hyperparmeters
log_prior_cbar = lambda param: log_prior_param(param, param_min=HP_bounds[0][0], param_max=HP_bounds[0][1])
log_prior_l    = lambda param: log_prior_param(param, param_min=HP_bounds[1][0], param_max=HP_bounds[1][1])
log_prior_r    = lambda param: log_prior_param(param, param_min=HP_bounds[2][0], param_max=HP_bounds[2][1])
# ===========================================

kernel = lambda X1, X2, cbar, l, r: MD_kernel(X1, X2, cbar, l, r)
HP_log_priors = [log_prior_cbar, log_prior_l, log_prior_r]
MDkernels = {'func': {"kernel": kernel, "log_priors": HP_log_priors}}

# Model parameters bounds (for theta):
min_theta, max_theta = model_bounds[:, 0], model_bounds[:, 1]

# Hyperparameters bounds (for phi):
min_phi, max_phi = HP_bounds[:, 0], HP_bounds[:, 1]
# ===========================================

def woMD_test():
    # bounds for all parameters
    min_param, max_param = min_theta, max_theta
    
    md = ModelDiscrepancy(exp_data=exp_dict, 
                          model_predict=model_predict, 
                          log_priors_model=log_prior_theta,
                          MD=False)
    
    samples = pocomc_sampling(min_param, max_param, 
                         log_posterior=md.log_posterior,  
                         n_effective=2000, n_active=1000, n_steps=1, n_total=3000, n_evidence=3000, 
                         samples_save_dir = None)
    return np.mean(samples[:,0]), np.std(samples[:,0])

def wMD_test():
    # bounds for all parameters
    min_param, max_param = np.concatenate([min_theta, min_phi]), np.concatenate([max_theta, max_phi])
    
    md = ModelDiscrepancy(exp_data=exp_dict, 
                          model_predict=model_predict, 
                          log_priors_model=log_prior_theta,
                          MD=True, MDkernels=MDkernels)
    
    samples = pocomc_sampling(min_param, max_param, 
                         log_posterior=md.log_posterior,  
                         n_effective=2000, n_active=800, n_steps=2, n_total=3000, n_evidence=3000, 
                         samples_save_dir = None)
    return np.mean(samples[:,0]), np.std(samples[:,0])
# ===========================================

def main():
    line1 = "-" * 75
    line2 = "=" * 35
    print(f'\n{line2}\nModel discrepancy: Starting tests\n{line2}\n')
    # =========================================
    
    failures = 0
    # tol_mean, tol_std = 0.05, 0.05  # tolerances
    
    def check(name, mean, std):
        nonlocal failures
        if not np.isfinite(mean) or not np.isfinite(std):
            print(f"\n{line1}\nFAIL {name}: non-finite stats\n{line1}\n")
            failures += 1
            return
        print(f"\n{line1}\nPASS: {name} passed. mean={mean:.3f} (target {true_param_mean:.3f}), std={std:.3f} (target {true_param_std:.3f})\n{line1}\n")
    # =========================================
    
    try:
        m, s = woMD_test()
        check("Without MD", m, s)
    except Exception as e:
        print(f"{line1}\nFAIL w/o MD: {type(e).__name__}: {e}\n{line1}\n")
        failures += 1

    try:
        m, s = wMD_test()
        check("With MD", m, s)
    except Exception as e:
        print(f"\n{line1}\nFAIL w/ MD: {type(e).__name__}: {e}\n{line1}\n")
        failures += 1

    if failures == 0:
        print(f'\n{line2}\nModel discrepancy: All test PASSED\n{line2}\n')
    else:
        print(f'\n{line2}\nModel discrepancy: {failures} test FAILED\n{line2}\n')
        
    return failures
# =========================================

if __name__ == "__main__":
    sys.exit(main())
