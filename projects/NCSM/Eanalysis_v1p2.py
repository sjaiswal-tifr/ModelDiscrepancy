#!/usr/bin/env python3
import os, sys
import numpy as np
from scipy.stats import beta
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern

src_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "src"))
sys.path.append(src_dir)
from ModelDiscrepancy import ModelDiscrepancy as MD
from sampling_methods import pocomc_sampling as pocomc

# ======================================================================================
observables = None
experimental_data = None
model_param_bounds = None  # Model parameter bounds
HP_bounds = None  # GP hyperparameter bounds
# ======================================================================================

# Define function for general beta priors 
def log_prior_param(param, alpha_val, beta_val, param_min, param_max):
    """
    Function for setting log priors for parameter based on a Beta distribution.
    """
    scale = param_max - param_min  # Rescaling factor

    if param_min < param < param_max:
        # Rescale param to the range [param_min, param_max]
        param_rescaled = (param - param_min) / scale
        
        prior = beta.pdf(param_rescaled, alpha_val, beta_val) / scale
        return np.log(prior)
    else:
        return -np.inf
        
# ===========================================================================================

def MD_kernelE(X1, X2, cbar,
               s_Lambda, r_Lambda, l_Lambda, 
               s_L,      r_L,      l_L):
    """
    Kernel over (Lambda,L):
    K = ( cabr^2 + (Exp[-(Lam^2 / s_Lam^2)^r_Lam] * Exp[-(L^2 / s_L^2)^r_L])^2 ) * Matern32(Lam; l_Lam) * Matern32(L; l_L)
    """
    # unpack
    Lambda, L = X1[:, 0], X1[:, 1]

    # amplitude envelopes: the marginal variances
    aLambda = np.exp(- (Lambda / s_Lambda)**r_Lambda)                   # shape (n,)
    aL      = np.exp(- (L      / s_L     )**r_L)                        # shape (n,)
    sigma   = np.outer(aLambda, aLambda) * np.outer(aL, aL) # shape (n, n)

    # the coorelations
    c_Lambda = Matern(length_scale=l_Lambda, nu=1.5)(Lambda[:, None], Lambda[:, None])
    c_L   = Matern(length_scale=l_L, nu=1.5)(L[:, None],  L[:, None])

    # full kernel
    K = ( (cbar**2) + sigma) * c_Lambda * c_L

    # add small jitter to diagonal of K
    idx = np.diag_indices_from(K)
    K[idx] += 1e-12 * np.abs(K[idx])
    
    return K

# ===========================================================================================

# # Hypermarameter bounds :  cbar, s_Lambda, r_Lambda, l_Lambda, s_L, r_L, l_L      
# HP_bounds = np.array([
#                       [ 0.0   ,  5.0   ],    # for cbar
#                       [ 0.001 ,  50.0  ],    # for s_Lambda
#                       [ 0.0   ,  15.0  ],    # for r_Lambda
#                       [ 0.01  ,  50.0  ],    # for l_Lambda
#                       [ 0.001 ,  50.0  ],    # for s_L
#                       [ 0.0   ,  15.0  ],    # for r_L
#                       [ 0.01  ,  50.0  ]     # for l_L
#                       ])

# ===========================================================================================

def get_exp_dict(experimental_data, observables):
    """
    Create dictionary to pass to ModelDiscrepancy class for only active observables.

    Returns:
       - x (2d array): array with values of input space (pT/centrality/N_max/time)
       - y (nd array): array with values of output observables
       - exp_dict (dict): dictionary with all observables provided in experimental_data
       - exp_active_dict (dict): dictionary with observable considered in analysis
    """
    
    x = experimental_data[:, 1:]
    y = experimental_data[:, :1]
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    exp_dict = {}  # Create the dictionary structure
    
    # Iterate over observables
    for i, obs in enumerate(observables):
        mean_values = y[:, i]  # Mean values
        std_values = np.random.uniform(1e-8, 1e-6, size=mean_values.shape)   # Small random values
        
        exp_dict[obs] = {
            "x": x.tolist(),
            "mean": mean_values.tolist(),
            "std": std_values.tolist()
        }

    # Dictionary to pass to ModelDiscrepancy class. Remove observables that are False
    exp_active_dict = {obs: data for obs, data in exp_dict.items() if observables[obs]}
    return x, y, exp_dict, exp_active_dict

# ======================================================================================

def init_globals():
    """
    Initialize global variables that will be used by model_predict():
       - x (1d array): array with values of input space (pT/centrality/N_max/time)
       - y (2d array): array with values of output observables
       - exp_dict (dict): dictionary with all observables provided in experimental_data
       - exp_active_dict (dict): dictionary with observable considered in analysis
       - active_observables (list): list with names of active observables
    """
    global x, y, exp_dict, exp_active_dict, active_observables
    
    x, y, exp_dict, exp_active_dict = get_exp_dict(experimental_data, observables)
    active_observables = list(exp_active_dict.keys())

# ======================================================================================

def fE(Lambda, L, Einf, A0, A1, A2, kinf):
    """
    Model for E as a function of (Lambda, L) and parameters Einf, A0, A1, A2, kinf.
    """
    EUV = A0 * np.exp(- 2.0 * Lambda**2 / A1**2)
    EIR = A2 * np.exp(- 2.0 * kinf * L)
    return Einf + EUV + EIR
    
def model_predict(theta):
    """
    Function for model prediction compatible with ModelDiscrepancy class
    """
    Lambda, L = x[:,0], x[:,1]
    Einf, A0, A1, A2, kinf = theta[0], theta[1], theta[2], theta[3], theta[4]
    funcE = fE(Lambda, L, Einf, A0, A1, A2, kinf)

    size = len(active_observables) * len(x)
    return funcE, np.full(size, 0)
    
# ======================================================================================

def pocomc_sampling(save_dir,
                    n_eff_mult=500, n_act_mult=200, 
                    n_steps_mult=1, n_total=10000,
                    save_every_n=10, resume=False, ncores=-1):
    """
    pocoMC sampling with model discrepancy.
    """
    
    # Define a list of log_prior functions for model parameters
    log_prior_theta = [
        lambda param, i=i: log_prior_param(
            param, alpha_val=1.1, beta_val=1.1,  # for flat priors
            param_min=model_param_bounds[i][0], param_max=model_param_bounds[i][1]
        )
        for i in range(len(model_param_bounds))
    ]
    # -----------------------------------------------

    # Define a list of log_prior functions GP hyperparmeters
    log_prior_cbar     = lambda param: log_prior_param(param, alpha_val=1.1, beta_val=1.1, param_min=HP_bounds[0][0], param_max=HP_bounds[0][1])
    log_prior_s_Lambda = lambda param: log_prior_param(param, alpha_val=1.1, beta_val=1.1, param_min=HP_bounds[1][0], param_max=HP_bounds[1][1])
    log_prior_r_Lambda = lambda param: log_prior_param(param, alpha_val=1.1, beta_val=1.1, param_min=HP_bounds[2][0], param_max=HP_bounds[2][1])
    log_prior_l_Lambda = lambda param: log_prior_param(param, alpha_val=1.1, beta_val=1.1, param_min=HP_bounds[3][0], param_max=HP_bounds[3][1])
    log_prior_s_L      = lambda param: log_prior_param(param, alpha_val=1.1, beta_val=1.1, param_min=HP_bounds[4][0], param_max=HP_bounds[4][1])
    log_prior_r_L      = lambda param: log_prior_param(param, alpha_val=1.1, beta_val=1.1, param_min=HP_bounds[5][0], param_max=HP_bounds[5][1])
    log_prior_l_L      = lambda param: log_prior_param(param, alpha_val=1.1, beta_val=1.1, param_min=HP_bounds[6][0], param_max=HP_bounds[6][1])
    # -----------------------------------------------

    # Define kernel dict compatible with ModelDiscrepancy for active observables 
    kernel = lambda X1, X2, cbar, s_Lambda, r_Lambda, l_Lambda, s_L, r_L, l_L :  MD_kernelE(X1, X2, cbar, 
                                                                                            s_Lambda, r_Lambda, l_Lambda, 
                                                                                            s_L, r_L, l_L)
    
    HP_log_priors = [log_prior_cbar, log_prior_s_Lambda, log_prior_r_Lambda, log_prior_l_Lambda, 
                     log_prior_s_L,      log_prior_r_L,      log_prior_l_L]

    MDkernels = {obs: {"kernel": kernel, "log_priors": HP_log_priors}
                           for obs, flag in observables.items() if flag}

    # Instantiate class
    md = MD(exp_data=exp_active_dict,
            model_predict=model_predict,
            log_priors_model=log_prior_theta,
            MD=True,
            MDkernels=MDkernels)

    # -----------------------------------------------
    # Model parameters bounds (for theta):
    min_theta = model_param_bounds[:, 0]
    max_theta = model_param_bounds[:, 1]
    # Hyperparameters bounds (for phi):
    min_phi = []
    max_phi = []
    for obs in MDkernels:
        n_hp = len(MDkernels[obs]["log_priors"])  # number of hyperparameters for this observable
        # Use the first n_hp rows of HP_bounds for this observable
        min_phi.append(HP_bounds[:n_hp, 0])
        max_phi.append(HP_bounds[:n_hp, 1])
    min_phi = np.concatenate(min_phi)
    max_phi = np.concatenate(max_phi)

    # Bounds for all parameters
    min_param = np.concatenate([min_theta, min_phi])
    max_param = np.concatenate([max_theta, max_phi])
    # -----------------------------------------------

    # parameters needed for pocomc sampling ------>
    num_param = len(min_param)  
    n_effective = n_eff_mult * num_param
    n_active = n_act_mult * num_param
    n_steps = n_steps_mult * num_param
    
    samples = pocomc(min_param, max_param, 
                     log_posterior = md.log_posterior, samples_save_dir = save_dir,
                     n_effective = n_effective, n_active = n_active, 
                     n_steps = n_steps, n_total = n_total,
                     save_every_n = save_every_n, resume = resume, ncores = ncores)

    return samples, md
# ======================================================================================

