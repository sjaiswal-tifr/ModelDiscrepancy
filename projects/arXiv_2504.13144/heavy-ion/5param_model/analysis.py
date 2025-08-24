#!/usr/bin/env python3
import os, sys
import numpy as np
from scipy.stats import beta
from scipy import optimize
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern

src_dir = os.path.abspath(os.path.join(os.getcwd(), "../../../..", "src"))
sys.path.append(src_dir)
from ModelDiscrepancy import ModelDiscrepancy as MD
from sampling_methods import pocomc_sampling as pocomc

# ======================================================================================

observables = None
experimental_data = None
emu = None
RunDataDir = None

# ======================================================================================

def MD_kernel(X1, X2, cbar, l, r, s):
    """
    Compute the covariance matrix using a custom kernel:
    K(X1, X2) = cbar^2 * (X1 @ X2.T)^r * exp(-||X1 - X2||^2 / (2 * l^2)).

    This kernel serves as the core of the Model Discrepancy framework and should be modified
    to suit specific problems.

    Parameters:
        - X1 (numpy.ndarray): A matrix where each row represents an input vector.
        - X2 (numpy.ndarray): A matrix where each row represents an input vector.
        - cbar (float): The marginal variance parameter.
        - l (float): The length scale parameter for the RBF term.
        - r (float): The power applied to the dot product term.

    Returns:
        - numpy.ndarray: A covariance matrix computed using the defined kernel.
    """
    # Compute the dot product term
    dot_product_term = np.dot(X1, X2.T) ** r #if r > 0 else 1

    # Compute the RBF kernel term
    rbf_kernel = RBF(length_scale=l)
    # rbf_kernel = Matern(length_scale=l, nu=1.5)
    rbf_term = rbf_kernel(X1, X2)

    # Combine terms
    cov_matrix =  s**2 + cbar**2 * dot_product_term * rbf_term
    return cov_matrix
    
# ======================================================================================

# Define function for general beta priors 
def log_prior_param(param, alpha_val, beta_val, param_min, param_max):
    """
    Log prior for parameter based on a Beta distribution.
    """
    scale = param_max - param_min  # Rescaling factor

    if param_min < param < param_max:
        # Rescale param to the range [param_min, param_max]
        param_rescaled = (param - param_min) / scale
        
        prior = beta.pdf(param_rescaled, alpha_val, beta_val) / scale
        return np.log(prior)
    else:
        return -np.inf

# ======================================================================================

# Model parameters in iEBE-MUSIC for bayesian analysis
# format: parameter_name: label, min, max
# shear_viscosity_3_eta_over_s_T_kink_in_GeV: Teta, 0.13, 0.3                 # GeV
# shear_viscosity_3_eta_over_s_low_T_slope_in_GeV:  etaLowSlope, -3, 1        # 1/GeV
# shear_viscosity_3_eta_over_s_high_T_slope_in_GeV: etaHighSlope, -1, 3       # 1/GeV
# shear_viscosity_3_eta_over_s_at_kink: eta_0, 0.0, 0.4
# eps_switch: esw, 0.1, 0.5                                                   # GeV/fm^3

model_param_bounds = np.array([[ 0.13,  0.3 ],
                               [-3.0 ,  1.0 ],
                               [-1.0 ,  3.0 ],
                               [ 0.0 ,  0.4 ],
                               [ 0.1 ,  0.5 ]])


HP_bounds = np.array([[ 0.0 , 5.0 ],    # for cbar
                      [ 0.0 , 10.0 ],   # for l
                      [ 0.0 , 3.0 ],    # for r
                      [ 0.0 , 5.0 ]])   # for s

# Define a list of log_prior functions for model parameters
log_prior_theta = [
    lambda param, i=i: log_prior_param(
        param, alpha_val=1.01, beta_val=1.01,  # for flat priors
        param_min=model_param_bounds[i][0], param_max=model_param_bounds[i][1]
    )
    for i in range(len(model_param_bounds))
]

# Define priors for GP hyperparmeters
log_prior_cbar = lambda param: log_prior_param(param, alpha_val=1.01, beta_val=1.01, param_min=HP_bounds[0][0], param_max=HP_bounds[0][1])
log_prior_l    = lambda param: log_prior_param(param, alpha_val=1.01, beta_val=1.01, param_min=HP_bounds[1][0], param_max=HP_bounds[1][1])
log_prior_r    = lambda param: log_prior_param(param, alpha_val=1.01, beta_val=1.01, param_min=HP_bounds[2][0], param_max=HP_bounds[2][1])
log_prior_s    = lambda param: log_prior_param(param, alpha_val=1.01, beta_val=1.01, param_min=HP_bounds[3][0], param_max=HP_bounds[3][1])

# ======================================================================================

def get_exp_dict(experimental_data, observables):
    """
    Create dictionary to pass to ModelDiscrepancy class for only active observables
    """

    x_values = experimental_data[:, 0]
    
    exp_dict = {}  # Create the dictionary structure
    
    # Iterate over observables
    for i, obs in enumerate(observables):
        mean_values = experimental_data[:, 2*i + 1]  # Mean values
        std_values = experimental_data[:, 2*i + 2]   # Std values
        
        exp_dict[obs] = {
            "x": x_values.tolist(),
            "mean": mean_values.tolist(),
            "std": std_values.tolist()
        }
    # Dictionary to pass to ModelDiscrepancy class. Remove observables that are False
    exp_active_dict = {obs: data for obs, data in exp_dict.items() if observables[obs]}
    return exp_dict, exp_active_dict

# ======================================================================================

def init_globals():
    """
    Initialize global variables that will be used by model_predict()
    """
    global exp_dict, exp_active_dict, active_observables
    exp_dict, exp_active_dict = get_exp_dict(experimental_data, observables)
    active_observables = list(exp_active_dict.keys())

# ======================================================================================

def model_predict(theta):
    """
    Defining function for emulator prediction for compatibility with ModelDiscrepancy class
    """
    theta = np.atleast_2d(theta)
    mean, std = emu.predict(X_new=theta)  # Shape: (num_samples, total_x_values_across_all_observables)
    
    # Step 1: Map predictions to all observables
    full_pred_dict = {}
    col_start = 0

    for obs in observables.keys():
        num_x = len(exp_dict[obs]["x"])  # Number of x values for this observable
        col_end = col_start + num_x  # Determine slice range for this observable

        # Store in dictionary
        full_pred_dict[obs] = {
            "mean": mean[:, col_start:col_end],  # Extract relevant slice
            "std": std[:, col_start:col_end]
        }
        col_start = col_end  # Move to next observable's range

    # Step 2: Filter predictions to include only active observables
    active_pred_dict = {obs: full_pred_dict[obs] for obs in active_observables}

    # Step 3: Convert filtered predictions to two NumPy arrays
    mean_active = np.concatenate([active_pred_dict[obs]["mean"] for obs in active_observables], axis=1)
    std_active = np.concatenate([active_pred_dict[obs]["std"] for obs in active_observables], axis=1)

    return mean_active.flatten(), std_active.flatten()  # flatten to return as 1darray

# ======================================================================================

def pocomc_sampling_woMD(dir_name, 
                         n_effective_mltpl=100, 
                         n_active_mltpl=40,
                         n_steps_mltpl=1,
                         n_total=15000, n_evidence=15000, 
                         save_every_n=20, resume=False, ncores=-1):
    """
    pocoMC sampling without model discrepancy
    We are confident the model is correct at all x.
    """

    # Instantiate class
    md = MD(exp_data=exp_active_dict, 
            model_predict=model_predict, 
            log_priors_model=log_prior_theta,
            MD=False, MDkernels=None)
    
    # Model parameters bounds (for theta):
    min_param = model_param_bounds[:,0]
    max_param = model_param_bounds[:,1]

    # for pocomc
    num_param = len(min_param)  
    n_effective = n_effective_mltpl * num_param
    n_active = n_active_mltpl * num_param
    n_steps = n_steps_mltpl * num_param
    
    save_dir = f"{RunDataDir}/{dir_name}/wo_MD"

    samples = pocomc(min_param, max_param, 
                     log_posterior = md.log_posterior, samples_save_dir = save_dir,
                     n_effective = n_effective, n_active = n_active, 
                     n_steps = n_steps, n_total = n_total, n_evidence = n_evidence, 
                     save_every_n = save_every_n, resume = resume, ncores = ncores)
    # samples = np.load(f'{save_dir}/pocomc_chain_{n_total}.npy')
    
    return samples, md

# ======================================================================================

def pocomc_sampling_wMD_kernel1(dir_name, 
                                n_effective_mltpl=100, 
                                n_active_mltpl=40,
                                n_steps_mltpl=1, 
                                n_total=15000, n_evidence=15000, 
                                save_every_n=20, resume=False, ncores=-1):
    """
    pocoMC sampling with model discrepancy and kernel 1.
    We trust the model is more reliable at small x than at large x. 
    4 hyperparameters: \bar{c}, l, r, s
    """

    # Define kernel dict compatible with ModelDiscrepancy for active observables 
    kernel = lambda X1, X2, cbar, l, r, s: MD_kernel(X1, X2, cbar, l, r, s)
    HP_log_priors = [log_prior_cbar, log_prior_l, log_prior_r, log_prior_s]

    Discrepancy_kernels = {obs: {"kernel": kernel, "log_priors": HP_log_priors}
                           for obs, flag in observables.items() if flag}
        
    # Instantiate class
    md = MD(exp_data=exp_active_dict, 
            model_predict=model_predict, 
            log_priors_model=log_prior_theta,
            MD=True, 
            MDkernels=Discrepancy_kernels)

    # -----------------------------------------------
    # Model parameters bounds (for theta):
    min_theta = model_param_bounds[:, 0]
    max_theta = model_param_bounds[:, 1]
    # Hyperparameters bounds (for phi):
    min_phi = []
    max_phi = []
    for obs in Discrepancy_kernels:
        n_hp = len(Discrepancy_kernels[obs]["log_priors"])  # number of hyperparameters for this observable
        # Use the first n_hp rows of HP_bounds for this observable
        min_phi.append(HP_bounds[:n_hp, 0])
        max_phi.append(HP_bounds[:n_hp, 1])
    min_phi = np.concatenate(min_phi)
    max_phi = np.concatenate(max_phi)

    # Bounds for all parameters
    min_param = np.concatenate([min_theta, min_phi])
    max_param = np.concatenate([max_theta, max_phi])
    # -----------------------------------------------

    # for pocomc
    num_param = len(min_param)  
    n_effective = n_effective_mltpl * num_param
    n_active = n_active_mltpl * num_param
    n_steps = n_steps_mltpl * num_param
    
    save_dir = f"{RunDataDir}/{dir_name}/w_MD_kernel1"
    
    samples = pocomc(min_param, max_param, 
                     log_posterior = md.log_posterior, samples_save_dir = save_dir,
                     n_effective = n_effective, n_active = n_active, 
                     n_steps = n_steps, n_total = n_total, n_evidence = n_evidence, 
                     save_every_n = save_every_n, resume = resume, ncores = ncores)
    # samples = np.load(f'{save_dir}/pocomc_chain_{n_total}.npy')

    return samples, md

# ======================================================================================

def pocomc_sampling_wMD_kernel2(dir_name,
                                n_effective_mltpl=100, 
                                n_active_mltpl=40,
                                n_steps_mltpl=1,
                                n_total=15000, n_evidence=15000,
                                save_every_n=20, resume=False, ncores=-1):
    """
    pocoMC sampling with model discrepancy and kernel 2.
    We are confident model is correct at small x, but our confidence in its validity 
    decreases as x increases. 3 hyperparameters: \bar{c}, l, r 
    """

    # Define kernel dict compatible with ModelDiscrepancy for active observables 
    kernel = lambda X1, X2, cbar, l, r: MD_kernel(X1, X2, cbar, l, r, s=0)
    HP_log_priors = [log_prior_cbar, log_prior_l, log_prior_r]

    Discrepancy_kernels = {obs: {"kernel": kernel, "log_priors": HP_log_priors}
                           for obs, flag in observables.items() if flag}


    # Instantiate class
    md = MD(exp_data=exp_active_dict, 
            model_predict=model_predict, 
            log_priors_model=log_prior_theta,
            MD=True, 
            MDkernels=Discrepancy_kernels)

    # -----------------------------------------------
    # Model parameters bounds (for theta):
    min_theta = model_param_bounds[:, 0]
    max_theta = model_param_bounds[:, 1]
    # Hyperparameters bounds (for phi):
    min_phi = []
    max_phi = []
    for obs in Discrepancy_kernels:
        n_hp = len(Discrepancy_kernels[obs]["log_priors"])  # number of hyperparameters for this observable
        # Use the first n_hp rows of HP_bounds for this observable
        min_phi.append(HP_bounds[:n_hp, 0])
        max_phi.append(HP_bounds[:n_hp, 1])
    min_phi = np.concatenate(min_phi)
    max_phi = np.concatenate(max_phi)

    # Bounds for all parameters
    min_param = np.concatenate([min_theta, min_phi])
    max_param = np.concatenate([max_theta, max_phi])
    # -----------------------------------------------

    # for pocomc
    num_param = len(min_param)  
    n_effective = n_effective_mltpl * num_param
    n_active = n_active_mltpl * num_param
    n_steps = n_steps_mltpl * num_param
    
    save_dir = f"{RunDataDir}/{dir_name}/w_MD_kernel2"
    
    samples = pocomc(min_param, max_param, 
                     log_posterior = md.log_posterior, samples_save_dir = save_dir,
                     n_effective = n_effective, n_active = n_active, 
                     n_steps = n_steps, n_total = n_total, n_evidence = n_evidence, 
                     save_every_n = save_every_n, resume = resume, ncores = ncores)
    # samples = np.load(f'{save_dir}/pocomc_chain_{n_total}.npy')

    return samples, md

# ======================================================================================


# # # compute MAP
# # theta_median = np.median(samples, axis=(0,1))#model_param_bounds.mean(axis=1)
# # dist = lambda *args: -md.log_posterior(*args)
# # result = optimize.minimize(dist, theta_median, method = 'Nelder-Mead')
# # MAP = result['x']

# # np.set_printoptions(suppress=True)  #supress .e format in output
# # print(result)
# # print("MAP values:", MAP)

