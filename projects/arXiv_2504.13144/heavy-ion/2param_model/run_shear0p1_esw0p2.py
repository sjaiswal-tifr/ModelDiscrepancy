#!/usr/bin/env python3
import os, sys
import numpy as np
import dill
import gzip

# Add the parent directory and src directory to the system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

src_dir = os.path.abspath(os.path.join(os.getcwd(), "../../../..", "src"))
sys.path.append(src_dir)
from get_quantiles import quantiles

import analysis as ana

import logging
from pathlib import Path

def setup_logging(logfile="run.log", level=logging.INFO):
    """Configure root logger to log to both console and file."""
    log_path = Path(logfile)
    log_path.parent.mkdir(parents=True, exist_ok=True)   # <-- make parent dirs

    # Clear any existing handlers
    logging.getLogger().handlers.clear()

    # Configure logging
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),                  # console
            logging.FileHandler(log_path, mode="w")   # file
        ]
    )

# ======================================================================================

# specify experiment ------->
exp = "Grad_shear0p1_esw0p2"

# load experimental data and specify directory name to save mcmc chains ----->
experimental_data = np.loadtxt(f'../experimental_data/{exp}.dat')
RunDataDir = f"mcmc_data/{exp}"

with gzip.open(f'../emulator_nodf.dill.gz', 'rb') as f:
    emu = dill.load(f)

# ---------------------------------------------------

fixed_param = np.array([0.21, 0.0, 0.0])  # fix T_kink, a_low, a_high
ana.fixed_param = fixed_param

num_model_param = 2  # number of model parameters

# ---------------------------------------------------

# # normalMode: May be enough ---------------------->
# n_eff_mult = 200
# n_act_mult = 80  # 0.4 x n_eff
# n_steps_mult = 2

# beastMode: Default ------------------------------->
n_eff_mult = 500
n_act_mult = 150  # 0.3 x n_eff
n_steps_mult = 2

# # godMode. This is waste of resource ------------->
# n_eff_mult = 1000
# n_act_mult = 300  # 0.3 x n_eff
# n_steps_mult = 3

# ---------------------------------------------------

ana.experimental_data = experimental_data
ana.emu = emu
ana.RunDataDir = RunDataDir

# ======================================================================================

def save_res(observables, dir_name, sampling_methods, sample_settings):
    """
    Run the specified sampling methods and save quantile results.
    
    Parameters
    ----------
    observables : Any
        (Unused in this snippet; include as needed.)
    dir_name : str
        Directory name to use for the sampling outputs.
    sampling_methods : list of tuples
        Example:
            [
                (ana.pocomc_sampling_woMD, "wo_MD", True),
                (ana.pocomc_sampling_wMD_kernel1, "w_MD_kernel1", True),
                (ana.pocomc_sampling_wMD_kernel2, "w_MD_kernel2", False)
            ]
    """
    
    for method, sub_dir, run in sampling_methods:
        if not run:
            continue  # Skip methods not marked to run
        for samp, resume in sample_settings:
            samples, md = method(dir_name,
                                 n_effective_mltpl=n_eff_mult, 
                                 n_active_mltpl=n_act_mult,
                                 n_steps_mltpl=n_steps_mult,
                                 n_total=samp,
                                 n_evidence=samp,
                                 save_every_n=save_every_n,
                                 resume=resume
                                )

            # Repeat fixed_param for each row in samples
            fixed_params_repeated = np.tile(fixed_param, (samples.shape[0], 1))
            # Concatenate the fixed parameters in front of samples1
            samples_fixed_params = np.concatenate((fixed_params_repeated, samples), axis=1)

            mod = f'model'
            save_quant_name = f"{RunDataDir}/{dir_name}/{sub_dir}/quantiles_{samp}_{mod}.txt"

            emu_input = len(fixed_param) + num_model_param
            quantiles(
                samples=samples_fixed_params[:,:emu_input],
                MDclass=md,
                emulator=emu,
                save_filename=save_quant_name,
                whichmodel=mod,
                n_realizations=100
            )

            if sub_dir != "wo_MD":
                mod = f'modelPlusGP'
                save_quant_name = f"{RunDataDir}/{dir_name}/{sub_dir}/quantiles_{samp}_{mod}.txt"
                quantiles(
                    samples=samples_fixed_params,
                    MDclass=md,
                    emulator=emu,
                    save_filename=save_quant_name,
                    whichmodel=mod,
                    n_realizations=100
                )

# ======================================================================================
# ======================================================================================

save_every_n = 10  # saves sampler state after every save_every_n
# Define the sample size and resume settings for each run.
sample_settings = [
    (20000, False),  # int: save after this many samples. Bool: whether to resume sampling from saved state.
    (30000, True),
    (40000, True)
]

# --------------------------------------------------------------------------------------

observables= {
	'CH:dN_d2pT' :  True,
	'CH:v22' :      False,
	'CH:v32' :      False,
    'pi+:dN_d2pT' : False,
	'pi+:v22' :     False,
    'p:dN_d2pT' :   False,
	'p:v22' :       False,
    }

dir_name = "obs_CH_dNdpT"
setup_logging(f"logs/{dir_name}.log")

sampling_methods = [
    (ana.pocomc_sampling_woMD, "wo_MD",               True),
    (ana.pocomc_sampling_wMD_kernel1, "w_MD_kernel1", True),
    (ana.pocomc_sampling_wMD_kernel2, "w_MD_kernel2", True)
]

# Call the initialization function to reinitialize globals that depend on these values:
ana.observables = observables
ana.init_globals()

save_res(observables, dir_name, sampling_methods, sample_settings)

# --------------------------------------------------------------------------------------

observables= {
	'CH:dN_d2pT' :  True,
	'CH:v22' :      True,
	'CH:v32' :      True,
    'pi+:dN_d2pT' : False,
	'pi+:v22' :     False,
    'p:dN_d2pT' :   False,
	'p:v22' :       False,
    }

dir_name = "obs_CH_dNdpT_v2_v3"
setup_logging(f"logs/{dir_name}.log")

sampling_methods = [
    (ana.pocomc_sampling_woMD, "wo_MD",               True),
    (ana.pocomc_sampling_wMD_kernel1, "w_MD_kernel1", True),
    (ana.pocomc_sampling_wMD_kernel2, "w_MD_kernel2", True)
]

# Call the initialization function to reinitialize globals that depend on these values:
ana.observables = observables
ana.init_globals()

save_res(observables, dir_name, sampling_methods, sample_settings)

# --------------------------------------------------------------------------------------

observables= {
	'CH:dN_d2pT' :  True,
	'CH:v22' :      True,
	'CH:v32' :      True,
    'pi+:dN_d2pT' : True,
	'pi+:v22' :     True,
    'p:dN_d2pT' :   True,
	'p:v22' :       True,
    }

dir_name = "obs_CH_pi+_p"
setup_logging(f"logs/{dir_name}.log")

sampling_methods = [
    (ana.pocomc_sampling_woMD, "wo_MD",               True),
    (ana.pocomc_sampling_wMD_kernel1, "w_MD_kernel1", True),
    (ana.pocomc_sampling_wMD_kernel2, "w_MD_kernel2", True)
]

# Call the initialization function to reinitialize globals that depend on these values:
ana.observables = observables
ana.init_globals()

save_res(observables, dir_name, sampling_methods, sample_settings)

# --------------------------------------------------------------------------------------
