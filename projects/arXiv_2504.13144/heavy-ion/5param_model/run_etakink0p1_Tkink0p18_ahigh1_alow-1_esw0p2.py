#!/usr/bin/env python3

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# helps with OpenMP after fork on some systems
# os.environ.setdefault("KMP_AFFINITY", "disabled")
# os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

# import multiprocessing as mp
# # Only set if not already set by a parent / launcher
# if mp.get_start_method(allow_none=True) is None:
#     try:
#         mp.set_start_method("forkserver")  # best to avoid fork-with-threads
#     except (RuntimeError, ValueError):
#         mp.set_start_method("spawn", force=True)  # portable fallback
# -----------------------------------------------------------------

import sys
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

def main():
    
    # specify experiment ------->
    # exp = "Grad_shear0p1_esw0p2"
    exp = "Grad_etakink0p1_Tkink0p18_ahigh1_alow-1_esw0p2"
    
    # load experimental data and specify directory name to save mcmc chains ----->
    experimental_data = np.loadtxt(f'../experimental_data/{exp}.dat')
    RunDataDir = f"mcmc_data/{exp}"
    
    with gzip.open(f'../emulator_nodf.dill.gz', 'rb') as f:
        emu = dill.load(f)
    
    # ---------------------------------------------------
    
    num_model_param = 5  # number of model parameters
    
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
                
                # Compute observable prediction quantiles 
                mod = f'model'
                save_quant_name = f"{RunDataDir}/{dir_name}/{sub_dir}/quantiles_{samp}_{mod}.txt"
    
                quantiles(samples=samples[:, :num_model_param],
                          MDclass=md,
                          emulator=emu,
                          save_filename=save_quant_name,
                          whichmodel=mod,
                          n_realizations=100
                         )
    
                if sub_dir != "wo_MD":
                    mod = f'modelPlusGP'
                    save_quant_name = f"{RunDataDir}/{dir_name}/{sub_dir}/quantiles_{samp}_{mod}.txt"
                    quantiles(samples=samples,
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
    
# ======================================================================================

if __name__ == "__main__":
    main()

    