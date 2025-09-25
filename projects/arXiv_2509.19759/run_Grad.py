#!/usr/bin/env python3

import os, sys
# Uncomment these only if you face error during sampling (like sampling stuck/error with workers) 
# SIGNIFICANTLY SLOWS DOWN SAMPLING (by ~2x-3x)
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")
# os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# -----------------------------------------------------------------

import numpy as np
import gzip, dill

src_dir = os.path.abspath(os.path.join(os.getcwd(), "../..", "src"))
sys.path.append(src_dir)
emu_dir = os.path.abspath(os.path.join(os.getcwd(), "./emulators"))
sys.path.append(emu_dir)

from get_quantiles import quantiles
import analysis as ana
from pathlib import Path

import logging

def setup_logging(logfile="run.log", level=logging.INFO):
    """Configure root logger to log to both console and file."""
    log_path = Path(logfile)
    log_path.parent.mkdir(parents=True, exist_ok=True)   # <-- make parent dirs
    logging.getLogger().handlers.clear()   # Clear any existing handlers

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
    idf = 'Grad'   # 'Grad' or 'CE'
    
    # load experimental data and specify directory name to save mcmc chains ----->
    experimental_data = np.loadtxt('./experimental_data/PbPb2760_exp_data.txt')
    RunDataDir = f"mcmc_data/PbPb2760/{idf}"
    
    # load emulator
    with gzip.open(f"./emulators/PbPb2760/Emulator_AKSGP_{idf}.dill.gz", "rb") as f:
        emu = dill.load(f)

    # ---------------------------------------------------
    num_model_param = 17  # number of model parameters
    # ---------------------------------------------------

    ana.experimental_data = experimental_data
    ana.emu = emu
    ana.RunDataDir = RunDataDir

    # ======================================================================================
    
    def save_res(observables, dir_name, sampling_methods, sample_settings,
                n_eff_mult, n_act_mult, n_steps_mult):
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
                mod = 'model'
                save_quant_name = f"{RunDataDir}/{dir_name}/{sub_dir}/quantiles_{samp}_{mod}.txt"
    
                quantiles(samples=samples[:, :num_model_param],
                          MDclass=md,
                          emulator=emu,
                          save_filename=save_quant_name,
                          whichmodel=mod,
                          n_realizations=100
                         )
    
                if sub_dir != "wo_MD":
                    mod = 'modelPlusGP'
                    save_quant_name = f"{RunDataDir}/{dir_name}/{sub_dir}/quantiles_{samp}_{mod}.txt"
                    quantiles(samples=samples,
                              MDclass=md,
                              emulator=emu,
                              save_filename=save_quant_name,
                              whichmodel=mod,
                              n_realizations=100
                             )
                    mod = 'GP'
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
    
    save_every_n = 1  # saves sampler state after every save_every_n
    # Define the sample size and resume settings for each run.
    sample_settings = [
        (20000, False),  # int: save after this many samples. Bool: whether to resume sampling from saved state.
        (30000, True),
        (40000, True)
    ]
    # --------------------------------------------------------------------------------------
    
    observables = {
        'dNch_deta'      : True,
        'dET_deta'       : True,
        'dN_dy_pion'     : True,
        'dN_dy_kaon'     : True,
        'dN_dy_proton'   : True,
        'mean_pT_pion'   : True,
        'mean_pT_kaon'   : True,
        'mean_pT_proton' : True,
        'pT_fluct'       : True,
        'v22'            : True,
        'v32'            : True,
        'v42'            : True
    }
    
    # Call the initialization function to reinitialize globals that depend on these values:
    ana.observables = observables
    ana.init_globals()
    
    dir_name = "all_obs"
    # --------------------------------------------------------------------------------------
    
    # beastMode: ------------------------------->
    n_eff_mult = 500
    n_act_mult = 150  # 0.3 x n_eff
    n_steps_mult = 2
    
    sampling_methods = [
        (ana.pocomc_sampling_woMD, "wo_MD", True),
        (ana.pocomc_sampling_wMD,  "w_MD",  False)
    ]
    
    setup_logging(f"logs/PbPb2760/{idf}/{dir_name}_wo_MD.log")
    save_res(observables, dir_name, sampling_methods, sample_settings,
            n_eff_mult, n_act_mult, n_steps_mult)

    # normalMode: ---------------------->
    n_eff_mult = 200
    n_act_mult = 80  # 0.4 x n_eff
    n_steps_mult = 2
    
    sampling_methods = [
        (ana.pocomc_sampling_woMD, "wo_MD", False),
        (ana.pocomc_sampling_wMD,  "w_MD",  True)
    ]
    
    setup_logging(f"logs/PbPb2760/{idf}/{dir_name}_w_MD.log")
    save_res(observables, dir_name, sampling_methods, sample_settings,
            n_eff_mult, n_act_mult, n_steps_mult)
    
    # ======================================================================================

    save_every_n = 1  # saves sampler state after every save_every_n
    # Define the sample size and resume settings for each run.
    sample_settings = [
        (20000, False),  # int: save after this many samples. Bool: whether to resume sampling from saved state.
        (30000, True),
        (40000, True)
    ]
    # --------------------------------------------------------------------------------------
    
    observables = {
        'dNch_deta'      : True,
        'dET_deta'       : True,
        'dN_dy_pion'     : False,
        'dN_dy_kaon'     : False,
        'dN_dy_proton'   : False,
        'mean_pT_pion'   : False,
        'mean_pT_kaon'   : False,
        'mean_pT_proton' : False,
        'pT_fluct'       : False,
        'v22'            : True,
        'v32'            : True,
        'v42'            : True
    }
    
    # Call the initialization function to reinitialize globals that depend on these values:
    ana.observables = observables
    ana.init_globals()
    
    dir_name = "dNch_dET_vn"
    # --------------------------------------------------------------------------------------
    
    # beastMode: ------------------------------->
    n_eff_mult = 500
    n_act_mult = 150  # 0.3 x n_eff
    n_steps_mult = 2
    
    sampling_methods = [
        (ana.pocomc_sampling_woMD, "wo_MD", True),
        (ana.pocomc_sampling_wMD,  "w_MD",  False)
    ]
    
    setup_logging(f"logs/PbPb2760/{idf}/{dir_name}_wo_MD.log")
    save_res(observables, dir_name, sampling_methods, sample_settings,
            n_eff_mult, n_act_mult, n_steps_mult)

    # normalMode: ---------------------->
    n_eff_mult = 200
    n_act_mult = 80  # 0.4 x n_eff
    n_steps_mult = 2
    
    sampling_methods = [
        (ana.pocomc_sampling_woMD, "wo_MD", False),
        (ana.pocomc_sampling_wMD,  "w_MD",  True)
    ]
    
    setup_logging(f"logs/PbPb2760/{idf}/{dir_name}_w_MD.log")
    save_res(observables, dir_name, sampling_methods, sample_settings,
            n_eff_mult, n_act_mult, n_steps_mult)
    
# ======================================================================================

if __name__ == "__main__":
    main()

