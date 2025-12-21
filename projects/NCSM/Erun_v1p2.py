#!/usr/bin/env python3
import os
# Uncomment these only if you face error during sampling (like sampling stuck/error with workers) 
# SIGNIFICANTLY SLOWS DOWN SAMPLING (by ~2x-3x)
os.environ.setdefault("OMP_NUM_THREADS", "1")
# -----------------------------------------------------------------

import numpy as np
import importlib
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
            logging.FileHandler(log_path, mode="a")   # file
        ]
    )
    
# ======================================================================================

def main():

    # v1p2 -- K = ( cabr^2 + (Exp[-(Lam^2 / s_Lam^2)^r_Lam] * Exp[-(L^2 / s_L^2)^r_L])^2 ) * Matern32(Lam; l_Lam) * Matern32(L; l_L)
    version = 'v1p2'
    modname = f"Eanalysis_{version}"
    ana = importlib.import_module(modname)

    # ======================================================================================
    def run(exp, max_Nmax):
        # Load data ========>>>
        
        # Column 0 -> E/R, Column 1 -> Nmax, Column 2 -> hw, Column 3 -> Lambda_eff, Column 4 -> L_eff
        exp_E = np.loadtxt(f'experimental_data/{exp}E_NNLOopt.txt')
        mask = exp_E[:, 1] <= max_Nmax   # Selecting only columns with Nmax <= max_Nmax
        experimental_data = exp_E[mask][:, [0,3,4]]
        
        # Covert Lambda_eff from MeV to fm^{-1}
        HBARC=197.3269804
        experimental_data[:, 1] /= HBARC
        # =======================================
        
        setup_logging(f"logs/{exp}_{version}.log")
        logger = logging.getLogger(__name__)
        logger.info(
            f"{exp} : Data considering till maximum Nmax={max_Nmax}: {experimental_data.shape[0]}. (All Nmax: {exp_E.shape[0]})"
        )
        # =======================================
    
        observables = {
            'Energy': True
        }
        
        data_dir = f'mcmc_data_try/{exp}_{version}/E_Nmax{max_Nmax}'
        
        ana.experimental_data = experimental_data
        ana.observables = observables
        ana.model_param_bounds = model_param_bounds
        ana.HP_bounds = HP_bounds
        ana.init_globals()
        
        samples, md = ana.pocomc_sampling(save_dir=data_dir,
                                          n_eff_mult=n_eff_mult, n_act_mult=n_act_mult, 
                                          n_steps_mult=n_steps_mult, n_total=10000,
                                          save_every_n=100, resume=False, ncores=-1)

        samples, md = ana.pocomc_sampling(save_dir=data_dir,
                                          n_eff_mult=n_eff_mult, n_act_mult=n_act_mult, 
                                          n_steps_mult=n_steps_mult, n_total=20000,
                                          save_every_n=100, resume=True, ncores=-1)
    # ======================================================================================
    
    # # fast ------------------------------->
    # n_eff_mult = 120
    # n_act_mult = 80 
    # n_steps_mult = 1
    
    # beastMode: Default ------------------------------->
    n_eff_mult = 500
    n_act_mult = 150  # 0.3 x n_eff
    n_steps_mult = 2
    # ======================================================================================

    HP_bounds = np.array([
                          [ 0.0   ,  5.0   ],    # for cbar
                          [ 0.001 ,  50.0  ],    # for s_Lambda
                          [ 0.0   ,  15.0  ],    # for r_Lambda
                          [ 0.01  ,  50.0  ],    # for l_Lambda
                          [ 0.001 ,  50.0  ],    # for s_L
                          [ 0.0   ,  15.0  ],    # for r_L
                          [ 0.01  ,  50.0  ]     # for l_L
                          ])
    # ======================================================================================


    # ======================================================================================
    # Bounds: For H2_v1p2. both matern 3/2. Hypermarameter: cbar, s_Lambda, r_Lambda, l_Lambda, s_L, r_L, l_L
    model_param_bounds = np.array([[ -5.0  , 1.0    ],   # Einf
                                  [ 0.0   , 8000.0  ],   # A0
                                  [ 0.01  , 5.0     ],   # A1
                                  [ 0.0   , 1000.0  ],   # A2
                                  [ 0.1   , 2.0     ]    # kinf
                                  ])

    exp = 'H2'     # Nmax: 10, 20, 30, ..., 250
    run(exp=exp, max_Nmax=30)
    run(exp=exp, max_Nmax=70)
    run(exp=exp, max_Nmax=250)
    # ======================================================================================
    
    
    # ======================================================================================
    # Bounds: For H3_v1p2. both matern 3/2. Hypermarameter: cbar, s_Lambda, r_Lambda, l_Lambda, s_L, r_L, l_L
    model_param_bounds = np.array([[ -11.0 , -5.0    ],   # Einf
                                  [ 0.0    , 2000.0  ],   # A0
                                  [ 0.01   , 5.0     ],   # A1
                                  [ 0.0    , 10000.0 ],   # A2
                                  [ 0.1    , 2.0     ]    # kinf
                                  ])

    exp = 'H3'    # Nmax: 4, 6, 8, ..., 40
    run(exp=exp, max_Nmax=14)
    run(exp=exp, max_Nmax=18)
    run(exp=exp, max_Nmax=40)
    # ======================================================================================  


    # ======================================================================================
    # Bounds: For He3_v1p2. both matern 3/2. Hypermarameter: cbar, s_Lambda, r_Lambda, l_Lambda, s_L, r_L, l_L
    model_param_bounds = np.array([[ -9.0   , -5.0   ],   # Einf
                                  [ 0.0    , 2000.0  ],   # A0
                                  [ 0.01   , 5.0     ],   # A1
                                  [ 0.0    , 2000.0  ],   # A2
                                  [ 0.1    , 2.0     ]    # kinf
                                  ])

    exp = 'He3'    # Nmax: 10, 12, 14, ..., 40
    run(exp=exp, max_Nmax=14)
    run(exp=exp, max_Nmax=18)
    run(exp=exp, max_Nmax=40)
    # ======================================================================================
    
    
    # ======================================================================================
    # Bounds: For He4_v1p2. both matern 3/2. Hypermarameter: cbar, s_Lambda, r_Lambda, l_Lambda, s_L, r_L, l_L
    model_param_bounds = np.array([[ -30.0  , -25.0  ],   # Einf
                                  [ 0.0    , 1000.0  ],   # A0
                                  [ 0.01   , 5.0     ],   # A1
                                  [ 5000.0 , 15000.0 ],   # A2
                                  [ 0.1    , 2.0     ]    # kinf
                                  ])

    exp = 'He4'    # Nmax: 4, 6, 8, ..., 20
    run(exp=exp, max_Nmax=10)
    run(exp=exp, max_Nmax=14)
    run(exp=exp, max_Nmax=20)
    # ======================================================================================
  
  
    # ======================================================================================
    # Bounds: For He6_v1p2. both matern 3/2. Hypermarameter: cbar, s_Lambda, r_Lambda, l_Lambda, s_L, r_L, l_L
    model_param_bounds = np.array([[ -31.0 , -18.0   ],   # Einf
                                  [ 0.0    , 1000.0  ],   # A0
                                  [ 0.01   , 5.0     ],   # A1
                                  [ 10000.0, 20000.0 ],   # A2
                                  [ 0.1    , 2.0     ]    # kinf
                                  ])
                         
    exp = 'He6'    # Nmax: 4, 6, 8, ..., 18
    run(exp=exp, max_Nmax=10)
    run(exp=exp, max_Nmax=12)
    run(exp=exp, max_Nmax=18)
    # ====================================================================================== 
    

    # ======================================================================================
    # Bounds: For Li6_v1p2. both matern 3/2. Hypermarameter: cbar, s_Lambda, r_Lambda, l_Lambda, s_L, r_L, l_L
    model_param_bounds = np.array([[ -40.0  , -22.0  ],   # Einf
                                  [ 0.0    , 1000.0  ],   # A0
                                  [ 0.01   , 5.0     ],   # A1
                                  [ 8000.0 , 20000.0 ],   # A2
                                  [ 0.1    , 2.0     ]    # kinf
                                  ])

    exp = 'Li6'    # Nmax: 4, 6, 8, ..., 22
    run(exp=exp, max_Nmax=14)
    run(exp=exp, max_Nmax=18)
    run(exp=exp, max_Nmax=22)
    # ======================================================================================
    
    
    # ======================================================================================
    # Bounds: For He8_v1p2. both matern 3/2. Hypermarameter: cbar, s_Lambda, r_Lambda, l_Lambda, s_L, r_L, l_L
    model_param_bounds = np.array([[ -34.0 , -18.0   ],   # Einf
                                  [ 0.0    , 1000.0  ],   # A0
                                  [ 0.01   , 50.0    ],   # A1
                                  [ 10000.0, 30000.0 ],   # A2
                                  [ 0.1    , 2.0     ]    # kinf
                                  ])
                         
    exp = 'He8'    # Nmax: 4, 6, 8, 10, 12
    run(exp=exp, max_Nmax=8)
    run(exp=exp, max_Nmax=10)
    run(exp=exp, max_Nmax=12)
    # ======================================================================================


# ======================================================================================
    
if __name__ == "__main__":
    main()

    