#!/usr/bin/env python3

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sampling_methods import pocomc_sampling, emcee_sampling, zeus_sampling
import numpy as np

# =========================================

mu, stdev = 0.5, 0.1
low, high = -2.0, 2.0
var = stdev**2
log_c = -0.5 * np.log(2 * np.pi * var)
inv_var = 1.0 / var

def log_post(x):
    if not (low < x < high):
        return -np.inf
    d = x - mu
    return log_c - 0.5 * inv_var * (d * d)
    
# =========================================

min_param, max_param = np.atleast_1d(low), np.atleast_1d(high)

def pocomc_res():
    samples = pocomc_sampling(min_param, max_param, log_posterior=log_post, 
                              n_effective=500, n_active=150, n_steps=1, n_total=1000, n_evidence=1000, 
                              save_every_n = 100, samples_save_dir = None, resume=False)   
    
    return np.mean(samples[:,0]), np.std(samples[:,0])

def emcee_res():
    samples = emcee_sampling(min_param, max_param, log_posterior=log_post, 
                             nburn=100, nsteps=100, num_walker_per_dim=10, 
                             samples_save_dir = None)
    
    return np.mean(samples[:,0]), np.std(samples[:,0])

    
def zeus_res():
    samples = zeus_sampling(min_param, max_param, log_posterior=log_post, 
                            nburn=100, nsteps=100, num_walker_per_dim=10, 
                            samples_save_dir = None)
    
    return np.mean(samples[:,0]), np.std(samples[:,0])

# =========================================

def main():
    line1 = "-" * 90
    line2 = "=" * 35
    print(f'\n{line2}\nSampling methods: Starting tests\n{line2}')
    # =========================================
    
    failures = 0
    tol_mean, tol_std = 0.05, 0.05  # tolerances
    
    def check(name, mean, std):
        nonlocal failures
        if not np.isfinite(mean) or not np.isfinite(std):
            print(f"{line1}\nFAIL {name}: non-finite stats\n{line1}\n")
            failures += 1
            return
        ok_mean = abs(mean - mu) <= tol_mean
        ok_std  = abs(std  - stdev) <= tol_std and std > 0
        if ok_mean and ok_std:
            print(f"{line1}\nPASS: {name} sampling passed. mean={mean:.3f} (target {mu:.3f}), std={std:.3f} (target {stdev:.3f})\n{line1}\n")
        else:
            print(f"{line1}\nFAIL: {name} sampling failed. mean={mean:.3f} (target {mu:.3f}), std={std:.3f} (target {stdev:.3f})\n{line1}\n")
            failures += 1
    # =========================================

    # Run samplers (count errors as failures)
    try:
        m, s = pocomc_res()
        check("pocomc", m, s)
    except Exception as e:
        print(f"{line1}\nFAIL pocomc: {type(e).__name__}: {e}\n{line1}\n")
        failures += 1

    try:
        m, s = emcee_res()
        check("emcee", m, s)
    except Exception as e:
        print(f"{line1}\nFAIL emcee: {type(e).__name__}: {e}\n{line1}\n")
        failures += 1
        
    try:
        m, s = zeus_res()
        check("zeus", m, s)
    except Exception as e:
        print(f"{line1}\nFAIL zeus: {type(e).__name__}: {e}\n{line1}\n")
        failures += 1

    if failures == 0:
        print(f'\n{line2}\nSampling methods: All test PASSED\n{line2}\n')
    else:
        print(f'\n{line2}\nSampling methods: {failures} test FAILED\n{line2}\n')
        
    return failures
    
# =========================================

if __name__ == "__main__":
    sys.exit(main())
    