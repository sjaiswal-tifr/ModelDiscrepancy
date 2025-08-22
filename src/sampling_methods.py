#!/usr/bin/env python3
'''
Created January 2025

@author: Sunil Jaiswal (sjaiswal.tifr@gmail.com)
'''

import os
import numpy as np
import warnings
import pocomc
import emcee
import zeus
from multiprocess import Pool
from scipy.stats import uniform
import time
import logging
logger = logging.getLogger(__name__)

# ================================ pocoMC sampler ==================================
def pocomc_sampling(min_param, max_param, log_posterior, samples_save_dir = "pocomc",
                    n_effective=2000, n_active=800, n_steps=None,
                    # n_prior=2048, n_max_steps=200, 
                    n_total=5000, n_evidence=5000,
                    save_every_n = 20, resume = False, ncores = -1
                   ):
    """
    This function is based on PocoMC package (version 1.2.1).
    pocoMC is a Preconditioned Monte Carlo (PMC) sampler that uses 
    normalizing flows to precondition the target distribution.

    Parameters:
      - min_param (list or array): Minimum values parameters
      - max_param (list or array): Maximum values parameters
      - log_posterior (callable): Log posterior
      - samples_save_dir (string): Directory name in which the samples will be saved

      - n_effective (int): The effective sample size maintained during the run. Default: 2000.
      - n_active (int): Number of active particles. Default: 800. Must be < n_effective.
      - n_steps (int): Number of MCMC steps after logP plateau. Default: None -> n_steps=n_dim. 
                       Higher values lead to better exploration but increases computational cost.
      - n_total (int): Total effectively independent samples to be collected. Default: 5000.
      - n_evidence (int): Number of importance samples used to estimate evidence. Default: 5000. 
                          If 0, the evidence is not estimated using importance sampling.

      - save_every_n (int): Save sampler state after every 'save_every_n' iterations. 
                            Can be used to resume sampling.
      - resume (Bool): Resume sampling from final state. Default: False. 
                       If True, pocomc_sampler_final.state must exist in 'samples_save_dir'.
      - ncores: Number of cores to use in sampling. 
                  -1 (default): use all availaible cores
                   n (int): use n cores
    """

    if samples_save_dir is not None:
        # Save results
        os.makedirs(samples_save_dir, exist_ok=True)
            
    # Generating prior range for pocomc
    priors = [uniform(lower, upper - lower) for lower, upper in zip(min_param, max_param)]
    prior = pocomc.Prior(priors)
    
    ndim = len(min_param)  # number of parameters in the model

    # -------------------------------------------------------------------------------------------- 
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
    # --------------------------------------------------------------------------------------------
    
    start_time = time.time()

    if samples_save_dir == None:
        sampler = pocomc.Sampler(prior=prior, likelihood=log_posterior,
                                 n_effective=n_effective, n_active=n_active, n_steps=n_steps,
                                 # dynamic=True, n_prior=n_prior, n_max_steps=n_max_steps,
                                 pytorch_threads=None,  # use all availaible threads for preconditioning (really speeds up!)
                                 pool=num_cores
                                 )
    
    else:
        sampler = pocomc.Sampler(prior=prior, likelihood=log_posterior,
                                 n_effective=n_effective, n_active=n_active, n_steps=n_steps,
                                 # dynamic=True, n_prior=n_prior, n_max_steps=n_max_steps,
                                 pytorch_threads=None,  # use all availaible threads for preconditioning (really speeds up!)
                                 pool=num_cores, 
                                 output_dir=f"{samples_save_dir}",  # Directory to save states
                                 output_label="pocomc_sampler"
                                 )

    
    if resume:
        logger.info(f"Resuming sampling using pocomc in {num_cores} cores...")
        
        sampler.run(n_total=n_total, # This is the number of samples we want to draw in total, including the ones we already have.
                    n_evidence=n_evidence,
                    resume_state_path = f"{samples_save_dir}/pocomc_sampler_final.state",
                    save_every = save_every_n, 
                    progress=True)

    else:
        logger.info(f"Starting sampling using pocomc in {num_cores} cores...")

        if samples_save_dir == None:
            sampler.run(n_total=n_total, 
                        n_evidence=n_evidence, 
                        progress=True)
        else:
            sampler.run(n_total=n_total, 
                        n_evidence=n_evidence, 
                        save_every = save_every_n, 
                        progress=True)
    
    # Generating the posterior samples
    # samples, weights, logl, logp = sampler.posterior() # Weighted posterior samples
    samples, _, _ = sampler.posterior(resample=True)  # equal weights for samples 
    
    # Generating the evidence
    # logz, logz_err = sampler.evidence() # Bayesian model evidence estimate and uncertainty
    # print("Log evidence: ", logz)
    # print("Log evidence error: ", logz_err)
    
    # logging.info('Writing pocoMC chains to file...')
    # chain_data = {'chain': samples, 'weights': weights, 'logl': logl,
    #                 'logp': logp, 'logz': logz, 'logz_err': logz_err}

    end_time = time.time()
    logger.info(f"Sampling complete. Time taken: {(end_time - start_time)/60:.2f} minutes.")
    logger.info(f"Samples shape: {samples.shape}")

    if samples_save_dir is not None:
        # Save results
        filename = f"{samples_save_dir}/pocomc_chain_{n_total}.npy"
        np.save(filename, samples)
        logger.info(f"Samples saved in : {filename}\n")
        
    return samples


# ================================ emcee sampler ==================================
def emcee_sampling(min_param, max_param, log_posterior, 
                   nburn=2000, nsteps=2000, num_walker_per_dim=8, 
                   samples_save_dir = "emcee"):
    """
    Function for sampling in parallel using emcee.
    
    Parameters:
      - min_param (list or array): minimum values parameters
      - max_param (list or array): maximum values parameters
      - log_posterior (callable): log posterior
      - nburn (int): "burn-in" period to let chains stabilize
      - nsteps (int): # number of MCMC steps to take after burn-in
      - num_walker_per_dim (int): number of walker per dimension
      - samples_save_dir (string): directory name in which the samples will be saved
    """
    if len(min_param) != len(max_param):
            raise ValueError(
                f"min_param and max_param must have the same length."
            )

    ndim = len(min_param)  # number of parameters in the model
    nwalkers = ndim*num_walker_per_dim  # number of MCMC walkers
    
    start_time = time.time()
    logger.info(f"Starting MCMC sampling using emcee with {nwalkers} walkers in parallel...")

    # Generate starting guesses within the prior bounds
    starting_guesses =  min_param + (max_param - min_param) * np.random.rand(nwalkers,ndim)
    
    # Create a pool of workers using multiprocessing
    with Pool() as pool:
        # Pass the pool to the sampler
        sampler = emcee.EnsembleSampler(nwalkers,ndim,log_posterior,pool=pool)
    
        # "Burn-in" period
        pos, prob, state = sampler.run_mcmc(starting_guesses, nburn, progress=True)
        sampler.reset()
    
        # Sampling period
        pos, prob, state = sampler.run_mcmc(pos, nsteps, progress=True)

    # Collect samples
    samples = sampler.get_chain(flat=True)
    
    end_time = time.time()
    logger.info(f"Sampling complete. Time taken: {(end_time - start_time)/60:.2f} minutes.")
    logger.info(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f} (in total {nwalkers*nsteps} steps)")
    logger.info(f"Samples shape: {samples.shape}")

    if samples_save_dir is not None:
        # Save results
        os.makedirs(samples_save_dir, exist_ok=True)
        filename = f"{samples_save_dir}/emcee_chain.npy"
        np.save(filename, samples)
        logger.info(f"Samples saved in : {filename}\n")
    return samples

    
# ================================ Zeus sampler ==================================
def zeus_sampling(min_param, max_param, log_posterior, 
                   nburn=2000, nsteps=2000, num_walker_per_dim=8, 
                   samples_save_dir = "zeus"):
    """
    Function for sampling in parallel using zeus.
    
    Parameters:
      - min_param (list or array): minimum values parameters
      - max_param (list or array): maximum values parameters
      - log_posterior (callable): log posterior
      - nburn (int): "burn-in" period to let chains stabilize
      - nsteps (int): # number of MCMC steps to take after burn-in
      - num_walker_per_dim (int): number of walker per dimension
      - samples_save_dir (string): directory name in which the samples will be saved
    """
    
    if len(min_param) != len(max_param):
            raise ValueError(
                f"min_param and max_param must have the same length."
            )

    ndim = len(min_param)  # number of parameters in the model
    nwalkers = ndim*num_walker_per_dim  # number of MCMC walkers
    
    start_time = time.time()
    logger.info(f"Starting MCMC sampling using zeus with {nwalkers} walkers in parallel...")

    # Generate starting guesses within the prior bounds
    starting_guesses =  min_param + (max_param - min_param) * np.random.rand(nwalkers,ndim)

    # Create a pool of workers using multiprocessing
    with Pool() as pool:
        # Pass the pool to the sampler
        # Initialise the Ensemble Sampler.
        sampler = zeus.EnsembleSampler(nwalkers, ndim, log_posterior, pool=pool)
        sampler.run_mcmc(starting_guesses, nburn, progress=True)

    # Get the burnin samples
    burnin = sampler.get_chain()

    # Set the new starting positions of walkers based on their last positions
    start = burnin[-1]

    with Pool() as pool:
        sampler = zeus.EnsembleSampler(nwalkers, ndim, log_posterior, moves=zeus.moves.GlobalMove(), pool=pool)
        sampler.run_mcmc(start, nsteps, progress=True)

    # Collect samples
    samples = sampler.get_chain(flat=True)
    
    end_time = time.time()
    logger.info(f"Sampling complete. Time taken: {(end_time - start_time)/60:.2f} minutes.")
    logger.info(f"Samples shape: {samples.shape}")

    if samples_save_dir is not None:
        # Save results
        os.makedirs(samples_save_dir, exist_ok=True)
        filename = f"{samples_save_dir}/zeus_chain.npy"
        np.save(filename, samples)
        logger.info(f"Samples saved in : {filename}\n")
    return samples

# ==================================================================================

if __name__ == "__main__":
    # This block will only run when the script is executed directly.
    # You can add test calls to pocomc_sampling or other methods here.
    pass
