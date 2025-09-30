# ModelDiscrepancy
A Bayesian framework for model-data comparison that accounts for theoretical uncertainties

Accurate comparisons between theoretical models and experimental data are critical for scientific progress. However, inferred physical model parameters can vary significantly with the chosen physics model, highlighting the importance of properly accounting for theoretical uncertainties. We present a Bayesian framework that explicitly quantifies these uncertainties by statistically modeling theory errors, guided by qualitative knowledge of a theory's varying reliability across the input domain. 

For references, please see the following paper:
- Sunil Jaiswal, Chun Shen, Richard J. Furnstahl, Ulrich Heinz, Matthew T. Pratola, "Bayesian model-data comparison incorporating theoretical uncertainties", [https://arxiv.org/abs/2504.13144]


## Overview

The repository is organized as follows:

```text
.
├── src/
│   ├── ModelDiscrepancy.py
│   ├── sampling_methods.py
│   ├── get_quantiles.py
│   └── plot_scripts.py
├── projects/
│   └── [arXiv_identifier]/
└── environment.yml

```

**Directory and File Descriptions**

- `src/` : Contains the core Python source code for Bayesian inference methods.
  - `ModelDiscrepancy.py` : A class for computing the likelihood and posterior distributions of Bayesian models.
  - `sampling_methods.py` : A collection of functions for MCMC sampling.
  - `get_quantiles.py` : Functions to calculate quantiles from MCMC chains for various observables.
  - `plot_scripts.py` : Helper functions for generating informative plots, such as corner plots, from MCMC results.
- `projects/` : A directory for specific analysis projects. Subdirectory names are based on related arXiv paper identifiers.
- `environment.yml` : Specifies all the necessary dependencies for setting up the project's development environment using Conda.


**Dependencies**

To set up the required environment, use the provided `environment.yml` file with Conda.

```sh
conda env create -f environment.yml -n myenv
conda activate myenv
```

## Cite this work

Please use the following BibTeX entry to cite this work:

```bibtex
@article{Jaiswal:2025hyp,
    author = "Jaiswal, Sunil and Shen, Chun and Furnstahl, Richard J. and Heinz, Ulrich and Pratola, Matthew T.",
    title = "{Bayesian model-data comparison incorporating theoretical uncertainties}",
    eprint = "2504.13144",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "4",
    year = "2025"
}
```

## Contact details

Sunil Jaiswal (<jaiswal.61@osu.edu>)  
Department of Physics  
The Ohio State University  
Columbus, Ohio 43210, USA 
