import os, sys, json
import numpy as np
from pathlib import Path
import gzip, dill
import matplotlib.pyplot as plt
import seaborn as sns

aksgp_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(aksgp_dir)

from AKSGP import Emulator as AKSGP

import warnings
warnings.filterwarnings("ignore")

import logging
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
exp_save_dir = Path("../../../experimental_data")

# load bins from file
bin_file = json.loads(Path(f"{exp_save_dir}/PbPb2760_bins.json").read_text())
bins = {k: np.array(item["data"], dtype=item["dtype"]) for k, item in bin_file.items()}
active_obs = list(bins.keys())
counts_per_obs = {k: v.shape[0] for k, v in bins.items()}

# ======================================================================================

# Load training data for Grad and train emulator
train_dir = 'simulation_data/Grad/train'
X = np.loadtxt(os.path.join(train_dir, 'X.txt'))
Ymean = np.loadtxt(os.path.join(train_dir, 'Ymean.txt'))
Ystd = np.loadtxt(os.path.join(train_dir, 'Ystd.txt'))

# ----------------------------------
setup_logging(f"logs/AKSGP_Grad.log")

emu_AKSGP = AKSGP(X, Ymean, Ystd)
emu_AKSGP.fit()

with gzip.open("../Emulator_AKSGP_Grad.dill.gz", "wb") as f:
    dill.dump(emu_AKSGP, f, protocol=dill.HIGHEST_PROTOCOL)

# ======================================================================================

# Load training data for CE and train emulator
train_dir = 'simulation_data/CE/train'
X = np.loadtxt(os.path.join(train_dir, 'X.txt'))
Ymean = np.loadtxt(os.path.join(train_dir, 'Ymean.txt'))
Ystd = np.loadtxt(os.path.join(train_dir, 'Ystd.txt'))

# ----------------------------------
setup_logging(f"logs/AKSGP_CE.log")

emu_AKSGP = AKSGP(X, Ymean, Ystd)
emu_AKSGP.fit()

with gzip.open("../Emulator_AKSGP_CE.dill.gz", "wb") as f:
    dill.dump(emu_AKSGP, f, protocol=dill.HIGHEST_PROTOCOL)
    
# ======================================================================================

