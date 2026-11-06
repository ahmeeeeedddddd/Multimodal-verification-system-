"""Central configuration for the fingerprint module (Task 2).

Dataset  : Sokoto Coventry Fingerprint Dataset (SOCOFing)
Kaggle   : https://www.kaggle.com/datasets/ruizgara/socofing
Access   : via Kaggle API (KAGGLE_USERNAME + KAGGLE_KEY env vars,
           or ~/.kaggle/kaggle.json)
"""
from pathlib import Path

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
ROOT_DIR   = Path(__file__).resolve().parents[2]      # repo root
DATA_DIR   = ROOT_DIR / "data"
SOCOFING_DIR = DATA_DIR / "socofing"                  # unzipped dataset root
# SOCOFing real images live under: SOCOFING_DIR/SOCOFing/Real/
REAL_DIR   = SOCOFING_DIR / "SOCOFing" / "Real"
MODELS_DIR = ROOT_DIR / "fingerprint" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Kaggle dataset slug
KAGGLE_DATASET = "ruizgara/socofing"

# ---------------------------------------------------------------
# Image settings
# ---------------------------------------------------------------
IMG_SIZE   = (96, 96)       # resize all fingerprints to this (H, W)
CLAHE_CLIP = 2.0            # CLAHE clipLimit
CLAHE_GRID = (8, 8)         # CLAHE tileGridSize

# ---------------------------------------------------------------
# Feature settings
# ---------------------------------------------------------------
# HOG
HOG_ORIENTATIONS  = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

# LBP
LBP_RADIUS    = 3
LBP_N_POINTS  = 8 * LBP_RADIUS     # 24
LBP_METHOD    = "uniform"
LBP_N_BINS    = LBP_N_POINTS + 2   # uniform LBP produces P+2 bins

# Gabor bank  (wavelengths × orientations)
GABOR_WAVELENGTHS  = [4, 8, 16]    # pixels
GABOR_ORIENTATIONS = [0, 45, 90, 135]  # degrees

# ---------------------------------------------------------------
# Dataset split
# ---------------------------------------------------------------
N_ENROLLED  = 50     # subjects used for training / verification
N_UNKNOWN   = 10     # subjects used as unknown impostors (never trained on)
FINGERS_PER_SUBJECT = 5   # SOCOFing has 5 fingers per subject
VAL_SIZE    = 0.20
TEST_SIZE   = 0.20

# ---------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------
RANDOM_STATE = 42

