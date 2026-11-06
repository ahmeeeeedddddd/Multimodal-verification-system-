"""Central configuration for the voice module (Task 3 + Task 4)."""
from pathlib import Path

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]       # repo root
DATA_DIR = ROOT_DIR / "data"
FSDD_DIR = DATA_DIR / "fsdd" / "recordings"          # *.wav files live here
SPEECH_CMD_DIR = DATA_DIR / "speech_commands"        # used later in Task 4
MODELS_DIR = ROOT_DIR / "voice" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------
# Audio settings
# ---------------------------------------------------------------
SAMPLE_RATE = 8000        # FSDD is recorded at 8 kHz
DURATION_SEC = 1.0        # every clip is padded/cropped to this length
N_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)
TOP_DB = 25               # silence-trimming threshold (dB below peak)

# ---------------------------------------------------------------
# Feature settings
# ---------------------------------------------------------------
N_MFCC = 13

# ---------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------
RANDOM_STATE = 42