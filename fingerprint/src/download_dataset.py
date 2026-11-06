"""Download and cache the SOCOFing dataset via the Kaggle API.

No browser / manual download needed.  The script uses the Kaggle Python
client which reads credentials from:
  1. Environment variables  KAGGLE_USERNAME  and  KAGGLE_KEY
  2. ~/.kaggle/kaggle.json  (the JSON file you download from
     kaggle.com → Account → API → Create New API Token)

Usage (command line):
    python download_dataset.py

Usage (import):
    from download_dataset import ensure_dataset
    real_dir = ensure_dataset()   # Path to the Real/ folder
"""
import os
import zipfile
from pathlib import Path

from config import SOCOFING_DIR, REAL_DIR, KAGGLE_DATASET


def ensure_dataset(force: bool = False) -> Path:
    """Download SOCOFing via Kaggle API if not already present.

    Args:
        force: re-download even if the folder already exists.

    Returns:
        Path to the ``Real/`` subfolder containing the fingerprint images.
    """
    if REAL_DIR.exists() and any(REAL_DIR.iterdir()) and not force:
        print(f"[dataset] SOCOFing already present at {REAL_DIR}")
        return REAL_DIR

    # Lazy import so that the rest of the module works without kaggle installed
    try:
        import kaggle  # noqa: F401 – triggers credential check
    except ImportError:
        raise ImportError(
            "kaggle package not installed. Run: pip install kaggle"
        )
    except SystemExit:
        raise EnvironmentError(
            "Kaggle API credentials not found.\n"
            "Either set KAGGLE_USERNAME and KAGGLE_KEY environment variables,\n"
            "or place kaggle.json in ~/.kaggle/  (download from "
            "kaggle.com → Account → API → Create New API Token)."
        )

    from kaggle.api.kaggle_api import KaggleApi
    api = KaggleApi()
    api.authenticate()

    SOCOFING_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[dataset] Downloading {KAGGLE_DATASET} → {SOCOFING_DIR} …")
    api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(SOCOFING_DIR),
        unzip=True,
        quiet=False,
    )
    print(f"[dataset] Done.  Real images at: {REAL_DIR}")
    return REAL_DIR


def list_real_images() -> list[Path]:
    """Return sorted list of all Real fingerprint image paths."""
    return sorted(REAL_DIR.glob("*.BMP"))


if __name__ == "__main__":
    real_dir = ensure_dataset()
    imgs = list_real_images()
    print(f"Found {len(imgs)} real fingerprint images in {real_dir}")
    if imgs:
        print("First 5:", [p.name for p in imgs[:5]])

