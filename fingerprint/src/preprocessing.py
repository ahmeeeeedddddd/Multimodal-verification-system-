"""Fingerprint image preprocessing.

Pipeline per image:
  1. Load as grayscale
  2. Resize to IMG_SIZE
  3. CLAHE enhancement  (improves ridge contrast)
  4. Normalize to [0, 1] float32

All functions accept both file paths and numpy arrays.
"""
import cv2
import numpy as np
from pathlib import Path

from config import IMG_SIZE, CLAHE_CLIP, CLAHE_GRID


# ------------------------------------------------------------------
# CLAHE instance (created once and reused)
# ------------------------------------------------------------------
_CLAHE = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)


def load_image(path) -> np.ndarray:
    """Load a fingerprint image as uint8 grayscale array.

    Args:
        path: file path (str or Path).

    Returns:
        2-D uint8 numpy array (H, W).

    Raises:
        FileNotFoundError: if the image cannot be read.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def preprocess(img, size=IMG_SIZE) -> np.ndarray:
    """Full preprocessing pipeline: resize → CLAHE → normalize.

    Args:
        img: uint8 grayscale ndarray *or* a file path.
        size: (height, width) tuple for resizing.

    Returns:
        float32 ndarray of shape (H, W) with values in [0, 1].
    """
    if not isinstance(img, np.ndarray):
        img = load_image(img)

    # Ensure uint8 for OpenCV operations (caller may pass float32 [0,1])
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

    # 1. Resize (bicubic for quality)
    h, w = size
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

    # 2. CLAHE  (operates on uint8)
    img = _CLAHE.apply(img)

    # 3. Normalize to [0, 1] float32
    img = img.astype(np.float32) / 255.0

    return img


def preprocess_file(path, size=IMG_SIZE) -> np.ndarray:
    """Convenience wrapper: load file then preprocess."""
    return preprocess(load_image(path), size)
