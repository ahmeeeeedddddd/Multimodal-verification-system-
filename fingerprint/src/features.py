"""Fingerprint feature extraction (the interface used by fusion and pipeline).

Feature vector layout (fixed length, 1-D float32):
    [ Gabor (N_gabor) | HOG (N_hog) | LBP (LBP_N_BINS) ]

where
    N_gabor = len(GABOR_WAVELENGTHS) × len(GABOR_ORIENTATIONS) × 2
              (mean + std energy per filter)
    N_hog   = depends on IMG_SIZE, HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK
    N_lbp   = LBP_N_BINS

The same function is called during training, single-image verification,
and from the fusion module — guaranteeing identical representations.
"""
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

from config import (
    IMG_SIZE,
    HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK,
    LBP_RADIUS, LBP_N_POINTS, LBP_METHOD, LBP_N_BINS,
    GABOR_WAVELENGTHS, GABOR_ORIENTATIONS,
)


# ------------------------------------------------------------------
# Pre-build Gabor filter bank
# ------------------------------------------------------------------
def _build_gabor_bank() -> list[tuple]:
    """Return a list of (kernel_real, kernel_imag) for each filter."""
    bank = []
    for lam in GABOR_WAVELENGTHS:
        for deg in GABOR_ORIENTATIONS:
            theta = np.deg2rad(deg)
            sigma = lam / np.pi * np.sqrt(np.log(2) / 2) * (2 + 1) / (2 - 1)
            ksize = int(6 * sigma + 1) | 1          # make odd
            kr = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, lam, gamma=0.5, psi=0,
                ktype=cv2.CV_32F,
            )
            ki = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, lam, gamma=0.5,
                psi=np.pi / 2, ktype=cv2.CV_32F,
            )
            bank.append((kr, ki))
    return bank


_GABOR_BANK = _build_gabor_bank()
N_GABOR = len(_GABOR_BANK) * 2       # mean + std per filter
N_LBP   = LBP_N_BINS


def _gabor_features(img: np.ndarray) -> np.ndarray:
    """Apply Gabor bank; return mean+std energy per filter (float32)."""
    # img is float32 [0,1]; convert to uint8 for filter2D
    img8 = (img * 255).astype(np.uint8)
    feats = []
    for kr, ki in _GABOR_BANK:
        resp_r = cv2.filter2D(img8, cv2.CV_32F, kr)
        resp_i = cv2.filter2D(img8, cv2.CV_32F, ki)
        energy = np.sqrt(resp_r ** 2 + resp_i ** 2)
        feats.extend([energy.mean(), energy.std()])
    return np.array(feats, dtype=np.float32)


def _hog_features(img: np.ndarray) -> np.ndarray:
    """Extract HOG descriptor (float32 1-D array)."""
    feats = hog(
        img,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feats.astype(np.float32)


def _lbp_features(img: np.ndarray) -> np.ndarray:
    """Extract normalised LBP histogram (float32 1-D array of length LBP_N_BINS)."""
    img8 = (img * 255).astype(np.uint8)
    lbp = local_binary_pattern(img8, LBP_N_POINTS, LBP_RADIUS, method=LBP_METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins=LBP_N_BINS,
                           range=(0, LBP_N_BINS), density=True)
    return hist.astype(np.float32)


def extract_fingerprint_features(img: np.ndarray) -> np.ndarray:
    """Extract a fixed-length feature vector from a PREPROCESSED fingerprint.

    Args:
        img: float32 2-D array of shape IMG_SIZE, values in [0, 1].
             (output of preprocessing.preprocess).

    Returns:
        1-D float32 array: [Gabor | HOG | LBP].
    """
    gabor = _gabor_features(img)
    hog_f = _hog_features(img)
    lbp   = _lbp_features(img)
    return np.concatenate([gabor, hog_f, lbp]).astype(np.float32)


# ------------------------------------------------------------------
# Utility: compute and report feature dimension
# ------------------------------------------------------------------
def feature_dim() -> int:
    """Return the total feature vector length for the current config."""
    dummy = np.zeros(IMG_SIZE, dtype=np.float32)
    return int(extract_fingerprint_features(dummy).shape[0])


if __name__ == "__main__":
    dim = feature_dim()
    print(f"Fingerprint feature vector length: {dim}")
    print(f"  Gabor  : {N_GABOR}")
    print(f"  HOG    : {dim - N_GABOR - N_LBP}")
    print(f"  LBP    : {N_LBP}")

