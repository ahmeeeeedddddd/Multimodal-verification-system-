"""Quick smoke-tests for the fingerprint feature extraction module.

Run from fingerprint/src/:
    python test_features.py
"""
import sys
import numpy as np
import os; sys.path.insert(0, os.path.dirname(__file__))

from config import IMG_SIZE
from features import extract_fingerprint_features, feature_dim, N_GABOR, N_LBP


def test_output_shape():
    """Feature vector length must match feature_dim()."""
    img  = np.random.default_rng(0).random(IMG_SIZE).astype(np.float32)
    feats = extract_fingerprint_features(img)
    dim   = feature_dim()
    assert feats.shape == (dim,), f"Expected ({dim},), got {feats.shape}"
    print(f"  [PASS] output_shape  (dim={dim})")


def test_output_dtype():
    """Feature vector must be float32."""
    img   = np.zeros(IMG_SIZE, dtype=np.float32)
    feats = extract_fingerprint_features(img)
    assert feats.dtype == np.float32, f"Expected float32, got {feats.dtype}"
    print("  [PASS] output_dtype")


def test_deterministic():
    """Same image must yield identical features."""
    rng  = np.random.default_rng(42)
    img  = rng.random(IMG_SIZE).astype(np.float32)
    f1   = extract_fingerprint_features(img)
    f2   = extract_fingerprint_features(img)
    assert np.allclose(f1, f2), "Feature extraction is not deterministic"
    print("  [PASS] deterministic")


def test_different_images():
    """Different images should (usually) produce different features."""
    rng  = np.random.default_rng(7)
    img1 = rng.random(IMG_SIZE).astype(np.float32)
    img2 = rng.random(IMG_SIZE).astype(np.float32)
    f1   = extract_fingerprint_features(img1)
    f2   = extract_fingerprint_features(img2)
    assert not np.allclose(f1, f2), "Different images gave identical features!"
    print("  [PASS] different_images")


def test_gabor_lbp_subvectors():
    """Gabor and LBP portions must have expected lengths."""
    img   = np.ones(IMG_SIZE, dtype=np.float32) * 0.5
    feats = extract_fingerprint_features(img)
    dim   = feature_dim()
    n_hog = dim - N_GABOR - N_LBP
    print(f"  Feature breakdown: Gabor={N_GABOR}  HOG={n_hog}  LBP={N_LBP}")
    assert dim == N_GABOR + n_hog + N_LBP
    print("  [PASS] gabor_lbp_subvectors")


if __name__ == "__main__":
    print("Testing feature extraction …")
    test_output_shape()
    test_output_dtype()
    test_deterministic()
    test_different_images()
    test_gabor_lbp_subvectors()
    print("All feature tests passed.")

