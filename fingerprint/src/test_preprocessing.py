"""Quick smoke-tests for the fingerprint preprocessing module.

Run from fingerprint/src/:
    python test_preprocessing.py
"""
import sys
import numpy as np

# ---- add src to path so imports work ----
import os; sys.path.insert(0, os.path.dirname(__file__))

from config import IMG_SIZE
from preprocessing import preprocess


def test_preprocess_shape():
    """Output shape must equal IMG_SIZE."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (200, 150), dtype=np.uint8)
    out = preprocess(img)
    assert out.shape == IMG_SIZE, f"Expected {IMG_SIZE}, got {out.shape}"
    print("  [PASS] preprocess_shape")


def test_preprocess_dtype_range():
    """Output must be float32 in [0, 1]."""
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (100, 100), dtype=np.uint8)
    out = preprocess(img)
    assert out.dtype == np.float32, f"Expected float32, got {out.dtype}"
    assert 0.0 <= out.min() and out.max() <= 1.0, (
        f"Range [{out.min():.4f}, {out.max():.4f}] out of [0,1]"
    )
    print("  [PASS] preprocess_dtype_range")


def test_preprocess_deterministic():
    """Same image must produce identical output."""
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, (96, 96), dtype=np.uint8)
    out1 = preprocess(img)
    out2 = preprocess(img)
    assert np.allclose(out1, out2), "preprocess is not deterministic"
    print("  [PASS] preprocess_deterministic")


if __name__ == "__main__":
    print("Testing preprocessing …")
    test_preprocess_shape()
    test_preprocess_dtype_range()
    test_preprocess_deterministic()
    print("All preprocessing tests passed.")

