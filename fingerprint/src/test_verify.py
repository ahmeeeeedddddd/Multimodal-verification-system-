"""Smoke-test for FingerprintVerifier with a mock model.

Run from fingerprint/src/:
    python test_verify.py
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import tempfile
import importlib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import IMG_SIZE
from features import feature_dim


def _build_mock_model(classes: list[int], feat_dim: int, seed: int = 0):
    """Train a tiny SVM on random data for testing purposes."""
    rng = np.random.default_rng(seed)
    X   = rng.random((len(classes) * 10, feat_dim)).astype(np.float32)
    y   = np.tile(classes, 10)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    SVC(kernel="rbf", probability=True, random_state=seed)),
    ])
    pipe.fit(X, y)
    return pipe


def _make_bundle_and_verifier(classes, feat_dim_val, threshold, tmpdir, seed=0):
    """Helper: saves a mock bundle and returns a FingerprintVerifier."""
    model = _build_mock_model(classes, feat_dim_val, seed)
    bundle = {
        "model":       model,
        "threshold":   threshold,
        "classes":     classes,
        "feature_dim": feat_dim_val,
    }
    bundle_path = Path(tmpdir) / "fingerprint_svm.joblib"
    joblib.dump(bundle, bundle_path)

    # Patch MODELS_DIR in config, then reload verify module to pick it up
    import config
    config.MODELS_DIR = Path(tmpdir)

    import verify_fingerprint
    importlib.reload(verify_fingerprint)
    return verify_fingerprint.FingerprintVerifier("svm")


def test_verify_accepted():
    """A genuine-looking score above threshold must be accepted."""
    classes      = [1, 2, 3]
    feat_dim_val = feature_dim()

    with tempfile.TemporaryDirectory() as tmpdir:
        v = _make_bundle_and_verifier(classes, feat_dim_val, threshold=0.0, tmpdir=tmpdir)
        img = np.random.default_rng(0).random(IMG_SIZE).astype(np.float32)
        res = v.verify_image(img, claimed_id=1)

        assert "probability" in res
        assert "accepted"    in res
        assert "threshold"   in res
        assert res["accepted"] is True, "Expected accepted=True with threshold=0"
        print(f"  [PASS] verify_accepted  (prob={res['probability']:.4f})")


def test_unknown_id_returns_zero():
    """Claiming an unknown subject ID must return probability 0 and reject."""
    classes      = [10, 20, 30]
    feat_dim_val = feature_dim()

    with tempfile.TemporaryDirectory() as tmpdir:
        v = _make_bundle_and_verifier(classes, feat_dim_val, threshold=0.5, tmpdir=tmpdir, seed=1)
        img = np.random.default_rng(1).random(IMG_SIZE).astype(np.float32)
        res = v.verify_image(img, claimed_id=999)

        assert res["probability"] == 0.0, "Expected 0 prob for unknown ID"
        assert res["accepted"] is False,  "Expected rejection for unknown ID"
        print("  [PASS] unknown_id_returns_zero")


if __name__ == "__main__":
    print("Testing FingerprintVerifier …")
    test_verify_accepted()
    test_unknown_id_returns_zero()
    print("All verifier tests passed.")
