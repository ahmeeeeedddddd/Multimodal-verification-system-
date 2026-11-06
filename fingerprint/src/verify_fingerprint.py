"""Fingerprint verification API.

Usage (from code):
    verifier = FingerprintVerifier("svm")

    # From an image file:
    result = verifier.verify_file("path/to/finger.BMP", claimed_id=42)

    # From a preprocessed numpy image (float32, shape IMG_SIZE):
    result = verifier.verify_image(img_array, claimed_id=42)

    result → {
        "claimed":     42,
        "probability": 0.87,
        "accepted":    True,
        "threshold":   0.65,
    }

    # Feature vector only (for fusion):
    vec = verifier.feature_vector("path/to/finger.BMP")  # 1-D float32
"""
import joblib
import numpy as np

from config import MODELS_DIR
from preprocessing import preprocess_file, preprocess
from features import extract_fingerprint_features


class FingerprintVerifier:
    """Wraps a trained fingerprint model and provides a clean verification API."""

    def __init__(self, model_name: str = "svm",
                 threshold: float | None = None):
        """Load a saved model bundle.

        Args:
            model_name: "svm" or "rf" (must match a saved fingerprint_<name>.joblib).
            threshold:  override the saved threshold (default: use the one
                        tuned on validation data).
        """
        bundle = joblib.load(MODELS_DIR / f"fingerprint_{model_name}.joblib")
        self.model       = bundle["model"]
        self.classes     = [int(c) for c in bundle["classes"]]
        self.threshold   = bundle["threshold"] if threshold is None else threshold
        self.feature_dim = bundle["feature_dim"]

    @property
    def enrolled_ids(self) -> list[int]:
        """Subject IDs known to the model."""
        return list(self.classes)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def feature_vector(self, source) -> np.ndarray:
        """Extract the fixed-length feature vector from an image.

        Args:
            source: file path (str / Path) *or* preprocessed float32 ndarray.

        Returns:
            1-D float32 array ready for the model.
        """
        if isinstance(source, np.ndarray):
            img = preprocess(source)
        else:
            img = preprocess_file(source)
        return extract_fingerprint_features(img)

    # ------------------------------------------------------------------
    # Probability / verification
    # ------------------------------------------------------------------
    def genuine_probability(self, source, claimed_id: int) -> float:
        """P(claimed_id | fingerprint).  Returns 0.0 for unknown IDs.

        Args:
            source:     file path or preprocessed ndarray.
            claimed_id: integer subject ID to verify against.

        Returns:
            Probability in [0, 1].
        """
        if claimed_id not in self.classes:
            return 0.0
        feats = self.feature_vector(source).reshape(1, -1)
        proba = self.model.predict_proba(feats)[0]
        return float(proba[self.classes.index(claimed_id)])

    def verify(self, source, claimed_id: int) -> dict:
        """Full verification decision for a single sample.

        Returns:
            dict with keys: claimed, probability, accepted, threshold.
        """
        p = self.genuine_probability(source, claimed_id)
        return {
            "claimed":     claimed_id,
            "probability": p,
            "accepted":    p >= self.threshold,
            "threshold":   self.threshold,
        }

    def verify_file(self, path, claimed_id: int) -> dict:
        """Convenience wrapper that accepts a file path."""
        return self.verify(path, claimed_id)

    def verify_image(self, img: np.ndarray, claimed_id: int) -> dict:
        """Convenience wrapper that accepts a raw (not yet preprocessed) image."""
        return self.verify(img, claimed_id)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python verify_fingerprint.py <image_path> <claimed_subject_id>")
        sys.exit(1)

    path, claimed = sys.argv[1], int(sys.argv[2])

    for model_name in ("svm", "rf"):
        v = FingerprintVerifier(model_name)
        result = v.verify_file(path, claimed)
        print(f"[{model_name.upper()}] claimed={result['claimed']}  "
              f"prob={result['probability']:.4f}  "
              f"accepted={result['accepted']}  "
              f"(thr={result['threshold']:.4f})")

