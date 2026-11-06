"""Voice verification API.

Usage:
    verifier = VoiceVerifier("svm")
    result = verifier.verify(audio, claimed_id="george")
    result -> {"claimed": ..., "probability": ..., "accepted": ..., "threshold": ...}
"""
import joblib
import numpy as np

from config import MODELS_DIR, SAMPLE_RATE
from features import extract_voice_features
from preprocessing import preprocess, load_audio


class VoiceVerifier:
    def __init__(self, model_name: str = "svm", threshold: float | None = None):
        bundle = joblib.load(MODELS_DIR / f"voice_{model_name}.joblib")
        self.model = bundle["model"]
        self.classes = list(bundle["classes"])
        # default threshold = the one tuned on validation data
        self.threshold = bundle["threshold"] if threshold is None else threshold

    @property
    def enrolled_ids(self):
        return list(self.classes)

    def features(self, raw_audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Raw waveform -> feature vector (same code as training)."""
        return extract_voice_features(preprocess(raw_audio, sr), sr)

    def genuine_probability(self, raw_audio, claimed_id: str,
                            sr: int = SAMPLE_RATE) -> float:
        """P(claimed identity | audio). Returns 0.0 for unknown IDs."""
        if claimed_id not in self.classes:
            return 0.0
        feats = self.features(raw_audio, sr).reshape(1, -1)
        proba = self.model.predict_proba(feats)[0]
        return float(proba[self.classes.index(claimed_id)])

    def verify(self, raw_audio, claimed_id: str, sr: int = SAMPLE_RATE) -> dict:
        p = self.genuine_probability(raw_audio, claimed_id, sr)
        return {"claimed": claimed_id, "probability": p,
                "accepted": p >= self.threshold, "threshold": self.threshold}

    def verify_file(self, path, claimed_id: str) -> dict:
        return self.verify(load_audio(path), claimed_id)