import os
import joblib
import numpy as np

from face_features import preprocess_face, FaceFeatureExtractor
from voice_features import extract_voice_features
from fingerprint_features import preprocess_fingerprint, extract_fingerprint_features
from metrics import compute_metrics, compute_eer


class HierarchicalVerifier:
    """End-to-End Hierarchical Multimodal Verification Pipeline."""

    def __init__(
        self,
        face_bundle_path="saved_models/face_verifier.pkl",
        fusion_bundle_path="saved_models/fusion_verifier.pkl",
        fingerprint_bundle_path="saved_models/fingerprint_verifier.pkl",
        theta_low=0.35,
        theta_high=0.65,
        theta_fusion=None,
        theta_fingerprint=None
    ):
        self.face_bundle = joblib.load(face_bundle_path) if os.path.exists(face_bundle_path) else None
        self.fusion_bundle = joblib.load(fusion_bundle_path) if os.path.exists(fusion_bundle_path) else None
        self.fingerprint_bundle = joblib.load(fingerprint_bundle_path) if os.path.exists(fingerprint_bundle_path) else None

        self.theta_low = float(theta_low)
        self.theta_high = float(theta_high)
        
        # Default stage 2 & 3 thresholds from models or tuned EER
        if theta_fusion is not None:
            self.theta_fusion = float(theta_fusion)
        elif self.fusion_bundle:
            self.theta_fusion = float(self.fusion_bundle.get("threshold", 0.50))
        else:
            self.theta_fusion = 0.50

        if theta_fingerprint is not None:
            self.theta_fingerprint = float(theta_fingerprint)
        elif self.fingerprint_bundle:
            self.theta_fingerprint = float(self.fingerprint_bundle.get("threshold", 0.50))
        else:
            self.theta_fingerprint = 0.50

    @property
    def enrolled_identities(self):
        """List of enrolled identity IDs available in the pipeline."""
        ids = set()
        if self.face_bundle and "templates" in self.face_bundle:
            ids.update(self.face_bundle["templates"].keys())
        if self.fusion_bundle and "templates" in self.fusion_bundle:
            ids.update(self.fusion_bundle["templates"].keys())
        return sorted(list(ids))

    def compute_face_score(self, face_feat, claimed_id):
        """Compute genuine probability for face.

        Accepts either:
          - A raw image array (H x W x 3 or H x W) → features auto-extracted.
            If the image is a tight face crop (small, nearly-square), face
            detection inside the crop is skipped to avoid double-detection.
          - A pre-extracted 1-D feature vector → used directly.
        """
        if self.face_bundle is None or claimed_id not in self.face_bundle["templates"]:
            return 0.0

        # If raw image array is passed, extract features
        if hasattr(face_feat, "ndim") and face_feat.ndim in (2, 3) and face_feat.shape[0] > 1 and face_feat.shape[1] > 1:
            extractor = self.face_bundle["extractor"]
            h, w = face_feat.shape[:2]
            # Already a tight crop: skip inner face detection
            already_cropped = (max(h, w) <= 500) and (0.70 <= w / float(h) <= 1.40)
            u8, norm = preprocess_face(face_feat, already_cropped=already_cropped)
            face_feat = extractor.extract(u8, norm)
        else:
            face_feat = np.asarray(face_feat).ravel()

        template = self.face_bundle["templates"][claimed_id]
        diff = np.abs(face_feat - template).reshape(1, -1)
        prob = self.face_bundle["model"].predict_proba(diff)[0, 1]
        return float(prob)

    def compute_fusion_score(self, face_feat, voice_feat, claimed_id):
        """Compute genuine probability for early-fused face + voice."""
        if self.fusion_bundle is None or claimed_id not in self.fusion_bundle["templates"]:
            return 0.0
        
        # If raw image array is passed
        if hasattr(face_feat, "ndim") and face_feat.ndim in (2, 3) and face_feat.shape[0] > 1 and face_feat.shape[1] > 1:
            extractor = self.face_bundle["extractor"] if self.face_bundle else self.fusion_bundle.get("face_extractor")
            h, w = face_feat.shape[:2]
            already_cropped = (max(h, w) <= 500) and (0.70 <= w / float(h) <= 1.40)
            u8, norm = preprocess_face(face_feat, already_cropped=already_cropped)
            face_feat = extractor.extract(u8, norm)
        else:
            face_feat = np.asarray(face_feat).ravel()

        # If raw audio waveform is passed
        if hasattr(voice_feat, "ndim") and voice_feat.ndim == 1 and len(voice_feat) > 100:
            voice_feat = extract_voice_features(voice_feat)
        else:
            voice_feat = np.asarray(voice_feat).ravel()

        scaler_face = self.fusion_bundle["scaler_face"]
        scaler_voice = self.fusion_bundle["scaler_voice"]
        template = self.fusion_bundle["templates"][claimed_id]

        face_norm = scaler_face.transform(face_feat.reshape(1, -1))
        voice_norm = scaler_voice.transform(voice_feat.reshape(1, -1))
        fused = np.hstack([face_norm, voice_norm])

        diff = np.abs(fused - template).reshape(1, -1)
        prob = self.fusion_bundle["model"].predict_proba(diff)[0, 1]
        return float(prob)

    def compute_voice_score(self, voice_feat, claimed_id):
        """Compute genuine probability for voice-only verification."""
        if self.fusion_bundle is None or claimed_id not in self.fusion_bundle["templates"]:
            return 0.0

        if hasattr(voice_feat, "ndim") and voice_feat.ndim == 1 and len(voice_feat) > 100:
            voice_feat = extract_voice_features(voice_feat)
        else:
            voice_feat = np.asarray(voice_feat).ravel()

        scaler_voice = self.fusion_bundle["scaler_voice"]
        voice_norm = scaler_voice.transform(voice_feat.reshape(1, -1))
        tmpl = np.asarray(self.fusion_bundle["templates"][claimed_id]).ravel()
        voice_tmpl = tmpl[1618:]
        diff = np.abs(voice_norm.ravel() - voice_tmpl)
        dist = np.mean(diff)
        prob = float(np.exp(-dist / 1.5))
        return float(np.clip(prob, 0.0, 1.0))

    def compute_fingerprint_score(self, fingerprint_feat, claimed_id):
        """Compute genuine probability for fingerprint."""
        if self.fingerprint_bundle is None or claimed_id not in self.fingerprint_bundle["templates"]:
            return 0.0

        if hasattr(fingerprint_feat, "ndim") and fingerprint_feat.ndim == 2 and fingerprint_feat.shape[0] > 20:
            u8_fp, norm_fp = preprocess_fingerprint(fingerprint_feat)
            fingerprint_feat = extract_fingerprint_features(u8_fp, norm_fp)
        else:
            fingerprint_feat = np.asarray(fingerprint_feat).ravel()

        template = self.fingerprint_bundle["templates"][claimed_id]
        diff = np.abs(fingerprint_feat - template).reshape(1, -1)
        prob = self.fingerprint_bundle["model"].predict_proba(diff)[0, 1]
        return float(prob)

    def verify(
        self,
        face_input,
        voice_input=None,
        fingerprint_input=None,
        claimed_id=None
    ):
        """Run hierarchical verification on query biometric inputs.

        Args:
            face_input: ndarray (image or pre-extracted feature vector)
            voice_input: ndarray (audio waveform or pre-extracted voice vector, optional unless Stage 2 reached)
            fingerprint_input: ndarray (fingerprint image or feature vector, optional)
            claimed_id: int or str, identity being claimed

        Returns:
            dict containing:
                - verified: bool (accepted or rejected)
                - stage_reached: int (1, 2, or 3)
                - face_score: float
                - voice_score: float (if computed)
                - fusion_score: float (if Stage 2 reached)
                - fingerprint_score: float (if Stage 3 reached)
                - reason: str (explanation of the decision path)
        """
        # Ensure feature extraction if raw face image is provided
        if hasattr(face_input, "ndim") and face_input.ndim in (2, 3) and face_input.shape[0] > 1 and face_input.shape[1] > 1:
            extractor = self.face_bundle["extractor"]
            h, w = face_input.shape[:2]
            already_cropped = (max(h, w) <= 500) and (0.70 <= w / float(h) <= 1.40)
            u8, norm = preprocess_face(face_input, already_cropped=already_cropped)
            face_feat = extractor.extract(u8, norm)
        else:
            face_feat = np.asarray(face_input).ravel()

        # Step 1: Stage 1 Face Evaluation
        face_score = self.compute_face_score(face_feat, claimed_id)

        if face_score >= self.theta_high:
            return {
                "verified": True,
                "stage_reached": 1,
                "face_score": face_score,
                "fusion_score": None,
                "fingerprint_score": None,
                "reason": f"Stage 1 ACCEPT: High confidence face score ({face_score:.3f} >= θ_high {self.theta_high:.3f})"
            }
        elif face_score < self.theta_low:
            return {
                "verified": False,
                "stage_reached": 1,
                "face_score": face_score,
                "fusion_score": None,
                "fingerprint_score": None,
                "reason": f"Stage 1 REJECT: Confident impostor face score ({face_score:.3f} < θ_low {self.theta_low:.3f})"
            }

        # Step 2: Ambiguous Region -> Escalate to Stage 2 Fusion
        if voice_input is None:
            return {
                "verified": False,
                "stage_reached": 1,
                "face_score": face_score,
                "fusion_score": None,
                "fingerprint_score": None,
                "reason": f"UNCERTAIN in Stage 1 ({self.theta_low:.3f} <= score {face_score:.3f} < {self.theta_high:.3f}). Voice input required for Stage 2 Fusion."
            }

        if hasattr(voice_input, "ndim") and voice_input.ndim == 1 and len(voice_input) > 100:
            voice_feat = extract_voice_features(voice_input)
        else:
            voice_feat = np.asarray(voice_input).ravel()

        fusion_score = self.compute_fusion_score(face_feat, voice_feat, claimed_id)

        # Step 3: Optional Fingerprint Modality (Stage 3)
        if fingerprint_input is not None and self.fingerprint_bundle is not None:
            if hasattr(fingerprint_input, "ndim") and fingerprint_input.ndim == 2 and fingerprint_input.shape[0] > 20:
                u8_fp, norm_fp = preprocess_fingerprint(fingerprint_input)
                fp_feat = extract_fingerprint_features(u8_fp, norm_fp)
            else:
                fp_feat = np.asarray(fingerprint_input).ravel()

            fp_score = self.compute_fingerprint_score(fp_feat, claimed_id)
            
            # Tri-modal combined decision
            tri_score = 0.6 * fusion_score + 0.4 * fp_score
            verified = tri_score >= self.theta_fusion
            return {
                "verified": bool(verified),
                "stage_reached": 3,
                "face_score": face_score,
                "fusion_score": fusion_score,
                "fingerprint_score": fp_score,
                "combined_score": tri_score,
                "reason": f"Stage 3 TRI-MODAL {'ACCEPT' if verified else 'REJECT'}: Fusion ({fusion_score:.3f}) + Fingerprint ({fp_score:.3f}) -> Combined {tri_score:.3f}"
            }

        # Stage 2 Fusion Decision
        verified = fusion_score >= self.theta_fusion
        return {
            "verified": bool(verified),
            "stage_reached": 2,
            "face_score": face_score,
            "fusion_score": fusion_score,
            "fingerprint_score": None,
            "reason": f"Stage 2 FUSION {'ACCEPT' if verified else 'REJECT'}: Score ({fusion_score:.3f} {' >=' if verified else ' <'} θ_fusion {self.theta_fusion:.3f})"
        }


def tune_hierarchical_thresholds(
    y_true,
    face_scores,
    fusion_scores,
    theta_fusion=0.5,
    target_stage2_max=0.35,
    min_stage2_escalation=0.10
):
    """Tune upper and lower thresholds (theta_low, theta_high) on validation trials."""
    y_true = np.asarray(y_true).astype(int)
    face_scores = np.asarray(face_scores).astype(float)
    fusion_scores = np.asarray(fusion_scores).astype(float)

    best_score = -1.0
    best_pair = (0.35, 0.65)
    best_res = None

    # Search grid
    low_candidates = np.linspace(0.15, 0.48, 34)
    high_candidates = np.linspace(0.52, 0.85, 34)

    for low in low_candidates:
        for high in high_candidates:
            if low >= high:
                continue

            # Stage 1 determinations
            preds = np.zeros_like(y_true)
            stage2_mask = (face_scores >= low) & (face_scores < high)
            accept_stage1_mask = (face_scores >= high)
            reject_stage1_mask = (face_scores < low)

            preds[accept_stage1_mask] = 1
            preds[reject_stage1_mask] = 0
            preds[stage2_mask] = (fusion_scores[stage2_mask] >= theta_fusion).astype(int)

            stage2_rate = float(np.mean(stage2_mask))

            m = compute_metrics(y_true, preds)
            acc = m["accuracy"]

            # Penalize if outside desired Stage 2 escalation range
            if stage2_rate > target_stage2_max:
                penalty = (stage2_rate - target_stage2_max) * 0.5
            elif stage2_rate < min_stage2_escalation:
                penalty = (min_stage2_escalation - stage2_rate) * 0.2
            else:
                penalty = 0.0

            objective = acc - penalty

            if objective > best_score:
                best_score = objective
                best_pair = (float(low), float(high))
                best_res = {
                    "accuracy": acc,
                    "far": m["far"],
                    "frr": m["frr"],
                    "stage2_fraction": stage2_rate,
                    "f1": m["f1"]
                }

    return best_pair[0], best_pair[1], best_res
