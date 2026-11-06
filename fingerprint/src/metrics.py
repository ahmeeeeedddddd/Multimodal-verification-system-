"""Fingerprint verification metrics.

This module mirrors voice/src/metrics_basic.py so both modules share
the same metric API surface; it will eventually be superseded by the
shared metrics module built in Task 6.
"""
import numpy as np
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score,
)


def compute_eer(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Equal Error Rate and the threshold where FAR ≈ FRR.

    Returns:
        (eer, threshold)
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    return eer, float(thresholds[idx])


def far_frr(y_true: np.ndarray, scores: np.ndarray,
            threshold: float) -> tuple[float, float]:
    """False Accept Rate and False Reject Rate at a given threshold.

    FAR = fraction of impostors accepted (score ≥ threshold).
    FRR = fraction of genuines rejected  (score < threshold).
    """
    y_true  = np.asarray(y_true)
    accept  = np.asarray(scores) >= threshold
    far     = float(accept[y_true == 0].mean()) if (y_true == 0).any() else 0.0
    frr     = float((~accept[y_true == 1]).mean()) if (y_true == 1).any() else 0.0
    return far, frr


def auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area Under the ROC curve."""
    return float(roc_auc_score(y_true, scores))


def full_report(y_true: np.ndarray, scores: np.ndarray,
                threshold: float, label: str = "") -> dict:
    """Compute all relevant verification metrics.

    Returns a dict with keys: AUC, EER, FAR, FRR, threshold,
    precision, recall, F1, confusion_matrix.
    """
    _eer, _eer_thr = compute_eer(y_true, scores)
    _far, _frr     = far_frr(y_true, scores, threshold)
    preds          = (np.asarray(scores) >= threshold).astype(int)

    report = {
        "label":      label,
        "AUC":        auc(y_true, scores),
        "EER":        _eer,
        "EER_thr":    _eer_thr,
        "FAR":        _far,
        "FRR":        _frr,
        "threshold":  threshold,
        "precision":  float(precision_score(y_true, preds, zero_division=0)),
        "recall":     float(recall_score(y_true, preds, zero_division=0)),
        "F1":         float(f1_score(y_true, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
    }

    if label:
        print(f"\n── {label} ──")
    print(
        f"  AUC={report['AUC']:.4f}  EER={report['EER']:.4f}  "
        f"thr={threshold:.4f}  FAR={report['FAR']:.4f}  FRR={report['FRR']:.4f}\n"
        f"  Precision={report['precision']:.4f}  Recall={report['recall']:.4f}  "
        f"F1={report['F1']:.4f}"
    )
    return report

