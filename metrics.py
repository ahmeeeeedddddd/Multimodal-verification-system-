"""Shared Biometric Verification Evaluation Metrics Module.

Provides standard evaluation functions required across all modalities and fusion stages:
- Confusion Matrix
- False Acceptance Rate (FAR) and False Rejection Rate (FRR)
- Equal Error Rate (EER) and optimal threshold selection
- Receiver Operating Characteristic (ROC) curve and Area Under Curve (AUC)
- Detection Error Tradeoff (DET) curve (FAR vs FRR)
- Precision, Recall, and F1-score
- Visualization utilities for ROC, DET, and Confusion Matrix plots
"""

import numpy as np
from scipy.stats import norm
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt


def compute_metrics(y_true, y_scores_or_preds, threshold=0.5):
    """Compute comprehensive verification metrics.

    Args:
        y_true: 1D array-like of ground truth binary labels (1=genuine, 0=impostor).
        y_scores_or_preds: 1D array-like of genuine probabilities or binary predictions.
        threshold: float, decision threshold (if y_scores_or_preds are continuous probabilities).

    Returns:
        dict containing:
            accuracy, far, frr, precision, recall, f1, cm, auc (if scores provided),
            tn, fp, fn, tp.
    """
    y_true = np.asarray(y_true).astype(int)
    y_arr = np.asarray(y_scores_or_preds)

    # Determine if input is already binary or continuous probabilities
    is_binary = np.all(np.isin(y_arr, [0, 1])) and not np.issubdtype(y_arr.dtype, np.floating)
    if is_binary:
        y_pred = y_arr.astype(int)
        roc_auc = None
    else:
        y_pred = (y_arr >= threshold).astype(int)
        try:
            fpr, tpr, _ = roc_curve(y_true, y_arr)
            roc_auc = float(auc(fpr, tpr))
        except Exception:
            roc_auc = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    frr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "accuracy": acc,
        "far": far,
        "frr": frr,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "cm": cm,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "auc": roc_auc,
    }


def compute_eer(y_true, y_scores):
    """Compute Equal Error Rate (EER) and the optimal decision threshold where FAR ~= FRR."""
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    optimal_threshold = float(thresholds[idx])

    return eer, optimal_threshold


def compute_det_curve(y_true, y_scores):
    """Compute False Acceptance Rate (FAR) and False Rejection Rate (FRR) points for DET curve.

    Returns:
        far: np.ndarray
        frr: np.ndarray
        thresholds: np.ndarray
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    far = fpr
    frr = 1.0 - tpr
    return far, frr, thresholds


def plot_roc_curve(y_true, y_scores, title="ROC Curve", save_path=None):
    """Plot Receiver Operating Characteristic (ROC) curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    eer, eer_thr = compute_eer(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Chance")
    ax.scatter([eer], [1 - eer], color="red", zorder=5, label=f"EER = {eer:.4f}")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Accept Rate (FAR / FPR)")
    ax.set_ylabel("True Accept Rate (TAR / TPR)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_det_curve(curves_dict, title="Detection Error Tradeoff (DET) Curve", save_path=None):
    """Plot Detection Error Tradeoff (DET) curve (FAR vs FRR in normal deviance / probit coordinates).

    Args:
        curves_dict: dict of {label: (y_true, y_scores)}
        title: str, chart title
        save_path: optional str path to save plot
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    for label, (y_true, y_scores) in curves_dict.items():
        far, frr, _ = compute_det_curve(y_true, y_scores)
        # Avoid infinity in normal inverse CDF
        eps = 1e-4
        far_clipped = np.clip(far, eps, 1.0 - eps)
        frr_clipped = np.clip(frr, eps, 1.0 - eps)

        far_dev = norm.ppf(far_clipped)
        frr_dev = norm.ppf(frr_clipped)

        ax.plot(far_dev, frr_dev, lw=2, label=label)

    # Set standard tick marks for probit scale
    ticks = np.array([0.001, 0.01, 0.05, 0.20, 0.50, 0.80, 0.95])
    tick_labels = [f"{t*100:.1f}%" for t in ticks]
    tick_vals = norm.ppf(ticks)

    ax.set_xticks(tick_vals)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_vals)
    ax.set_yticklabels(tick_labels)

    ax.set_xlabel("False Accept Rate (FAR)")
    ax.set_ylabel("False Reject Rate (FRR)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig
