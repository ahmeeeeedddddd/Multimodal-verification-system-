"""Minimal verification metrics (temporary; replaced by the shared module)."""
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


def compute_eer(y_true, scores):
    """Equal Error Rate and the threshold where FAR == FRR."""
    fpr, tpr, thr = roc_curve(y_true, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def far_frr(y_true, scores, threshold):
    """FAR = impostors accepted, FRR = genuines rejected."""
    y_true = np.asarray(y_true)
    accept = np.asarray(scores) >= threshold
    far = accept[y_true == 0].mean()
    frr = (~accept[y_true == 1]).mean()
    return float(far), float(frr)


def auc(y_true, scores):
    return float(roc_auc_score(y_true, scores))