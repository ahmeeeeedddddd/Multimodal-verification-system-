import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

def compute_metrics(y_true, y_scores, threshold=0.5):
    y_pred = (y_scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    acc = accuracy_score(y_true, y_pred)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    return {
        "accuracy": acc,
        "far": far,
        "frr": frr,
        "cm": cm,
        "auc": roc_auc
    }

def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    optimal_threshold = thresholds[idx]
    return eer, optimal_threshold
