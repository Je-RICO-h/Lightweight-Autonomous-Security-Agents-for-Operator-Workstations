"""
Computes the policy-agent decision thresholds (tau_warn, tau_critical) from a
trained model's own ROC curve on its held-out test set. See docs/agent_design.md
section 3.

tau_critical is the ROC operating point at the Equal Error Rate (EER) — the
standard biometric-systems threshold, where false-accept rate equals
false-reject rate. tau_warn is a looser threshold (default: the point where
the false-accept rate first exceeds 2x the EER's false-accept rate), giving a
"suspicious but not conclusive" band below tau_critical.
"""
import numpy as np
from sklearn.metrics import auc, roc_curve


def compute_thresholds(y_test_impostor: np.ndarray, y_proba_impostor: np.ndarray,
                        warn_margin_multiplier: float = 2.0) -> dict:
    """
    y_test_impostor: 1 if the true label is "Other" (impostor), 0 if "User".
    y_proba_impostor: predicted probability of "Other" (impostor) for each row.
    """
    fpr, tpr, roc_thresholds = roc_curve(y_test_impostor, y_proba_impostor)
    fnr = 1 - tpr

    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    tau_critical = float(roc_thresholds[eer_idx])
    eer_far = float(fpr[eer_idx])
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)

    warn_candidates = np.where(fpr > warn_margin_multiplier * eer_far)[0]
    if len(warn_candidates) > 0:
        warn_idx = int(warn_candidates[0])
        tau_warn = float(roc_thresholds[warn_idx])
    else:
        tau_warn = tau_critical * 0.85

    return {
        "tau_warn": tau_warn,
        "tau_critical": tau_critical,
        "eer": eer,
        "eer_far": eer_far,
        "auc": float(auc(fpr, tpr)),
    }
