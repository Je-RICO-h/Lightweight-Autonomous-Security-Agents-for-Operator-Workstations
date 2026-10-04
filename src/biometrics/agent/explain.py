"""
Explainable step: why did the model think this window was not the enrolled
user? Runs SHAP locally on the owner's machine, so the feature row never
leaves it -- only the top few reasons are reported to the server.

Each reason says which feature pushed the decision toward "Other", this
window's value, and the enrolled user's typical value (from
feature_profile.json, written at train time).
"""
import json
import logging

import joblib
import numpy as np
import xgboost as xgb

from biometrics import paths
from biometrics.training.train_xgboost import IMPOSTOR_LABEL, is_one_hot

log = logging.getLogger(__name__)

# Short phrases for the on-screen popup. The server's Help page has the full glossary.
FEATURE_LABELS = {
    "hold_time": "key hold time",
    "seek_time": "gap between releasing one key and pressing the next",
    "interval": "time between key presses",
    "time_diff": "time between key presses",
    "error": "share of corrected keystrokes",
    "error_rate": "error rate",
    "accuracy": "typing accuracy",
    "key_per_second": "typing speed (keys/sec)",
    "wpm": "typing speed (words/min)",
    "mean_ngram_dwell_time": "average hold time over recent keys",
    "std_ngram_dwell_time": "consistency of hold time",
    "mean_ngram_flight_time": "average gap over recent keys",
    "std_ngram_flight_time": "consistency of gaps between keys",
    "rolling_mean_kps": "recent typing speed",
    "rolling_std_kps": "recent typing-speed variability",
    "rolling_mean_wpm": "recent words/min",
    "rolling_std_wpm": "recent words/min variability",
    "variation_in_typing_speed": "per-key speed variation",
    "is_burst": "share of rapid-burst keystrokes",
    "burst_id": "burst counter (session position)",
    "start_time": "time into the window (session position)",
    "end_time": "time into the window (session position)",
}


def feature_label(col: str) -> str:
    if col in FEATURE_LABELS:
        return FEATURE_LABELS[col]
    if col.startswith("key_") and is_one_hot(col):
        return f"how often key {col[len('key_'):]} is used"
    if col.startswith("combination_"):
        return f"how often combination {col[len('combination_'):]} is used"
    return col.replace("_", " ")


class Explainer:
    def __init__(self):
        import shap

        # Pickle-based loads are safe here: these are this machine's own
        # locally trained artifacts (xgboost_inference.py loads them the same
        # way). Nothing here is ever loaded from the server or the network.
        model = xgb.XGBClassifier()
        model.load_model(paths.MODEL_FILE)
        classes = list(joblib.load(paths.LABEL_ENCODER_FILE).classes_)
        self.columns = [str(c) for c in np.load(paths.OHE_COLS_FILE, allow_pickle=True)]
        self._shap = shap.TreeExplainer(model)
        # SHAP explains the model's positive class (index 1). Flip the sign when
        # that's the enrolled user, so "impact > 0" always means "looks less like the user".
        self._toward_impostor = 1.0 if classes.index(IMPOSTOR_LABEL) == 1 else -1.0

        if paths.FEATURE_PROFILE_FILE.exists():
            self.profile = json.loads(paths.FEATURE_PROFILE_FILE.read_text())
        else:
            log.warning("%s missing -- reasons will not show the user's typical values "
                        "(retrain, or run `biometrics describe-model`)", paths.FEATURE_PROFILE_FILE)
            self.profile = {}

    def reasons(self, feature_row: dict, k: int = 5) -> list[dict]:
        """Top-k timing/statistical features that pushed this window toward "Other",
        strongest first. Key and key-combination one-hot features are left out."""
        row = np.array([[float(feature_row.get(col, 0.0)) for col in self.columns]])
        values = self._shap.shap_values(row)
        if isinstance(values, list):
            values = values[-1]
        impact = self._toward_impostor * np.asarray(values)[0]

        reasons = []
        for i in np.argsort(-impact):
            if impact[i] <= 0 or len(reasons) == k:
                break
            col = self.columns[i]
            if is_one_hot(col):
                continue  # which keys were typed never leaves the machine (see reporter.py)
            value = float(row[0, i])
            typical = self.profile.get(col, {})
            reasons.append({
                "feature": col,
                "label": feature_label(col),
                "value": value,
                "typical": typical.get("typical"),
                "typical_low": typical.get("low"),
                "typical_high": typical.get("high"),
                "direction": _direction(value, typical),
                "impact": float(impact[i]),
            })
        return reasons


def _direction(value: float, typical: dict) -> str | None:
    if not typical:
        return None
    low = typical.get("low", typical["typical"])
    high = typical.get("high", typical["typical"])
    if value > high:
        return "higher"
    if value < low:
        return "lower"
    return "within range"
