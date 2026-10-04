"""
Plain-language meaning of every value the dashboard shows. Used for the "?"
tooltips and the Help page, so there is one place to read or change them.
"""
import math

# What each level does is set on the monitored machine in
# src/biometrics/agent/actions.py; these texts describe the shipped defaults.
LEVELS = {
    "allow": "Typing matches the enrolled user. Nothing happens and nothing is reported individually "
             "(only counted in the host's status).",
    "challenge": "Suspicious: the impostor score was at or above τ warn for 2 windows in a row. "
                 "By default the user is asked to re-authenticate; unanswered for 30 s, it escalates to an alert.",
    "alert": "High-confidence mismatch: the impostor score was at or above τ critical for 3 windows in a row, "
             "or a challenge was ignored. By default an informational popup; closing it without "
             "re-authenticating arms the host.",
    "critical": "The host was armed (an alert was dismissed without re-authenticating) and the typing "
                "is still suspicious (≥ τ warn). By default a window that only closes after re-authentication "
                "covers the screen, and the enforcement hook is called in dry-run mode (logged, no action).",
}

TRIGGERS = {
    "sustained_warn": "Impostor score ≥ τ warn for 2 consecutive windows.",
    "sustained_critical": "Impostor score ≥ τ critical for 3 consecutive windows.",
    "challenge_ignored": "The previous challenge was closed or timed out without re-authentication.",
    "after_dismissed_alert": "An earlier alert was dismissed without re-authentication, and a new window "
                             "crossed τ warn.",
    "supervisor_lockout": "Locked from this dashboard with the Lock button.",
}

OUTCOMES = {
    "reauthenticated": "The user passed the operating-system password prompt, so the model was wrong. Detection "
                       "resets. For a challenge or an alert, the keystrokes of that stretch are flagged for "
                       "review, and 'Refine model' learns them. After a critical, nothing is kept.",
    "closed": "The window was closed without re-authenticating.",
    "timed_out": "Nobody answered the challenge within 30 s.",
    "superseded": "A higher level fired while this one was still on screen, and replaced it.",
    None: "No answer yet: the popup is still on screen, or the host went offline first.",
}

TERMS = {
    "impostor_score": ("Impostor score",
                       "The model's probability (0–100%) that the person typing is NOT the enrolled user. "
                       "It is compared with τ warn and τ critical to pick the level."),
    "confidence": ("Model confidence",
                   "How sure the model is of the label it chose (User or Other). The impostor score is this "
                   "value when the label is Other, and 1 − this value when it is User."),
    "window": ("Window",
               "One prediction: a fixed number of keystrokes (15 by default), scored together."),
    "tau_warn": ("τ warn",
                 "The lower decision threshold, computed from the model's ROC curve at training time: the score "
                 "at which the enrolled user's own typing gets falsely flagged twice as often as at τ critical. "
                 "More sensitive, so it's used for the softer 'challenge' level."),
    "tau_critical": ("τ critical",
                     "The upper decision threshold: the ROC point at the Equal Error Rate, where false accepts "
                     "and false rejects are equally likely. Scores above it are a high-confidence mismatch."),
    "eer": ("EER (equal error rate)",
            "The error rate at the point where the false-accept rate equals the false-reject rate. "
            "Lower is better. This is the standard single-number quality measure for biometric systems."),
    "eer_far": ("False-alarm rate at τ critical",
                "Share of the enrolled user's own test keystrokes that score at or above τ critical, i.e. would "
                "be wrongly flagged."),
    "auc": ("AUC",
            "Area under the ROC curve: the probability that a random impostor window scores higher than a random "
            "genuine one. 1.0 is perfect separation, 0.5 is guessing."),
    "accuracy": ("Accuracy", "Share of held-out test keystrokes labelled correctly (User or Other)."),
    "precision": ("Precision", "Of everything the model labelled as this class, the share that really was."),
    "recall": ("Recall", "Of everything that really was this class, the share the model caught."),
    "f1": ("F1 score", "The harmonic mean of precision and recall."),
    "support": ("Support", "Number of test keystrokes of this class."),
    "test_rows": ("Test set",
                  "Keystrokes held out from training and split by recording session, so no session appears in "
                  "both sets. All metrics and thresholds are measured here."),
    "value": ("This window", "The feature's value, averaged over the keystrokes of the window that triggered."),
    "typical": ("Usually",
                "The enrolled user's typical value, taken from their own training data: the median, with the "
                "middle 50% (25th–75th percentile) as the range. Key and combination features show how often "
                "the user presses that key."),
    "direction": ("Direction", "Whether this window's value was above, below, or inside the user's usual range."),
    "impact": ("Impact",
               "How much this feature pushed the model toward 'not the user' (SHAP value, in log-odds). "
               "Features are listed strongest first. A feature can have impact even when it is within the "
               "usual range, because the model weighs features in combination."),
    "recent_scores": ("Recent scores",
                      "The impostor scores of the last 30 windows (small dots), with the latest one as the large "
                      "marker. Dots left of τ warn are windows that clearly matched the user."),
    "live_features": ("Live typing statistics",
                      "The latest window's value of each timing feature (marker), drawn over the user's usual range "
                      "from their own training data (shaded band = middle 50%, line = median). Key and key-combination "
                      "frequencies are never sent to this server, because they would reveal what is being typed."),
    "pending_review": ("Flagged for review",
                       "Keystroke stretches the model flagged, after which the owner re-authenticated at a "
                       "challenge or an alert. They are stored only on the owner's machine until 'Refine model' "
                       "learns them as the owner's typing."),
    "refine": ("Refine model",
               "Retrains the model on the owner's machine: the sessions flagged for review are added to the training "
               "data as the main user, the dataset is rebuilt, and the model, τ thresholds and statistics are "
               "recomputed. The agent switches to the new model without restarting (a few seconds)."),
    "lock": ("Lock",
             "Immediately shows the lock window on that machine (a critical incident). Only a successful "
             "re-authentication closes it. Delivered with the agent's next report (after its next scored window, or within 5 s when idle)."),
    "armed": ("Armed",
              "An alert was dismissed without re-authenticating. The next suspicious window goes straight "
              "to critical. Only a successful re-authentication disarms the host."),
    "online": ("Online",
               "The agent reported within the last 20 s. It reports after every scored window and every event, "
               "and sends a heartbeat every 5 s when idle."),
    "model_hash": ("Model hash",
                   "The first 12 characters of the SHA-256 of the host's model file. A new hash means the model "
                   "was retrained."),
    "data_boundary": ("What the server receives",
                      "Only decisions: level, score, the top reasons, and the user's response. Keystrokes, "
                      "feature rows and the model itself never leave the owner's machine."),
}

# unit: "s" (shown in ms), "share" (shown as %), or "" (plain number)
FEATURES = {
    "hold_time": ("Key hold time", "s", "How long each key is held down (dwell time)."),
    "seek_time": ("Seek time", "s",
                  "The gap between releasing one key and pressing the next (flight time). Measured from the "
                  "previous press instead when keys overlap."),
    "interval": ("Press-to-press interval", "s", "Time from one key press to the next."),
    "time_diff": ("Time between presses", "s", "Time between consecutive key presses. Pauses over 5 s start a new burst."),
    "mean_ngram_dwell_time": ("Recent hold time", "s", "Average hold time over the last 3 keys."),
    "std_ngram_dwell_time": ("Hold-time consistency", "s",
                             "Spread of hold time over the last 3 keys. Higher means less even."),
    "mean_ngram_flight_time": ("Recent seek time", "s", "Average seek time over the last 3 keys."),
    "std_ngram_flight_time": ("Seek-time consistency", "s", "Spread of seek time over the last 3 keys."),
    "key_per_second": ("Typing speed", "", "Keys per second since the start of the current burst."),
    "wpm": ("Words per minute", "", "Spaces typed per minute in the current burst."),
    "rolling_mean_kps": ("Recent typing speed", "", "Keys per second, averaged over the last 5 keys."),
    "rolling_std_kps": ("Typing-speed variability", "", "Spread of keys per second over the last 5 keys."),
    "rolling_mean_wpm": ("Recent words per minute", "", "Words per minute, averaged over the last 5 keys."),
    "rolling_std_wpm": ("WPM variability", "", "Spread of words per minute over the last 5 keys."),
    "variation_in_typing_speed": ("Per-key speed variation", "",
                                  "How much typing speed varies between presses of the same key."),
    "error": ("Corrections", "share", "Share of keystrokes later erased with Backspace/Delete."),
    "error_rate": ("Error rate", "share", "Running share of keystrokes that were corrections."),
    "accuracy": ("Typing accuracy", "share", "1 − error rate."),
    "is_burst": ("Burst typing", "share", "Share of presses that came within 0.1 s of the previous one."),
    "burst_id": ("Burst counter", "",
                 "The index of the current typing burst. This is session position, not a personal trait: if it "
                 "shows up as a reason, read it with caution."),
    "start_time": ("Press time", "s",
                   "When the key was pressed, relative to the start of the window. This is session position, "
                   "not a personal trait."),
    "end_time": ("Release time", "s",
                 "When the key was released, relative to the start of the window. This is session position."),
}

FEATURE_PATTERNS = {
    "key_…": ("Key frequency", "share",
              "How often this key appears in the window. A different share from usual can mean different "
              "habits (e.g. which Shift key, Backspace use) or simply different text being typed."),
    "combination_…": ("Key-combination frequency", "share",
                      "How often this set of simultaneously held keys (e.g. Shift+letter) appears in the window."),
}


def feature(col: str) -> tuple[str, str, str]:
    """(name, unit, description) for any model feature column."""
    if col in FEATURES:
        return FEATURES[col]
    if col.startswith("key_"):
        name, unit, desc = FEATURE_PATTERNS["key_…"]
        return f"Key {col[4:]}", unit, desc
    if col.startswith("combination_"):
        name, unit, desc = FEATURE_PATTERNS["combination_…"]
        return f"Combination {col[12:]}", unit, desc
    return col.replace("_", " "), "", ""


def format_value(col: str, value) -> str:
    if value is None:
        return "–"
    unit = feature(col)[1]
    if unit == "s":
        return f"{value * 1000:.0f} ms"
    if unit == "share":
        return f"{value:.0%}" if value >= 0.01 or value == 0 else f"{value:.1%}"
    return f"{value:.2f}"


def explain_reason(r: dict) -> dict:
    """Plain-language reading of one flagged parameter: {flagged, what, shap}."""
    col = r.get("feature") or ""
    name = feature(col)[0]
    value, typical = r.get("value"), r.get("typical")
    low, high = r.get("typical_low"), r.get("typical_high")
    direction = r.get("direction")

    if typical is None:
        what = f"{name} was {format_value(col, value)}. No usual value is known for this model."
    elif low is not None:
        usual = f"{format_value(col, low)}–{format_value(col, high)} (median {format_value(col, typical)})"
        if direction in ("higher", "lower"):
            ratio = f", {value / typical:.1f}× the median" if typical and value and direction == "higher" else ""
            what = (f"{name} was {format_value(col, value)}, {'above' if direction == 'higher' else 'below'} "
                    f"the usual {usual}{ratio}.")
        else:
            what = f"{name} was {format_value(col, value)}, inside the usual {usual}."
    else:
        what = f"{name} was {format_value(col, value)}; the user's usual share is {format_value(col, typical)}."

    impact = r.get("impact") or 0.0
    odds = math.exp(impact)
    shap = (f"SHAP +{impact:.2f}: on its own, this feature moved the model {impact:.2f} log-odds toward "
            f"'not the enrolled user', which multiplies the odds by about {odds:.1f}×.")
    if direction == "within range":
        shap += (" The value itself is not unusual. It counts because of how it combines with the other features "
                 "in this window.")
    return {"flagged": direction in ("higher", "lower"), "what": what, "shap": shap}


def scale(col: str, value, prof: dict) -> dict | None:
    """Positions (0-100 %) for a feature's range bar on the Statistics page."""
    if not prof or prof.get("low") is None:
        return None
    unit = feature(col)[1]
    top = 1.0 if unit == "share" else max(prof["high"] * 2.5, prof["typical"] * 3, 1e-9)
    pos = lambda v: max(0.0, min(100.0, 100.0 * v / top))
    return {"low": pos(prof["low"]), "high": pos(prof["high"]), "typical": pos(prof["typical"]),
            "value": None if value is None else pos(value),
            "off_scale": value is not None and value > top,
            "max": format_value(col, top)}
