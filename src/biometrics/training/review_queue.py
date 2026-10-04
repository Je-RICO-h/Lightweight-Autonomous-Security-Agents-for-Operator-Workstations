"""
Review queue and "refine model".

When the agent raises a challenge or an alert and the person then passes the
OS re-authentication, the model was wrong: that was the enrolled user. The
keystrokes from that suspicious stretch are saved here, flagged for review.
They are never learned automatically. "Refine model" (menu, `biometrics
refine`, or the dashboard) adds them to the training data labelled as the
main user and retrains.

Re-authentication at the *critical* level is deliberately not saved: by then
an alert was already dismissed, so the password alone is weaker evidence
that it was really the owner typing.
"""
import csv
import datetime
import json
import os

from biometrics import paths
from biometrics.training.train_xgboost import MAIN_USER_LABEL


def save_for_review(windows: list[list[dict]], incident: dict, queue_dir=None) -> str | None:
    """windows: the keystroke windows of the suspicious stretch, oldest first."""
    queue_dir = queue_dir or paths.REVIEW_DIR
    rows = []
    for window in windows:
        for i, row in enumerate(window):
            row = dict(row)
            if i == 0:
                row["start_time"] = 0.0  # marks a new chunk for the feature pipeline
            rows.append(row)
    if not rows:
        return None

    os.makedirs(queue_dir, exist_ok=True)
    stem = f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}-review-{incident['level']}-{incident['id']}"
    csv_path = os.path.join(queue_dir, f"{stem}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(queue_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
        json.dump({"level": incident["level"], "impostor_score": incident["impostor_score"],
                   "windows": len(windows), "keystrokes": len(rows),
                   "reasons": incident.get("reasons", [])}, f, indent=2)
    paths.hand_to_user(queue_dir)
    return csv_path


def pending(queue_dir=None) -> list[dict]:
    """Sessions waiting for refine, oldest first: {name, keystrokes, level, impostor_score}."""
    queue_dir = queue_dir or paths.REVIEW_DIR
    if not os.path.isdir(queue_dir):
        return []
    sessions = []
    for name in sorted(f[:-4] for f in os.listdir(queue_dir) if f.endswith(".csv")):
        meta = {}
        sidecar = os.path.join(queue_dir, f"{name}.json")
        if os.path.exists(sidecar):
            with open(sidecar, encoding="utf-8") as f:
                meta = json.load(f)
        with open(os.path.join(queue_dir, f"{name}.csv"), encoding="utf-8") as f:
            keystrokes = sum(1 for _ in f) - 1
        sessions.append({"name": name, "keystrokes": keystrokes, "level": meta.get("level"),
                         "impostor_score": meta.get("impostor_score")})
    return sessions


def refine(queue_dir=None, log=print) -> dict:
    """
    Moves every pending session into data/raw labelled as the main user, then
    re-processes the dataset and retrains the one model in models/xgboost.
    Returns the training metrics plus how many sessions/keystrokes were added.
    """
    from biometrics.processing.pipeline import process
    from biometrics.training.train_xgboost import train

    queue_dir = queue_dir or paths.REVIEW_DIR
    sessions = pending(queue_dir)
    for s in sessions:
        # The label is the part after the last "_" of the file name (see
        # processing.features.load_raw_sessions), so this lands as the main user.
        dest = paths.RAW_DIR / f"{s['name'].replace('_', '-')}_{MAIN_USER_LABEL}.csv"
        src = os.path.join(queue_dir, f"{s['name']}.csv")
        with open(src, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            rows[0]["start_time"] = "0.0"  # older queue files start at ~1e-7, not exactly 0
            with open(dest, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        os.remove(src)
        sidecar = os.path.join(queue_dir, f"{s['name']}.json")
        if os.path.exists(sidecar):
            os.remove(sidecar)
    log(f"Added {len(sessions)} reviewed session(s) to {paths.RAW_DIR} as '{MAIN_USER_LABEL}'")

    process(str(paths.RAW_DIR), str(paths.PROCESSED_CSV))
    result = train(str(paths.PROCESSED_CSV), str(paths.MODEL_DIR))
    paths.hand_to_user(paths.RAW_DIR, paths.PROCESSED_CSV.parent, paths.MODEL_DIR)
    return {**result, "sessions_added": len(sessions),
            "keystrokes_added": sum(s["keystrokes"] for s in sessions)}
