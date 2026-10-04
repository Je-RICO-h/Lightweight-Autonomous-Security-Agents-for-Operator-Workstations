"""
Every file location the project uses, anchored to the repository root so it
works from any working directory. One user per machine, one model per user:
there is exactly one model directory.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw"                        # labelled sessions: <timestamp>_<label>.csv
PROCESSED_CSV = ROOT / "data" / "processed" / "data_processed.csv"
REVIEW_DIR = ROOT / "data" / "retrain_review"          # re-authenticated windows waiting for "refine model"
INFER_OUTPUT_DIR = ROOT / "data" / "review"
MODEL_DIR = ROOT / "models" / "xgboost"

MODEL_FILE = MODEL_DIR / "xgboost_model.json"
LABEL_ENCODER_FILE = MODEL_DIR / "xgboost_label_encoder.joblib"
OHE_COLS_FILE = MODEL_DIR / "xgboost_ohe_cols.npy"
THRESHOLDS_FILE = MODEL_DIR / "thresholds.json"
MODEL_INFO_FILE = MODEL_DIR / "model_info.json"
FEATURE_PROFILE_FILE = MODEL_DIR / "feature_profile.json"


def model_paths() -> dict:
    """The dict Inference() expects."""
    return {"model_path": str(MODEL_FILE), "le_path": str(LABEL_ENCODER_FILE), "ohe_path": str(OHE_COLS_FILE)}


def owner() -> tuple[int, int]:
    """
    (uid, gid) of whoever owns this project. entrypoint.sh runs everything as
    root, and SUDO_UID can't be trusted to name the real person: started from
    an already-root shell (e.g. `sudo ./entrypoint.sh`), the inner sudo
    records root. The project folder's owner is always the actual user.
    """
    st = ROOT.stat()
    return st.st_uid, st.st_gid


def hand_to_user(*targets) -> None:
    """Running as root, give files we created back to the project's owner, so
    later plain-user commands can still read and replace them."""
    uid, gid = owner()
    if os.geteuid() != 0 or uid == 0:
        return
    for target in map(Path, targets):
        if not target.exists():
            continue
        for path in [target, *(target.rglob("*") if target.is_dir() else [])]:
            os.chown(path, uid, gid)
