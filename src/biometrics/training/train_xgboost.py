"""
Trains the XGBoost "is this the main user" classifier from the processed dataset.
Ported from Model_Training/XgBoost.ipynb.

Besides the model itself, every train run writes three files the protection
agent needs (all into model_dir):
  thresholds.json       tau_warn / tau_critical / EER / AUC (see thresholds.py)
  model_info.json       accuracy, per-class precision/recall, params -- shown
                        on the server dashboard's Statistics page
  feature_profile.json  the enrolled user's typical value per feature -- lets
                        the agent explain an alert as "0.14 s, usually 0.08 s"
"""
import datetime
import json
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

from biometrics.training.thresholds import compute_thresholds

MAIN_USER_LABEL = "User"
IMPOSTOR_LABEL = "Other"

DEFAULT_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'mlogloss',
    'eta': 0.1,  # Learning rate
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'seed': 42,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
}


def _is_corrupted(column_name: str) -> bool:
    """
    Escaped-byte key names, and names XGBoost refuses as feature names (it
    rejects '[', ']' and '<', e.g. the one-hot column for the '[' key). Such
    keys are simply not features; inference aligns to the model's saved column
    list, so it drops them the same way.
    """
    return '\\x' in column_name or '\\u' in column_name or any(c in column_name for c in "[]<")


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['start_time_timestamp', 'end_time_timestamp'])

    df['error'] = df['error'].astype(int)
    df['is_burst'] = df['is_burst'].astype(int)

    df_encoded = pd.get_dummies(df, columns=['key', 'combination'], dtype=int)

    clean_columns = [col for col in df_encoded.columns if not _is_corrupted(col)]
    df_encoded = df_encoded[clean_columns].copy()

    prev_values_end_time = df_encoded['end_time'].ffill()
    next_values_end_time = df_encoded['end_time'].bfill()
    missing_indices_end_time = df_encoded['end_time'].isnull()
    df_encoded.loc[missing_indices_end_time, 'end_time'] = \
        prev_values_end_time[missing_indices_end_time] + next_values_end_time[missing_indices_end_time]

    df_encoded['variation_in_typing_speed'] = df_encoded['variation_in_typing_speed'].interpolate(
        method='linear', limit_direction='both')
    df_encoded['hold_time'] = df_encoded['hold_time'].interpolate(method='linear', limit_direction='both')
    df_encoded['interval'] = df_encoded['interval'].interpolate(method='linear', limit_direction='both')

    assert not df_encoded.isna().any().any(), "DataFrame contains NaN values after cleaning!"

    return df_encoded


def _split(df_encoded: pd.DataFrame):
    """
    Leakage-safe split by session_id: 60% train / 20% validation / 20% test.
    Deterministic (random_state=42), so describe_model() recovers the exact
    same test set for an already-trained model.
    """
    df_encoded = df_encoded.copy()
    df_encoded.loc[df_encoded['label'] != MAIN_USER_LABEL, 'label'] = IMPOSTOR_LABEL

    session_ids = df_encoded['session_id']
    X = df_encoded.drop(columns=['label', 'session_id'])
    le = LabelEncoder()
    y_encoded = le.fit_transform(df_encoded['label'])

    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    temp_idx, test_idx = next(gss_test.split(X, y_encoded, groups=session_ids))
    groups_temp = session_ids.iloc[temp_idx]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_rel, val_rel = next(gss_val.split(X.iloc[temp_idx], y_encoded[temp_idx], groups=groups_temp))
    train_idx, val_idx = temp_idx[train_rel], temp_idx[val_rel]

    train_sessions = set(session_ids.iloc[train_idx])
    val_sessions = set(session_ids.iloc[val_idx])
    test_sessions = set(session_ids.iloc[test_idx])
    assert not (train_sessions & val_sessions)
    assert not (train_sessions & test_sessions)
    assert not (val_sessions & test_sessions)

    return X, y_encoded, le, train_idx, test_idx


def _evaluate(model, le, X_test, y_test) -> tuple[dict, dict]:
    """Returns (metrics, thresholds) on the held-out test set."""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, labels=list(range(len(le.classes_))),
                                   target_names=list(le.classes_), zero_division=0, output_dict=True)

    impostor_class_idx = list(le.classes_).index(IMPOSTOR_LABEL)
    y_test_impostor = (y_test == impostor_class_idx).astype(int)
    y_proba_impostor = model.predict_proba(X_test)[:, impostor_class_idx]
    thresholds = compute_thresholds(y_test_impostor, y_proba_impostor)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "per_class": {
            cls: {k: float(report[cls][k]) for k in ("precision", "recall", "f1-score", "support")}
            for cls in le.classes_
        },
    }
    return metrics, thresholds


def is_one_hot(col: str) -> bool:
    """key_<key> / combination_<keys> one-hot columns (key_per_second is numeric)."""
    return col.startswith(("key_", "combination_")) and col != "key_per_second"


def _feature_profile(X_user: pd.DataFrame) -> dict:
    """
    The enrolled user's typical value for every model feature, from their
    training rows only. One-hot key_/combination_ columns get their mean
    (= how often that key/combination appears); numeric columns get the
    median and interquartile range.
    """
    profile = {}
    for col in X_user.columns:
        values = X_user[col].astype(float)
        if is_one_hot(col):
            profile[col] = {"typical": float(values.mean())}
        else:
            p25, p50, p75 = values.quantile([0.25, 0.5, 0.75])
            profile[col] = {"typical": float(p50), "low": float(p25), "high": float(p75)}
    return profile


PARAMS_SHOWN = ("learning_rate", "eta", "max_depth", "subsample", "colsample_bytree",
                "reg_alpha", "reg_lambda", "device")


def _write_agent_artifacts(model_dir: str, model, le, X: pd.DataFrame, y_encoded, train_idx, test_idx,
                           metrics: dict, trained_at: str, params: dict, params_note: str = None) -> None:
    """Writes model_info.json and feature_profile.json (see module docstring)."""
    info = {
        "trained_at": trained_at,
        "algorithm": "XGBoost binary classifier (User vs Other)",
        "n_estimators": int(model.get_booster().num_boosted_rounds()),
        "params": {k: params[k] for k in PARAMS_SHOWN if params.get(k) is not None},
        "params_note": params_note,
        "n_features": int(X.shape[1]),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        **metrics,
    }
    with open(os.path.join(model_dir, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    user_idx = list(le.classes_).index(MAIN_USER_LABEL)
    X_user = X.iloc[train_idx][y_encoded[train_idx] == user_idx]
    with open(os.path.join(model_dir, "feature_profile.json"), "w") as f:
        json.dump(_feature_profile(X_user), f)


def train(
    processed_csv: str,
    model_dir: str,
    params: dict = None,
    n_estimators: int = 100,
    use_gpu: bool = False,
) -> dict:
    """
    Train the XGBoost user/other classifier and write the model, label encoder,
    one-hot column list and the agent artifacts to model_dir. Returns evaluation metrics.
    """
    df_encoded = _prepare_features(pd.read_csv(processed_csv))
    X, y_encoded, le, train_idx, test_idx = _split(df_encoded)

    run_params = dict(DEFAULT_PARAMS)
    if params:
        run_params.update(params)
    if use_gpu:
        run_params['device'] = 'cuda'

    model = xgb.XGBClassifier(n_estimators=n_estimators, **run_params)
    model.fit(X.iloc[train_idx], y_encoded[train_idx])

    metrics, thresholds = _evaluate(model, le, X.iloc[test_idx], y_encoded[test_idx])

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "xgboost_model.json")
    le_path = os.path.join(model_dir, "xgboost_label_encoder.joblib")
    ohe_path = os.path.join(model_dir, "xgboost_ohe_cols.npy")
    thresholds_path = os.path.join(model_dir, "thresholds.json")

    model.save_model(model_path)
    joblib.dump(le, le_path)
    np.save(ohe_path, X.columns.values)
    with open(thresholds_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    _write_agent_artifacts(model_dir, model, le, X, y_encoded, train_idx, test_idx, metrics,
                           trained_at=datetime.datetime.now().isoformat(timespec="seconds"),
                           params=run_params)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Thresholds (tau_warn={thresholds['tau_warn']:.4f}, "
          f"tau_critical={thresholds['tau_critical']:.4f}, "
          f"EER={thresholds['eer']:.4f}, AUC={thresholds['auc']:.4f}) "
          f"saved to: {thresholds_path}")

    return {
        **metrics,
        "thresholds": thresholds,
        "model_path": model_path,
        "le_path": le_path,
        "ohe_path": ohe_path,
        "thresholds_path": thresholds_path,
    }


def describe_model(processed_csv: str, model_dir: str) -> dict:
    """
    Regenerates thresholds.json, model_info.json and feature_profile.json for
    an ALREADY-trained model without retraining it: rebuilds the identical
    session split and re-evaluates the saved model on its own test set.
    """
    df_encoded = _prepare_features(pd.read_csv(processed_csv))
    X, y_encoded, le, train_idx, test_idx = _split(df_encoded)

    model_path = os.path.join(model_dir, "xgboost_model.json")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    ohe_cols = list(np.load(os.path.join(model_dir, "xgboost_ohe_cols.npy"), allow_pickle=True))
    X = X.reindex(columns=ohe_cols, fill_value=0)

    metrics, thresholds = _evaluate(model, le, X.iloc[test_idx], y_encoded[test_idx])
    with open(os.path.join(model_dir, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    trained_at = datetime.datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat(timespec="seconds")
    # A saved XGBoost model does not keep its training hyperparameters, so
    # they can't be read back -- record the project defaults and say so.
    _write_agent_artifacts(model_dir, model, le, X, y_encoded, train_idx, test_idx, metrics, trained_at,
                           params=DEFAULT_PARAMS,
                           params_note="Assumed project defaults: this model was trained before "
                                       "model_info.json existed, and saved XGBoost models do not "
                                       "store their training hyperparameters.")
    return {**metrics, "thresholds": thresholds}
