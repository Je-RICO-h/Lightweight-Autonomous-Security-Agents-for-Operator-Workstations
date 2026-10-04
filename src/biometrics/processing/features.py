"""
Feature engineering shared by the offline processing pipeline (Processing.ipynb)
and reused, in a streaming form, by biometrics.inference.xgboost_inference.Inference.
Ported from Processing/Processing.ipynb.
"""
import os

import numpy as np
import pandas as pd

MAIN_LABEL = "User"


def load_raw_sessions(raw_data_dir: str) -> pd.DataFrame:
    """Aggregate every session CSV in raw_data_dir into a single dataframe."""
    all_files_df = []

    for file in sorted(os.listdir(raw_data_dir)):
        if file.endswith(".csv"):
            file_path = os.path.join(raw_data_dir, file)
            temp_df = pd.read_csv(file_path)
            temp_df['label'] = file.split(".")[0].split("_")[-1]
            temp_df['session_id'] = file.split(".")[0]
            all_files_df.append(temp_df)

    if all_files_df:
        return pd.concat(all_files_df, ignore_index=True)
    return pd.DataFrame()


def add_missing_hold_end_times(df: pd.DataFrame) -> pd.DataFrame:
    """Add the missing hold times and end times (last entry per threshold chunk is sometimes incomplete)."""
    nan_hold_time_idx = df[df['hold_time'].isna()].index
    last_idx = df.index[-1]

    if not nan_hold_time_idx.empty:
        for idx in nan_hold_time_idx:
            if idx == last_idx:
                if idx > 0:
                    prev_val = df.loc[idx - 1, 'hold_time']
                    df.loc[idx, 'hold_time'] = prev_val
                continue

            prev_val = df.loc[idx - 1, 'hold_time']
            next_val = df.loc[idx + 1, 'hold_time']
            df.loc[idx, 'hold_time'] = (prev_val + next_val) / 2

    nan_end_time_idx = df[df['end_time'].isna()].index

    if not nan_end_time_idx.empty:
        for idx in nan_end_time_idx:
            if idx == last_idx:
                if idx > 0:
                    prev_val = df.loc[idx - 1, 'end_time']
                    df.loc[idx, 'end_time'] = prev_val
                continue

            prev_val = df.loc[idx - 1, 'end_time']
            next_val = df.loc[idx + 1, 'end_time']

            if next_val < prev_val:
                df.loc[idx, 'end_time'] = 0
            else:
                df.loc[idx, 'end_time'] = prev_val + next_val

    return df


def add_cumulative_and_wpm(df: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative key/space counts and derived key-per-second / words-per-minute features."""
    df['start_time_timestamp'] = pd.to_numeric(df['start_time_timestamp'])
    df['end_time_timestamp'] = pd.to_numeric(df['end_time_timestamp'])

    df['time_diff'] = df['start_time_timestamp'].diff().fillna(0)

    df['burst_id'] = (df['time_diff'] > 5.0).cumsum()

    df['cumulative_keys'] = df.groupby('burst_id').cumcount() + 1
    df['cumulative_spaces'] = df.groupby('burst_id')['key'].transform(lambda x: (x == 'Key.space').cumsum())

    df['elapsed_time_seconds'] = df.groupby('burst_id')['time_diff'].cumsum()

    df.loc[:, 'key_per_second'] = df['cumulative_keys'] / df['elapsed_time_seconds'].replace(0, np.nan)
    df.loc[:, 'wpm'] = (df['cumulative_spaces'] / (df['elapsed_time_seconds'] / 60)).replace([np.inf, -np.inf], 0)

    first_in_burst = df['burst_id'].diff() != 0
    df.loc[first_in_burst, 'elapsed_time_seconds'] = 0.001
    df.loc[first_in_burst, 'key_per_second'] = df.loc[first_in_burst, 'cumulative_keys'] / df.loc[first_in_burst, 'elapsed_time_seconds']
    df.loc[first_in_burst, 'wpm'] = (df.loc[first_in_burst, 'cumulative_spaces'] / (df.loc[first_in_burst, 'elapsed_time_seconds'] / 60)).replace([np.inf, -np.inf], 0)

    drop_cols = [c for c in ['cumulative_keys', 'elapsed_time_seconds', 'cumulative_spaces', 'chunk_id'] if c in df.columns]
    return df.drop(columns=drop_cols)


def add_ngrams(df: pd.DataFrame) -> pd.DataFrame:
    """Add n-gram-based rolling statistics for dwell (hold) and flight (seek) times."""
    df['mean_ngram_dwell_time'] = df['hold_time'].rolling(window=3, min_periods=2).mean().fillna(0)
    df['std_ngram_dwell_time'] = df['hold_time'].rolling(window=3, min_periods=2).std().fillna(0)
    df['mean_ngram_flight_time'] = df['seek_time'].rolling(window=3, min_periods=2).mean().fillna(0)
    df['std_ngram_flight_time'] = df['seek_time'].rolling(window=3, min_periods=2).std().fillna(0)
    return df


def add_rolling_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean/std features for typing speed and words-per-minute."""
    df = add_ngrams(df)
    df['rolling_mean_kps'] = df['key_per_second'].rolling(window=5, min_periods=1).mean().fillna(0)
    df['rolling_std_kps'] = df['key_per_second'].rolling(window=5, min_periods=1).std().fillna(0)
    df['rolling_mean_wpm'] = df['wpm'].rolling(window=5, min_periods=1).mean().fillna(0)
    df['rolling_std_wpm'] = df['wpm'].rolling(window=5, min_periods=1).std().fillna(0)
    return df


def add_typing_speed_variation(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-key typing speed variation (standard deviation of key_per_second grouped by key)."""
    variation_in_typing_speed = df.groupby('key')['key_per_second'].std()
    df['variation_in_typing_speed'] = df['key'].map(variation_in_typing_speed)
    return df


def add_burst_calculation(df: pd.DataFrame) -> pd.DataFrame:
    """Add burst-id features based on inter-key interval thresholding."""
    df['interval'] = df['start_time'].diff()

    threshold = 0.1
    df['is_burst'] = df['interval'] <= threshold

    burst_id = 0
    burst_ids = []
    for is_burst in df['is_burst']:
        if not is_burst:
            burst_id += 1
        burst_ids.append(burst_id)

    df['burst_id'] = burst_ids
    return df


def build_processed_dataset(raw_data_dir: str) -> pd.DataFrame:
    """
    Full offline pipeline: aggregate raw session CSVs, engineer features,
    and return the combined dataframe ready to be written to data/processed.
    Mirrors Processing/Processing.ipynb end to end.
    """
    data_aggregated = load_raw_sessions(raw_data_dir)
    if data_aggregated.empty:
        return data_aggregated

    data_aggregated = data_aggregated.drop(columns="language")

    normal_data = data_aggregated[data_aggregated['label'] == MAIN_LABEL].reset_index(drop=True)
    anomaly_data = data_aggregated[data_aggregated['label'] != MAIN_LABEL].reset_index(drop=True)

    normal_data = add_missing_hold_end_times(normal_data)
    anomaly_data = add_missing_hold_end_times(anomaly_data)

    normal_data['chunk_id'] = (normal_data['start_time'] == 0).cumsum()
    anomaly_data['chunk_id'] = (anomaly_data['start_time'] == 0).cumsum()

    normal_data = normal_data.groupby('chunk_id', group_keys=False).apply(add_cumulative_and_wpm)
    anomaly_data = anomaly_data.groupby('chunk_id', group_keys=False).apply(add_cumulative_and_wpm)

    normal_data = add_rolling_data(normal_data)
    anomaly_data = add_rolling_data(anomaly_data)

    normal_data = add_typing_speed_variation(normal_data)
    anomaly_data = add_typing_speed_variation(anomaly_data)

    normal_data = add_burst_calculation(normal_data)
    anomaly_data = add_burst_calculation(anomaly_data)

    return pd.concat([normal_data, anomaly_data], ignore_index=True)
