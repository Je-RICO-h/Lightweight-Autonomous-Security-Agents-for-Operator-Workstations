import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import xgboost as xgb
import joblib
import numpy as np
import pandas as pd
import time


class Inference:
    def __init__(self, model_paths, confidence_threshold=0.9, SEQUENCE_LENGTH=50):
        self.SEQUENCE_LENGTH = SEQUENCE_LENGTH  # Number of key presses in each sequence
        self.confidence_threshold = confidence_threshold

        if len(model_paths) > 0:
            self.model, self.le, self.ohe = self.__load_model(model_paths)
        else:
            raise Exception("Provide the model paths")


    def predict(self, data, return_feature_row=False):
        if len(data) < 10:
            return

        start_time_prep = time.time()
        prepared_data = self.__prepare_live_data(data)
        end_time_prep = time.time()

        processing_time = end_time_prep - start_time_prep
        print(f"\nData Process time: {processing_time:.4f} seconds.")

        start_time_pred = time.time()

        probabilities = self.model.predict_proba(prepared_data)
        predicted_class_index = np.argmax(probabilities, axis=1)
        predicted_confidence = np.max(probabilities, axis=1)

        predicted_class_final = np.bincount(predicted_class_index).argmax()
        predicted_confidence_final = np.mean(predicted_confidence)

        end_time_pred = time.time()

        prediction_time = end_time_pred - start_time_pred
        print(f"Predict time (Model Run): {prediction_time:.4f} seconds.")

        predicted_label = self.le.inverse_transform([predicted_class_final])[0]

        print(f"Result: {predicted_label} (Confidence: {predicted_confidence_final:.2f})")

        if return_feature_row:
            # Representative feature row for this window (mean across its
            # keystrokes), same 609-column shape as the model's training
            # input. Explained locally by agent/explain.py (SHAP); it never
            # leaves this machine.
            feature_row = prepared_data.mean(axis=0).to_dict()
            return predicted_label, predicted_confidence_final, feature_row

        return predicted_label, predicted_confidence_final


    def __load_model(self, model_paths):
        try:
            loaded_model = xgb.XGBClassifier()
            loaded_model.load_model(model_paths['model_path'])
            loaded_le = joblib.load(model_paths['le_path'])
            loaded_ohe = np.load(model_paths['ohe_path'], allow_pickle=True)

            return loaded_model, loaded_le, loaded_ohe

        except FileNotFoundError as e:
            print(f"Error: {e}. One or more model files were not found.")
            print("Please ensure you have saved the files correctly.")
            exit(1)


    def __prepare_live_data(self, data):
        df = pd.DataFrame(data)

        # Drop language
        df = df.drop(columns="language")

        # Data preprocessing, column creations
        df = self.__add_missing_hold_end_times(df)
        df['chunk_id'] = (df['start_time'] == 0).cumsum()
        df = df.groupby('chunk_id', group_keys=False).apply(self.__add_cumulative_and_wpm)
        df = self.__add_ngrams(df)
        df = self.__add_rolling_data(df)
        df = self.__add_typing_speed_variation(df)
        df = self.__add_burst_calculation(df)

        # Preparation for inference
        df = self.__prepare_for_inference(df)

        # Reindex
        df = df.reindex(columns=self.ohe, fill_value=0)

        return df


    def __prepare_for_inference(self, df):
        #Drop unnecessary columns
        df = df.drop(columns=['start_time_timestamp', 'end_time_timestamp'])

        #Convert to correct type
        df['error'] = df['error'].astype(int)
        df['is_burst'] = df['is_burst'].astype(int)

        # One-hot encode the categorical columns
        df = pd.get_dummies(df, columns=['key', 'combination'], dtype=int)
        df = self.__add_missing_data(df)

        return df


    def __add_missing_hold_end_times(self, df):
        # Deal with hold times
        nan_hold_time_idx = df[df['hold_time'].isna()].index

        # Check to make sure there are missing hold times before continuing.
        if nan_hold_time_idx.empty:
            return df

        last_idx = df.index[-1]

        for idx in nan_hold_time_idx:
            # Handle the last element separately to prevent KeyError
            if idx == last_idx:
                if idx > 0:
                    prev_val = df.loc[idx - 1, 'hold_time']
                    df.loc[idx, 'hold_time'] = prev_val
                continue

            prev_val = df.loc[idx - 1, 'hold_time']
            next_val = df.loc[idx + 1, 'hold_time']
            df.loc[idx, 'hold_time'] = (prev_val + next_val) / 2

        # Deal with end times
        nan_end_time_idx = df[df['end_time'].isna()].index

        # Check to make sure there are missing end times before continuing.
        if nan_end_time_idx.empty:
            return df

        for idx in nan_end_time_idx:
            # Handle the last element separately to prevent KeyError
            if idx == last_idx:
                if idx > 0:
                    prev_val = df.loc[idx - 1, 'end_time']
                    df.loc[idx, 'end_time'] = prev_val
                continue

            prev_val = df.loc[idx - 1, 'end_time']
            next_val = df.loc[idx + 1, 'end_time']

            # Apply the new condition: if next value is smaller than previous, make it 0
            if next_val < prev_val:
                df.loc[idx, 'end_time'] = 0
            else:
                # Otherwise, apply the original rule: add the next end time to the previous
                df.loc[idx, 'end_time'] = prev_val + next_val

        return df


    def __add_cumulative_and_wpm(self, df):
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

        #Drop Unnecessary columns
        df = df.drop(columns=['cumulative_keys', 'elapsed_time_seconds', 'cumulative_spaces'])

        return df


    def __add_ngrams(self, df):
        df['mean_ngram_dwell_time'] = df['hold_time'].rolling(
            window=3, min_periods=2).mean().fillna(0)
        df['std_ngram_dwell_time'] = df['hold_time'].rolling(
            window=3, min_periods=2).std().fillna(0)

        # Calculate the mean and standard deviation of seek_time over a rolling window.
        df['mean_ngram_flight_time'] = df['seek_time'].rolling(
            window=3, min_periods=2).mean().fillna(0)
        df['std_ngram_flight_time'] = df['seek_time'].rolling(
            window=3, min_periods=2).std().fillna(0)

        return df


    def __add_rolling_data(self, df):
        # Add n-gram-based features for hold and seek times
        df['mean_ngram_dwell_time'] = df['hold_time'].rolling(window=3, min_periods=2).mean().fillna(0)
        df['std_ngram_dwell_time'] = df['hold_time'].rolling(window=3, min_periods=2).std().fillna(0)
        df['mean_ngram_flight_time'] = df['seek_time'].rolling(window=3, min_periods=2).mean().fillna(0)
        df['std_ngram_flight_time'] = df['seek_time'].rolling(window=3, min_periods=2).std().fillna(0)

        # Add rolling features
        df['rolling_mean_kps'] = df['key_per_second'].rolling(window=5, min_periods=1).mean().fillna(0)
        df['rolling_std_kps'] = df['key_per_second'].rolling(window=5, min_periods=1).std().fillna(0)
        df['rolling_mean_wpm'] = df['wpm'].rolling(window=5, min_periods=1).mean().fillna(0)
        df['rolling_std_wpm'] = df['wpm'].rolling(window=5, min_periods=1).std().fillna(0)

        return df


    def __add_typing_speed_variation(self, df):
        #calculate standard deviation
        variation_in_typing_speed = df.groupby('key')['key_per_second'].std()

        # Add the standard deviation back
        df['variation_in_typing_speed'] = df['key'].map(variation_in_typing_speed)

        return df


    def __add_burst_calculation(self, df):
        # Calculate the interval
        df['interval'] = df['start_time'].diff()

        # Define a threshold for bursts
        threshold = 0.1
        df['is_burst'] = df['interval'] <= threshold

        # Assign burst IDs to consecutive keys that fall within the threshold
        burst_id = 0
        burst_ids = []

        for is_burst in df['is_burst']:
            if not is_burst:  # If there's a pause, increment the burst ID
                burst_id += 1
            burst_ids.append(burst_id)

        df['burst_id'] = burst_ids

        # Count the number of bursts
        burst_counts = df['burst_id'].nunique() - 1  # Subtract 1 to exclude the initial NaN burst

        return df


    def __add_missing_data(self, df):
        # Create Series with previous and next non-NaN values
        prev_values_end_time = df['end_time'].ffill()
        next_values_end_time = df['end_time'].bfill()

        # Identify missing indices in the 'end_time' column
        missing_indices_end_time = df['end_time'].isnull()

        # For the missing positions, calculate the sum of previous and next non-missing values
        df.loc[missing_indices_end_time, 'end_time'] = \
            prev_values_end_time[missing_indices_end_time] + next_values_end_time[missing_indices_end_time]

        df['variation_in_typing_speed'] = df['variation_in_typing_speed'].interpolate(method='linear', limit_direction='both')
        df['hold_time'] = df['hold_time'].interpolate(method='linear', limit_direction='both')
        df['interval'] = df['interval'].interpolate(method='linear', limit_direction='both')

        return df
