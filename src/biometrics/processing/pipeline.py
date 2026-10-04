"""Entry point for turning raw keystroke session CSVs into the processed training dataset."""
import os

from biometrics.processing.features import build_processed_dataset


def process(raw_data_dir: str, output_csv: str) -> str:
    df = build_processed_dataset(raw_data_dir)

    if df.empty:
        raise ValueError(f"No session CSV files found in {raw_data_dir}")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Processed {len(df)} rows from {raw_data_dir} -> {output_csv}")
    return output_csv
