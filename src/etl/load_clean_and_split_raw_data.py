import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.logger import setup_logger
from src.etl.clean_data import clean_data

logger = setup_logger()

DATA_DIR = Path('data/raw')
RAW_DATA_PATH = Path('data/raw/Location1.csv')

def split_data(df):
    n = len(df)
    cutoff_eval = int(n *  0.70)
    cutoff_holdout = int(n * 0.85)
    train_df = df.iloc[:cutoff_eval]
    eval_df = df.iloc[cutoff_eval: cutoff_holdout]
    holdout_df = df.iloc[cutoff_holdout:]
    return train_df, eval_df, holdout_df

def load_and_split_raw_data(
        raw_path: Path | str,
        output_dir: Path | str
):
    """ Load raw dataset and split it into train, eval, and holdout,
    and save to output_dir"""

    logger.info(
        "Raw data  cleaning and split started"
    )
    
    # load raw dataset
    df = pd.read_csv(raw_path)
    # clean dataset
    df = clean_data(df)
    # splits
    raw_train_df, raw_eval_df, raw_holdout_df = split_data(df)
    # save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_train_df.to_csv(output_dir / "cleaned_train.csv", index=False)
    raw_eval_df.to_csv(output_dir / "cleaned_eval.csv", index=False)
    raw_holdout_df.to_csv(output_dir / "cleaned_holdout.csv", index=False)
    logger.info(
        "Raw data cleaning and split completed"
    )
    return raw_train_df, raw_eval_df, raw_holdout_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load, clean, and split raw data")
    parser.add_argument("--input", type=str, default=str(RAW_DATA_PATH), help="Path to load cleaned data")
    parser.add_argument("--output", type=str, default=str(DATA_DIR), help="Path to save splitted cleaned data")

    args = parser.parse_args()
    load_and_split_raw_data(args.input, args.output)


