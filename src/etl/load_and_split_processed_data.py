import pandas as pd
import numpy as np
from pathlib import Path
from src.etl.feature_engineering import feature_engineering
from src.etl.clean_data import clean_data
from src.etl.validate_data import run_validations
from src.utils.logger import setup_logger

logger = setup_logger()

DATA_DIR = Path('data/processed')

def split_data(df):
    n = len(df)
    cutoff_eval = int(n *  0.70)
    cutoff_holdout = int(n * 0.85)
    train_df = df.iloc[:cutoff_eval]
    eval_df = df.iloc[cutoff_eval: cutoff_holdout]
    holdout_df = df.iloc[cutoff_holdout:]
    return train_df, eval_df, holdout_df

def load_and_split_processed_data(
        raw_path: str = "data/raw/Location1.csv",
        output_dir: Path | str = DATA_DIR
):
    """ Load raw dataset, clean it, validate it, perform feature
    engineering, and split data into train, eval, and holdout,
    and save to output_dir"""

    logger.info(
        "Data Split started"
    )
    
    # load raw dataset
    df = pd.read_csv(raw_path)
    # clean dataset
    df = clean_data(df)
    # validate dataset
    run_validations(df)
    # feature engineering
    col = 'windspeed_100m'
    df = feature_engineering(df, col)
    # splits
    train_df, eval_df, holdout_df = split_data(df)
    # save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    eval_df.to_csv(output_dir / "eval.csv", index=False)
    holdout_df.to_csv(output_dir / "holdout.csv", index=False)
    logger.info(
        "Data Split completed"
    )
    return train_df, eval_df, holdout_df


