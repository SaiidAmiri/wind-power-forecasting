from src.utils.logger import setup_logger

logger = setup_logger()
#from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import mlflow
from typing import Dict, Optional, Tuple
from src.training.model_factory import get_model



def train_model(
        train_path: Path | str ,
        eval_path: Path | str ,
        model_output: Path | str ,
        model_name: str ,
        params_path : Path | str,
        target_col: str
):
    """
    Train baseline model XGB and save it
    
    Returns
    ---------
    model: XGBRegressor
    metrics: dict[str, float]
    """
    # load training data
    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)
    # split data
    #target = "power"
    X_train, y_train = train_df.drop(columns = [target_col]), train_df[target_col]
    X_eval, y_eval = eval_df.drop(columns = [target_col]), eval_df[target_col]
    # train model
    model = get_model(model_name, params_path)
    model.fit(X_train, y_train)
    # save model
    output_path = Path(model_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logger.info(
        f"Model trained and saved to {output_path}"
    )
    return model

def load_training_data(path):

    logger.info(
        "Loading training dataset"
    )

    df = pd.read_parquet(path)

    return df

def split_data(df, target_col):
    n = len(df)
    train_size = int(n *  0.70)
    valid_test_size = int(n * 0.85)
    train = df.iloc[:train_size]
    valid = df.iloc[train_size: valid_test_size]
    test = df.iloc[valid_test_size:]
    X_train, y_train = train.drop(columns=[target_col]),       train[target_col]
    X_val,   y_val   = valid.drop(columns=[target_col]),       valid[target_col]
    X_test,  y_test  = test.drop(columns=[target_col]),       test[target_col]
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_ml_model(model, X_train, y_train):

    logger.info("Training model")

    model.fit(X_train, y_train)

    return model

def log_experiment(model_name, params):

    mlflow.log_params(params)

    mlflow.set_tag(
        "model_type",
        model_name
    )