from src.utils.logger import setup_logger

logger = setup_logger()
#from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import mlflow

def evaluate_model(
        eval_path: Path | str ,
        model_path: Path | str ,
        target_col: str
        ):
    eval_df = pd.read_csv(eval_path)
    X_eval, y_eval = eval_df.drop(columns= [target_col]), eval_df[target_col]
    # predict model
    model = joblib.load(model_path)
    y_pred = model.predict(X_eval)
    # evalaute model
    mae = float(mean_absolute_error(y_eval, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
    r2 = float(r2_score(y_eval, y_pred))
    metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    logger.info(
        f"Evaluation completed: mae={mae:.2f}, rmse={rmse:.2f}, r2={r2:.2f}"
    )
    return metrics