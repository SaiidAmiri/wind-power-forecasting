from src.utils.logger import setup_logger

logger = setup_logger()
#from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
from xgboost import XGBRegressor
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from typing import Dict, Optional, Tuple

def load_data(
        train_path: Path | str ,
        eval_path: Path | str ,
        target_col: str
        ):
    # load training data
    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)
    # split data
    X_train, y_train = train_df.drop(columns = [target_col]), train_df[target_col]
    X_eval, y_eval = eval_df.drop(columns = [target_col]), eval_df[target_col]
    return X_train, y_train, X_eval, y_eval

def tune_model(
        train_path: Path | str ,
        eval_path: Path | str ,
        model_output: Path | str ,
        model_name: str ,
        target_col: str ,
        n_trials: int,
        tracking_uri: Optional[str] ,
        experiment_name: str
        ) -> Tuple[dict, dict]:
    """
    Run optuna tuning, save best model, 
    and return (best_params, best_metrics)
    """
    if tracking_uri:
        tracking_path = Path(tracking_uri)
        tracking_path.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_path.resolve().as_uri())
    
    mlflow.set_experiment(experiment_name)
    X_train, y_train, X_eval, y_eval = load_data(train_path, eval_path, target_col)
    
    def objective(trial: optuna.Trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist"
        }

        with mlflow.start_run(nested=True):
            if model_name == "xgboost":
                model = XGBRegressor(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_eval)
                
                mae = float(mean_absolute_error(y_eval, y_pred))
                rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
                r2 = float(r2_score(y_eval, y_pred))
                mlflow.log_params(params)
                mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})
                # Log final model
                # mlflow.xgboost.log_model(model, name="xgboost_model")
        return rmse
    
    study = optuna.create_study(direction = "minimize")
    study.optimize(objective, n_trials= n_trials)
    best_params = study.best_trial.params
    logger.info(
        f"Best parameters from optuna determined: {best_params}"
    )

    # retrain the model
    if model_name == "xgboost":
        best_model = XGBRegressor(**{**best_params, "random_state": 42, "n_jobs": -1, "tree_method": "hist" })
        best_model.fit(X_train, y_train)
        # metrics
        y_pred = best_model.predict(X_eval)
                
        mae = float(mean_absolute_error(y_eval, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
        r2 = float(r2_score(y_eval, y_pred))
        best_metrics = {"mae": mae, "rmse": rmse, "r2": r2}
        logger.info(
            f"Best tuning model metrics: mae={mae:.2f}, rmse={rmse:.2f}, r2={r2:.2f}"
            )
        # save best model
        output_path = Path(model_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, output_path)
        logger.info(
            f"Model trained and saved to {output_path}"
            )
        with mlflow.start_run(run_name="best_xgb_model"):
            mlflow.log_params(best_params)
            mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})
            # Log final model
            mlflow.xgboost.log_model(best_model, 
                                     name="best_xgboost_model",
                                     registered_model_name="best_xgb_model"
                                     )
  

    return best_params, best_metrics
