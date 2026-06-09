import math
from pathlib import Path
from src.training.train_model import train_model
from src.training.evaluate_model import evaluate_model
from src.training.tune_model import tune_model
from src.utils.logger import setup_logger

logger = setup_logger()
import joblib

# Assume that feature engineering is already done and the csv files are saved 
TRAIN_PATH = "data/processed/train.csv" 
EVAL_PATH = "data/processed/eval.csv"

# Ensuring that the keys in the metrics dict are the same
def _assert_metrics(m):
    assert set(m.keys()) == {"mae", "rmse", "r2"}
    assert all(isinstance(v, float) and math.isfinite(v) for v in m.values())

# Training: train a quick model (with a small sample and few params to keep tests fast)
def test_train_create_model_and_metrics(tmp_path):
    out_path = tmp_path / "xgb_model.pkl"
    model_name = "xgboost"
    params_path = "src/configs/xgboost_config_test.yaml"
    target_col = "power"
    # small params
    train_model(
        train_path = TRAIN_PATH,
        eval_path = EVAL_PATH,
        model_output = out_path,
        model_name = model_name,
        params_path = params_path,
        target_col = target_col
    )

    assert out_path.exists()
    model = joblib.load(out_path)
    assert model is not None
    logger.info(
        "train_model test passed"
        )

# Evaluating: train a model first then evaluate it on eval data
def test_eval_works_with_saved_model(tmp_path):
    # train quick model
    out_path = tmp_path / "xgb_model.pkl"
    model_name = "xgboost"
    params_path = "src/configs/xgboost_config_test.yaml"
    target_col = "power"
    # small params
    train_model(
        train_path = TRAIN_PATH,
        eval_path = EVAL_PATH,
        model_output = out_path,
        model_name = model_name,
        params_path = params_path,
        target_col = target_col
    )
    metrics = evaluate_model(eval_path = EVAL_PATH,
                             model_path = out_path,
                             target_col = target_col)
    _assert_metrics(metrics)
    logger.info(
        "evaluate_model test passed"
        )
    
# Tuning: tune model with only few trials (fast for CI)
def test_tune_saves_best_model(tmp_path):
    model_out = tmp_path / "xgb_best.pkl"
    model_name = "xgboost"
    target_col = "power"
    tracking_dir = tmp_path / "mlruns"
    best_params, best_metrics = tune_model(
        train_path = TRAIN_PATH,
        eval_path = EVAL_PATH,
        model_output = model_out,
        model_name = model_name,
        target_col = target_col,
        n_trials = 2,
        tracking_uri = str(tracking_dir) ,
        experiment_name = "test_xgb_optuna"
    )
    assert model_out.exists()
    assert isinstance(best_params, dict) and best_params
    _assert_metrics(best_metrics)
    logger.info(
        "tune_model test passed"
        )

    
