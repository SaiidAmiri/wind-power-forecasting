import argparse
import pandas as pd
import joblib
from pathlib import Path
from src.etl.clean_data import clean_data
from src.etl.validate_data import run_validations
from src.etl.feature_engineering import feature_engineering
from src.utils.logger import setup_logger

logger = setup_logger()

# ---------------
# Default paths and Configuration
# ---------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = PROJECT_ROOT / "models" / "best_xgb_model.pkl"
TRAIN_FE_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "predictions.csv"
COL = ""

# Load training feature columns (strict schema from training dataset)
if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    TRAIN_FEATURE_COLUMNS = [c for c in _train_cols.columns if c != "power"] # excluding power column
else:
    TRAIN_FEATURE_COLUMNS = None

# --------------
# Core inference function
# --------------

def predict(
        df: pd.DataFrame,
        apply_feature_engineering: bool,
        model_path: Path | str = DEFAULT_MODEL
        ) -> pd.DataFrame:
    
    logger.info(
        f"Columns received by predict(): {df.columns.tolist()}"
    )
    
    if apply_feature_engineering:
        # Step 1: Preprocess raw data
        df = clean_data(df)
        run_validations(df)

        # Step 2: Feature engineering
        df = feature_engineering(df)

    # Step 3: Separate actuals if present
    y_true = None
    if "power" in df.columns:
        y_true = df["power"].tolist()

    # Step 4: Align columns with training schema
    if TRAIN_FEATURE_COLUMNS is not None:
        df = df.reindex(columns=TRAIN_FEATURE_COLUMNS, fill_value=0)
    
    # Step 5: Load model and predict
    model = joblib.load(model_path)
    preds = model.predict(df)

    # Step 6: Build output
    output_df = df.copy()
    output_df["predicted_power"] = preds
    if y_true is not None:
        output_df["actual_power"] = y_true

    return output_df

# --------------
# Main: CLI entrypoint: Running inference directly from terminal
# --------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on new wind power data (raw)")
    parser.add_argument("--input", type=str, required=True, help="Path to input RAW CSV file")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Path to save predictions")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to trained model")

    args = parser.parse_args()

    raw_df = pd.read_csv(args.input)
    preds_df = predict(
        raw_df,
        model_path=args.model
    )
    preds_df.to_csv(args.output, index=False)
    logger.info(
        f"Predictions saved to {args.output}"
    )