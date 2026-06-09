import sys
import os
from pathlib import Path
from src.utils.logger import setup_logger

logger = setup_logger()

import pandas as pd
import pytest
from src.inference.inference import predict
# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

@pytest.fixture(scope="session")
def sample_df():
    """
    Load a small sample for inference testing
    """
    sample_path = ROOT / "data/processed/eval.csv"
    df = pd.read_csv(sample_path).sample(5, random_state=42).reset_index(drop=True)
    return df

def test_inference_runs_and_returns_predictions(sample_df):
    """
    Ensure inference pipeline runs and returns predicted_price column
    """
    preds_df = predict(sample_df, apply_feature_engineering=False)

    # Check output is not empty
    assert not preds_df.empty

    # Must include prediction columns
    assert "predicted_power" in preds_df.columns

    # Predictions should be numeric
    assert pd.api.types.is_numeric_dtype(preds_df["predicted_power"])

    logger.info(
        "Inference pipeline test passed"
    )
    print(f"Predictions: {preds_df["predicted_power"].head()}")