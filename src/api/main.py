# Goal: Create a FastAPI app to serve the trained ML model into a web service that any system can call over a HTTP
from fastapi import FastAPI           # Web framework for APIs
from pathlib import Path              # For handling file paths cleanly
from typing import List, Dict, Any    # For type hints (clarity in endpoints)
import pandas as pd                   # To handle incoming JSON as DataFrames
import boto3, os                      # AWS SDK for Python + env variables

from src.utils.logger import setup_logger

logger = setup_logger()

# Import inference pipeline
from src.inference.inference import predict

# -------------
# Configuration
# -------------

S3_BUCKET = os.getenv("S3_BUCKET", "wind-power-forecasting-data")
REGION = os.getenv("REGION", "eu-central-1")
s3 = boto3.client("s3", region_name=REGION)

# Avoid re-downloading the model/data every time the app starts
def load_from_s3(key, local_path):
    """
    Download files from S3 if not already cached locally
    """
    local_path = Path(local_path)
    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
        logger.info(
            f"Downloading {key} from S3"
        )
        s3.download_file(S3_BUCKET, key, str(local_path))
    return str(local_path)

# ----------------
# Paths
# ----------------

# Download model + training features from S3 if not cached
MODEL_PATH = Path(load_from_s3("models/best_xgb_model.pkl", "models/best_xgb_model.pkl"))
TRAIN_FE_PATH = Path(load_from_s3("processed/train.csv", "processed/train.csv"))

# Load expected training features for alignment
if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    TRAIN_FEATURE_COLUMNS = [c for c in _train_cols.columns if c != "power"]
else:
    TRAIN_FEATURE_COLUMNS = None

# ----------------
# App
# ---------------

# Instantiate the FastAPI app
app = FastAPI(title="Wind Power Forecasting API")

# / -> Simple landing endpoint to confirm API is alive
@app.get("/")
def root():
    return {"message": "Wind Power Forecasting API is running"}

# /Health -> Check if model exists, return status info (like expected feature count)
@app.get("/health")
def health():
    status: Dict[str, Any] = {"model_path": str(MODEL_PATH)} 
    if not MODEL_PATH.exists():
        status["status"] = "unhealthy"
        status["error"] = "Model not found"
    else:
        status["status"] = "healthy"
        if TRAIN_FEATURE_COLUMNS:
            status["n_features_expected"] = len(TRAIN_FEATURE_COLUMNS)
    return status

# Prediction endpoint: Core ML Serving Endpoint
@app.post("/predict")
def predict_batch(data: List[dict]):
    if not MODEL_PATH.exists():
        return {"error": f"Model not found at {str(MODEL_PATH)}"}
    
    df = pd.DataFrame(data)
    if df.empty:
        return {"error": "No data provided"}
    preds_df = predict(df, apply_feature_engineering =True, model_path = MODEL_PATH)
    response = {"predictions": preds_df["predicted_power"].astype(float).tolist()}
    if "actual_power" in preds_df.columns:
        response["actuals"] = preds_df["actual_power"].astype(float).tolist()
    return response

# Return a preview of the most recent batch prediction
@app.get("/latest_predictions")
def latest_predictions(limit: int = 5):
    pred_dir = Path("data/predictions")
    files = sorted(pred_dir.glob("preds_*.csv"))
    if not files:
        return {"error": "No predictions found"}
    
    latest_file = files[-1]
    df = pd.read_csv(latest_file)
    return {
        "file": latest_file.name,
        "rows": int(len(df)),
        "preview": df.head(limit).to_dict(orient="records")
    }
