import boto3
from botocore.exceptions import ClientError
from pathlib import Path
from src.utils.logger import setup_logger

logger = setup_logger()

# Configuration

bucket = "wind-power-forecasting-data"
region = "eu-central-1"

# set project root as parent
local_processed_data_dir = Path("data/processed")
local_raw_data_dir = Path("data/raw")
local_model_dir = Path("models")

s3 = boto3.client("s3", region_name = region)

# helper functions

def s3_file_exists(bucket: str, s3_key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=s3_key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise

def upload_file(local_path: Path, s3_key: str):
    if not local_path.exists():
        logger.info(
            f"File not found: {local_path}"
        )
        return
    if s3_file_exists(bucket, s3_key):
        logger.info(f"s3://{bucket}/{s3_key} already exists. Skipping upload.")
        return
    logger.info(
        f"Uploading {local_path} to s3://{bucket}/{s3_key}"
    )
    s3.upload_file(str(local_path), bucket, s3_key)


if __name__ == "__main__":
    # upload required processed datasets
    upload_file(local_processed_data_dir / "holdout.csv", "processed/holdout.csv")
    upload_file(local_processed_data_dir / "train.csv", "processed/train.csv")
    # upload required processed datasets
    upload_file(local_raw_data_dir / "cleaned_holdout.csv", "raw/cleaned_holdout.csv")
    upload_file(local_raw_data_dir / "cleaned_train.csv", "raw/cleaned_train.csv")
    # upload model
    upload_file(local_model_dir / "best_xgb_model.pkl", "models/best_xgb_model.pkl")



