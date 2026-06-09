import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger()

REQUIRED_COLUMNS = [
    #"time",
    "temperature_2m",
    "relativehumidity_2m",
    "dewpoint_2m",
    "windspeed_10m",
    "windspeed_100m",
    "winddirection_10m",
    "winddirection_100m",
    "windgusts_10m",
    "power"
]


def validate_schema(df):

    missing_cols = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_cols:

        logger.error(
            f"Missing columns detected: {missing_cols}"
        )

        raise ValueError(
            f"Missing columns: {missing_cols}"
        )

    logger.info("Schema validation passed")


def validate_missing_values(df):

    nulls = df.isnull().sum()

    if nulls.any():

        logger.warning(
            f"Missing values detected:\n{nulls}"
        )

    else:
        logger.info(
            "No missing values detected"
        )


def validate_ranges(df):
    
    if ((df["windspeed_10m"] < 0) & (df["windspeed_100m"] < 0) & (df["power"] < 0)).any():

        logger.error(
            "Negative wind speed values detected"
        )

        raise ValueError(
            "Invalid wind speed values"
        )

    logger.info("Range validation passed")


def validate_duplicates(df):

    duplicates = df.duplicated().sum()

    if duplicates > 0:

        logger.warning(
            f"Duplicate rows detected: {duplicates}"
        )

    else:
        logger.info("No duplicate rows detected")


def run_validations(df):

    logger.info(
        "Starting data validation pipeline"
    )

    validate_schema(df)

    validate_missing_values(df)

    validate_ranges(df)

    validate_duplicates(df)

    logger.info(
        "All validations completed successfully"
    )