import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger()

def standardize_column_names(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df

def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    n_removed = before - len(df)
    logger.info(
        f"Removed {n_removed} duplicates"
    )
    return df

def drop_missing(df):
    return df.dropna()

def fill_missing(df):
    df = df.ffill()
    return df

def remove_invalid_values(df):
    if "power" in df.columns:
        df = df[(df["power"] >= 0) & (df["power"] <= 1)]
    if all(col in df.columns for col in ["windspeed_10m", "windspeed_100m"]):
        df = df[(df["windspeed_100m"] > 0) & (df["windspeed_10m"] > 0)]
    if "relativehumidity_2m" in df.columns:
        df = df[(df["relativehumidity_2m"] >= 0) & (df["relativehumidity_2m"] <= 100)]
    return df

def clean_data(df):

    logger.info("Starting cleaning pipeline")

    df = standardize_column_names(df)

    df = remove_duplicates(df)

    df = fill_missing(df)

    df = remove_invalid_values(df)

    logger.info("Cleaning complete")

    return df