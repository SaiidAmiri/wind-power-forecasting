import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger()

def create_time_features(df):
    logger.info(
        "Creating time-based features"
    )
    df['time'] = pd.to_datetime(df['time'])
    #df = df.sort_values('time').reset_index(drop=True)
    #df["time"] = pd.to_datetime(df["time"])
    #df = df.sort_values("time").set_index("time")
    df['year'] = df['time'].dt.year
    df['hour'] = df['time'].dt.hour
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    #df['year'] = df.index.year
    #df['month'] = df.index.month
    #df['day'] = df.index.day
    #df['hour'] = df.index.hour
    #df = df.drop(columns=["time"])

    return df

def create_cyclical_features(df):

    logger.info(
        "Creating cyclical features"
    )
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
    return df

def create_interaction_features(df):

    logger.info(
        "Creating weather interaction features"
    )
    df['windspeed_cubed'] = df['windspeed_100m'] ** 3 

    df['gust_factor'] = df['windgusts_10m'] / (df['windspeed_100m'] + 0.001) # 0.001 prevents division by zero

    df['temp_humidity_interaction'] = (
        df['temperature_2m'] * 
        df['relativehumidity_2m']
    )
    
    df["wind_temp_interaction"] = (
        df["windspeed_100m"] *
        df["temperature_2m"]
    )
    return df

def create_lag_features(df, col):

    logger.info(
        "Creating lag features"
    )

    lags = [1, 2]

    for lag in lags:

        df[f"{col}_lag_{lag}"] = (
            df[col].shift(lag)
        )

    return df

def create_rolling_features(df, col):

    logger.info(
        "Creating rolling window features"
    )

    windows = [3, 6]

    for window in windows:

        df[
            f"{col}_rolling_mean_{window}"
        ] = (
            df[col]
            .rolling(window)
            .mean()
        )

        df[
            f"{col}_rolling_std_{window}"
        ] = (
            df[col]
            .rolling(window)
            .std()
        )

    return df



def feature_engineering(df, col = 'windspeed_100m'):

    logger.info(f"Columns entering FE: {list(df.columns)}")

    logger.info(
        "Starting feature engineering pipeline"
    )

    df = create_time_features(df)

    df = create_cyclical_features(df)

    df = create_interaction_features(df)

    df = create_lag_features(
        df,
        col
    )

    df = create_rolling_features(
        df,
        col
    )

    logger.info(
        "Feature engineering complete"
    )

    return df