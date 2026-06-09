from src.utils.logger import setup_logger

logger = setup_logger()
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from src.utils.config_loader import (
    load_config
)




def get_model(model_name, path):
    config = load_config(path)

    logger.info(
        f"Creating model: {model_name}"
    )

    if model_name == "xgboost":

        return XGBRegressor(
            **config["parameters"],
            #early_stopping_rounds=50,
            eval_metric="rmse",
        )

    elif model_name == "lightgbm":

        return LGBMRegressor(
            **config["parameters"]
        )

    else:

        raise ValueError(
            f"Unsupported model: {model_name}"
        )