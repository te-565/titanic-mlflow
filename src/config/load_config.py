import os
from dotenv import load_dotenv
from loguru import logger


def load_config():
    """
    Description
    -----------
    Loads the configuration variables stored in the environment via the .env
    file into the application as a dictionary.

    Parameters
    ----------
    None

    Returns
    -------
    config: dict
        Configuration data for the application

    Raises
    ------
    Ex

    Examples
    --------

    config = load_config()

    """

    try:

        # Load in Environment Variables
        load_dotenv()

        config = {
            "app_name": os.getenv("APP_NAME"),
            "parameters_path": os.getenv("PARAMETERS_PATH"),
            "artifacts_path": os.getenv("ARTIFACTS_PATH"),
            "logs_path": os.getenv("LOGS_PATH"),
            "mlflow_experiment": os.getenv("MLFLOW_EXPERIMENT"),
            "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI")
        }

    except Exception as e:
        logger.exception(f"Error loading config via load_config() \n {e}")

    return config
