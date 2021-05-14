import os
import sys
from dotenv import load_dotenv
from loguru import logger
import yaml


def load_config(env_path):
    """
    Description
    -----------
    Loads the configuration variables stored in the environment via the .env
    file into the application as a dictionary.

    Parameters
    ----------
    env_path: str
        Location of the environment file to load.

    Returns
    -------
    config: dict
        Configuration data for the application

    Raises
    ------
    Exception:
        Raises & logs an exception if there's any errors in the function.

    Examples
    --------
    config = load_config(".env")

    """

    try:
        load_dotenv(env_path, override=True)
        config = dict(
            app_name=os.getenv("APP_NAME"),
            parameters_path=os.getenv("PARAMETERS_PATH"),
            artifact_path=os.getenv("ARTIFACT_PATH"),
            models_path=os.getenv("MODELS_PATH"),
            logs_path=os.getenv("LOGS_PATH"),
            mlflow_tracking_db=os.getenv("MLFLOW_TRACKING_DB"),
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
            mlflow_experiment=os.getenv("MLFLOW_EXPERIMENT"),
            train_test_raw_path=os.getenv("TRAIN_TEST_RAW_PATH"),
            holdout_raw_path=os.getenv("HOLDOUT_RAW_PATH")
        )

        return config

    except Exception as e:
        logger.exception("Error in load_config()")
        logger.exception(e)


def load_logger(
    app_name: str,
    logs_path: str
):
    """
    Description
    -----------
    Loads and initialises the logger for the application.

    Parameters
    ----------
    app_name: str
        The name of the application.

    logs_path: str
        Location to output logs to.

    Returns
    -------
    None

    Raises
    ------
    Exception:
        Raises & logs an exception if there's any errors in the function.

    Examples
    --------
    load_logger()
    """
    try:
        logger.add(
            sys.stderr,
            format="{time} {level} {message}",
            filter=app_name,
            level="INFO",
            colorize=True,
            backtrace=False,
            diagnose=False
        )
        logs_path = "{}/{}.log".format(logs_path, "{time}")
        logger.add(logs_path)

    except Exception as e:
        logger.exception("Error in load_logger()")
        logger.exception(e)


def load_parameters(parameters_path):
    """
    Description
    -----------
    Parse a parameters.yaml file into a Python data structure.

    Parameters
    ----------
    parameters_path: str
        The location of the parameters.yaml file to parse.

    Returns
    -------
    parameters: dict / list
        Python data structure containing the parsed .yaml file.

    Raises
    ------
    Exception:
        Raises & logs an exception if there's any errors in the function.

    Examples
    -------
    parameters = parse_yaml(location = "./location/of/file.yaml")
    """

    with open(parameters_path, 'r') as stream:

        try:

            # Try to parse the .yaml file
            parameters = yaml.safe_load(stream)
            return parameters

        except Exception:

            # Raise exception if the .yaml file can't be parsed
            logger.exception("Error in load_parameters()")
