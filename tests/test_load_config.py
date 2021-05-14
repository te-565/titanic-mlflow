import os
from src.utils.utils import load_config


def test_load_config():
    """Test the load_config function"""

    config = load_config(".env.test")

    assert config["app_name"] == os.getenv("APP_NAME")
    assert config["parameters_path"] == os.getenv("PARAMETERS_PATH")
    assert config["artifact_path"] == os.getenv("ARTIFACT_PATH")
    assert config["models_path"] == os.getenv("MODELS_PATH")
    assert config["logs_path"] == os.getenv("LOGS_PATH")
    assert config["mlflow_tracking_db"] == os.getenv("MLFLOW_TRACKING_DB")
    assert config["mlflow_tracking_uri"] == os.getenv("MLFLOW_TRACKING_URI")
    assert config["mlflow_experiment"] == os.getenv("MLFLOW_EXPERIMENT")
    assert config["train_test_raw_path"] == os.getenv("TRAIN_TEST_RAW_PATH")
    assert config["holdout_raw_path"] == os.getenv("HOLDOUT_RAW_PATH")
