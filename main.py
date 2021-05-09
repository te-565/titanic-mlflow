import mlflow
from src import (
    load_config,
    load_logger,
    load_parameters,
    ingest_split,
    create_pipeline
)

# Load config, logger & parameters
config = load_config(".env_dev")
logger = load_logger(app_name=config["app_name"], logs_path=config["logs_path"])
parameters = load_parameters(parameters_path=config["parameters_path"])

# Configure MLFlow
mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
mlflow.set_experiment(config["mlflow_experiment"])

# Start MLFlow Tracking
with mlflow.start_run():

    # Ingest & split the data
    X_train, X_test, y_train, y_test, X_holdout = ingest_split(
        train_raw_path=config["train_raw_path"],
        holdout_raw_path=config["holdout_raw_path"],
        target=parameters["target"],
        ingest_split_parameters=parameters["ingest_split_parameters"]
    )

    pipeline = create_pipeline(parameters["pipeline_parameters"])

