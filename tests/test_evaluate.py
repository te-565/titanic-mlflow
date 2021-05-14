import os
import mlflow
import sklearn
from src.utils import (
    load_config,
    load_logger,
    load_parameters,
)
from src.ingest_split import ingest_split
from src.preprocessing_pipeline import create_preprocessing_pipeline
from src.models import create_logreg_model
from src.model_pipeline import evaluate_model


def test_evaluate():

    config = load_config(".env.test")
    logger = load_logger(
        app_name=config["app_name"],
        logs_path=config["logs_path"]
    )

    # Configure MLFlow
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["mlflow_experiment"])

    # Start MLFlow Tracking
    with mlflow.start_run():

        parameters = load_parameters(parameters_path=config["parameters_path"])

        # Ingest the data
        X_train, X_test, y_train, y_test, X_holdout = ingest_split(
            train_test_raw_path=config["train_test_raw_path"],
            holdout_raw_path=config["holdout_raw_path"],
            target=parameters["target"],
            ingest_split_parameters=parameters["ingest_split_parameters"]
        )

        # Create the preprocessing pipeline
        preprocessing_pipeline = create_preprocessing_pipeline(
            pipeline_parameters=parameters["pipeline_parameters"]
        )

        # Create a model with hyperparameters
        model, model_name, cv = create_logreg_model(
            logreg_hyperparameters=parameters["logreg_hyperparameters"]
        )

        # Run the function
        model = evaluate_model(
            preprocessing_pipeline=preprocessing_pipeline,
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            artifact_path=config["artifact_path"],
            cv=cv
        )

        assert isinstance(
            model, sklearn.linear_model._logistic.LogisticRegression
        )
