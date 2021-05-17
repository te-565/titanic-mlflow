import os
import glob
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
from src.model_pipeline import create_model_pipeline


def test_create_model_pipeline():
    """Test for the create_model_pipeline"""    

    config = load_config(".env.test")
    logger = load_logger(
        app_name=config["app_name"],
        logs_path=config["logs_path"]
    )

    # Clear the dummy directory
    folders_to_clean = [
        config["artifact_path"],
        config["models_path"]
    ]

    for folder in folders_to_clean:
        files = glob.glob(folder)
        for f in files:
            try:
                # If it's a file delete it
                os.remove(f)
            except:
                # If it's a folder, pass
                pass

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

        # Fit the model via evaluate
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

        # Run the function
        model_pipeline, model = create_model_pipeline(
            preprocessing_pipeline=preprocessing_pipeline,
            model=model,
            model_name=model_name,
            X_train=X_train,
            artifact_path=config["artifact_path"],
            models_path=f"{config['models_path']}/{model_name}"
        )

        assert isinstance(model_pipeline, sklearn.pipeline.Pipeline)
        assert isinstance(
            model, sklearn.linear_model._logistic.LogisticRegression
        )