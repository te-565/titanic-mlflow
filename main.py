import plac
import mlflow
from src import (
    load_config,
    load_logger,
    load_parameters,
    ingest_split,
    create_preprocessing_pipeline,
    create_logreg_model,
    create_svc_model,
    evaluate_model,
    create_model_pipeline
)


@plac.flg(arg="deploy", help="Stage a model for deployment", abbrev="dep")
def run(deploy: bool = False):
    """Run the end-to-end pipeline"""

    # Load config, logger & parameters
    config = load_config(".env.dev")
    logger = load_logger(
        app_name=config["app_name"],
        logs_path=config["logs_path"]
    )
    parameters = load_parameters(parameters_path=config["parameters_path"])

    # Configure MLFlow
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["mlflow_experiment"])

    # Start MLFlow Tracking
    with mlflow.start_run():

        # Configure MLFlow
        mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
        mlflow.set_experiment(config["mlflow_experiment"])

        # Ingest & split the data
        X_train, X_test, y_train, y_test, X_holdout = ingest_split(
            train_test_raw_path=config["train_test_raw_path"],
            holdout_raw_path=config["holdout_raw_path"],
            target=parameters["target"],
            ingest_split_parameters=parameters["ingest_split_parameters"]
        )

        # Create the preprocessing pipeline
        preprocessing_pipeline = create_preprocessing_pipeline(
            pipeline_parameters=parameters["pipeline_parameters"],
            features_path=config["train_features_path"]
        )

        # Create a model with hyperparameters
        model, model_name, cv = create_logreg_model(
            logreg_hyperparameters=parameters["logreg_hyperparameters"]
        )

        # Evaluate the model
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

        if deploy:
            # Create the end-to-end model pipeline
            model_pipeline, model = create_model_pipeline(
                preprocessing_pipeline=preprocessing_pipeline,
                model=model,
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                artifact_path=config["artifact_path"],
                models_path=f"{config['models_path']}/{model_name}/"
            )

        mlflow.end_run()


if __name__ == "__main__":
    plac.call(run)
