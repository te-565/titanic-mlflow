from loguru import logger
from copy import copy
import sklearn
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


def create_model_pipeline(
    preprocessing_pipeline: sklearn.pipeline.Pipeline,
    model: sklearn,
    model_name: str,
    X_train: pd.core.frame.DataFrame,
    artifact_path: str,
    models_path: str
):
    """
    Description
    -----------
    Create a model pipeline by appending the supplied model to the supplied
    preprocessing_pipeline and logging nd saving the pipeline to enable
    deployment.

    Also create the signature (the input and output format of the data) via
    the MLFlow infer_signature funciton.

    Parameters
    ---------
    preprocessing_pipeline: sklearn.pipeline.Pipeline
        The scikit-learn pipeline used to preprocess the data

    model: sklearn
        The model to evaluate.

    model_name: str
        The name of the model.

    X_train: pd.core.frame.DataFrame
        The dataframe of features for training the model.

    y_train: pd.core.frame.DataFrame
        The dataframe containing the target for training the model.

    artifact_path: str
        The location of where to save MLFlow artifacts.

    models_path: str
        The location of where to save the MLFLow model pipeline.


    Returns:
    --------
    model_pipeline: sklearn.pipeline.Pipeline
        The overall end-to-end pipeline containing the preprocessing and model

    model: sklearn
        The fitted model.

    Raises
    ------
    Exception: Exception
        Generic exception for logging

    Examples
    --------
    model_pipeline, model = create_model_pipeline(
        preprocessing_pipeline=preprocessing_pipeline,
        model=logreg_model,
        model_name="model_name",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        artifact_path="path/to/folder",
        cv=5
    )
    """
    logger.info("Creating end-to-end model pipeline")

    try:
        # Append the preprocessing & model pipelines
        model_pipeline = copy(preprocessing_pipeline)
        model_pipeline.steps.append(
            ["Model", model]
        )

        # Infer the signature for the model
        signature = infer_signature(
            model_input=X_train,
            model_output=model_pipeline.predict(X_train)
        )

        # Log the model as an artifact
        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            artifact_path=artifact_path,
            conda_env="./environment.yaml",
            registered_model_name=model_name,
            signature=signature
        )

        # Save the model for serving
        mlflow.sklearn.save_model(
            sk_model=model_pipeline,
            path=models_path,
            conda_env="./environment.yaml",
            signature=signature
        )

        return model_pipeline, model

    except Exception:
        logger.exception("Error running create_model_pipeline()")
