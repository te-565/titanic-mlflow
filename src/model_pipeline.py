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
    y_train: pd.core.frame.DataFrame,
    artifact_path: str
):
    """

    """
    logger.info("Creating end-to-end model pipeline")

    # Append the preprocessing & model pipelines
    model_pipeline = copy(preprocessing_pipeline)
    model_pipeline.steps.append(
        ["Model", model]
    )

    # Fit the model
    model_pipeline.fit(X=X_train, y=y_train.values.ravel())

    # Infer the signature for the model
    signature = infer_signature(
        model_input=X_train,
        model_output=model_pipeline.predict(X_train)
    )

    # Log the model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path,
        conda_env="./environment.yaml",
        registered_model_name=model_name,
        signature=signature
    )

    return model_pipeline, model
