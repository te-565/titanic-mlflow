from loguru import logger
from sklearn.svm import SVC
import mlflow


def create_svc_model(svc_hyperparameters):
    """
    Description
    -----------
    Creates a Support Vector Classifier model for use in a scikit-learn pipeline
    based upon the input logreg_hyperparameters.

    The hyperparameters are also tracked as parameters in MLFlow.

    Parameters
    ---------
    svc_hyperparameters: dict
        The hyperparameters for the model

    Returns:
    --------
    model: sklearn.svm.SVC
        The scikit-learn SVC model

    model_name: str
        The name of the model

    cv: int
        The number of cross-validation folds to perform when evaluating the
        model

    Raises
    ------
    Exception: Exception
        Generic exception for logging

    Examples
    --------
    model, model_name, cv = create_logreg_model(
        logreg_hyperparameters=dict(
            model_name="svc",
            model_type="SVC",
            C="0.2",
            max_iter=-1,
            solver="lbfgs",
            n_jobs=-1,
            cv=5
        )
    )
    """

    logger.info("Running create_svc_model")

    try:
        # Create the model
        model = SVC(
            C=svc_hyperparameters["C"],
            kernel=svc_hyperparameters["kernel"],
            probability=svc_hyperparameters["probability"],
            max_iter=svc_hyperparameters["max_iter"],
        )

        # Log the parameters with MLFlow
        mlflow.log_param("model_name", svc_hyperparameters["model_name"])
        mlflow.log_param("model_type", svc_hyperparameters["model_type"])
        mlflow.log_param("C", svc_hyperparameters["C"])
        mlflow.log_param("kernel", svc_hyperparameters["kernel"])
        mlflow.log_param("probability", svc_hyperparameters["probability"])
        mlflow.log_param("max_iter", svc_hyperparameters["max_iter"])
        mlflow.log_param("cv", svc_hyperparameters["cv"])

        model_name = svc_hyperparameters["model_name"]
        cv = svc_hyperparameters["cv"]

        return model, model_name, cv

    except Exception:
        logger.exception("Error running create_svc_model()")
