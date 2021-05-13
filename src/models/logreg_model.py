from loguru import logger
from sklearn.linear_model import LogisticRegression
import mlflow


def create_logreg_model(logreg_hyperparameters: dict):
    """
    Description
    -----------
    Creates a Logistic Regression model for use in a scikit-learn pipeline
    based upon the input logreg_hyperparameters.

    The hyperparameters are also tracked as parameters in MLFlow.

    Parameters
    ---------
    logreg_hyperparameters: dict
        The hyperparameters for the model

    Returns:
    --------
    model: sklearn.linear_model.LogisticRegression
        The scikit-learn Logistic Regression model

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
        svc_hyperparameters=dict(
            model_name="svc",
            model_type="SVC",
            c=0.5,
            kernel="linear",
            probability=True,
            max_iter=-1,
            cv=5
        )
    )
    """

    logger.info("Running create_logreg_model()")

    try:
        # Create the model
        model = LogisticRegression(
            penalty=logreg_hyperparameters["penalty"],
            C=logreg_hyperparameters["C"],
            solver=logreg_hyperparameters["solver"],
            max_iter=logreg_hyperparameters["max_iter"],
            n_jobs=logreg_hyperparameters["n_jobs"]
        )

        # Log the parameters with MLFlow
        mlflow.log_param("model_name", logreg_hyperparameters["model_name"])
        mlflow.log_param("model_type", logreg_hyperparameters["model_type"])
        mlflow.log_param("penalty", logreg_hyperparameters["penalty"])
        mlflow.log_param("C", logreg_hyperparameters["C"])
        mlflow.log_param("solver", logreg_hyperparameters["solver"])
        mlflow.log_param("max_iter", logreg_hyperparameters["max_iter"])
        mlflow.log_param("n_jobs", logreg_hyperparameters["n_jobs"])
        mlflow.log_param("cv", logreg_hyperparameters["cv"])

        model_name = logreg_hyperparameters["model_name"]
        cv = logreg_hyperparameters["cv"]

        return model, model_name, cv

    except Exception:
        logger.exception("Error running create_logreg_model()")
