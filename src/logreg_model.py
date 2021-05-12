from loguru import logger
from sklearn.linear_model import LogisticRegression
import mlflow


def create_logreg_model(logreg_hyperparameters):
    """

    """

    logger.info("Running create_logreg_model")

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
