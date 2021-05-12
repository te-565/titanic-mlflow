from loguru import logger
from sklearn.svm import SVC
import mlflow


def create_svc_model(svc_hyperparameters):
    """

    """

    logger.info("Running create_svc_model")

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
