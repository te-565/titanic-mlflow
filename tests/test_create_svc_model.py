import mlflow
import sklearn
from src.utils import load_config
from src.models import create_svc_model

svc_hyperparameters = dict(
    model_name="test_svc",
    model_type="SVC",
    C=1,
    kernel="linear",
    probability=True,
    max_iter=10,
    cv=2
)


def test_create_svc_model():
    """Test for the create_svc_model function"""

    config = load_config(".env.test")

    # Configure MLFlow
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["mlflow_experiment"])

    with mlflow.start_run():

        # Run the function
        model, model_name, cv = create_svc_model(
            svc_hyperparameters=svc_hyperparameters
        )

        # Run the tests
        assert isinstance(model, sklearn.svm._classes.SVC)
        assert model_name == "test_svc"
        assert cv == 2

        mlflow.end_run()
