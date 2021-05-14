import mlflow
import sklearn
from src.utils import load_config
from src.models import create_logreg_model

logreg_hyperparameters = dict(
    model_name="test_logreg",
    model_type="Logistic Regression",
    penalty="l2",
    C=1,
    max_iter=10,
    solver="lbfgs",
    n_jobs=-1,
    cv=2
)


def test_create_logreg_model():
    """Test for the create_logreg_model function"""

    config = load_config(".env.test")

    # Configure MLFlow
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["mlflow_experiment"])

    with mlflow.start_run():

        # Run the function
        model, model_name, cv = create_logreg_model(
            logreg_hyperparameters=logreg_hyperparameters
        )

        # Run the tests
        assert isinstance(
            model, sklearn.linear_model._logistic.LogisticRegression
        )
        assert model_name == "test_logreg"
        assert cv == 2

        mlflow.end_run()
