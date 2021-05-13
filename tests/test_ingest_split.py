import mlflow
from src.utils import (
    load_config,
    load_parameters,
)
from src.ingest_split import ingest_split


def test_ingest_split():
    """Test the ingest_split_function"""

    # Load in the test configuration & parameters
    config = load_config(".env-test")
    parameters = load_parameters(parameters_path=config["parameters_path"])

    # Configure MLFlow
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["mlflow_experiment"])

    # Start MLFlow Tracking
    with mlflow.start_run():

        # Run the function
        X_train, X_test, y_train, y_test, X_holdout = ingest_split(
            train_test_raw_path=config["train_test_raw_path"],
            holdout_raw_path=config["holdout_raw_path"],
            target=parameters["target"],
            ingest_split_parameters=parameters["ingest_split_parameters"]
        )
        mlflow.end_run()

    # Run the tests
    for df in [X_train, X_test, X_holdout]:
        df.columns.tolist() == [
            'PassengerId',
            'Pclass',
            'Name',
            'Sex',
            'Age',
            'SibSp',
            'Parch',
            'Ticket',
            'Fare',
            'Cabin',
            'Embarked'
        ]

    for df in [y_train, y_test]:
        df.columns.tolist() == ["Survived"]
