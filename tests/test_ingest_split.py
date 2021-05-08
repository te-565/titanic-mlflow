from src import (
    load_config,
    load_parameters,
    ingest_split
)


def test_ingest_split():
    """Test the ingest_split_function"""

    # Load in the test configuration & parameters
    config = load_config(".env-test")
    parameters = load_parameters(parameters_path=config["parameters_path"])

    # Run the function
    X_train, X_test, y_train, y_test, X_holdout = ingest_split(
        train_raw_path=config["train_raw_path"],
        holdout_raw_path=config["holdout_raw_path"],
        uid=parameters["uid"],
        target=parameters["target"],
        ingest_split_parameters=parameters["ingest_split_parameters"]
    )

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
