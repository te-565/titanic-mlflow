from src import (
    load_config,
    load_parameters,
    ingest_split,
    impute_missing_values
)

def test_impute_missing_values():
    """Test the create_family_size function"""

    # Load in the test configuration & parameters
    config = load_config(".env-test")
    parameters = load_parameters(parameters_path=config["parameters_path"])

    # Unpack the parameters
    uid = parameters["uid"]
    impute_missing_values_kw_args = (
        parameters["pipeline_parameters"]["impute_missing_values_kw_args"]
    )
    strategy = impute_missing_values_kw_args["strategy"]

    # Import the data
    X_train, X_test, y_train, y_test, X_holdout = ingest_split(
        train_raw_path=config["train_raw_path"],
        holdout_raw_path=config["holdout_raw_path"],
        target=parameters["target"],
        ingest_split_parameters=parameters["ingest_split_parameters"]
    )

    # Run the function
    X_holdout = impute_missing_values(
        df=X_holdout,
        strategy=strategy
    )

    X_holdout = X_holdout.set_index(uid, drop=True)[["Embarked"]]

    # Run the tests
    assert X_holdout["Embarked"].loc[2] == "S"
    assert X_holdout["Embarked"].loc[9] == "S"
    assert X_holdout["Embarked"].loc[19] == "S"
