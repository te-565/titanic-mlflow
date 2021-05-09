from src import (
    load_config,
    load_parameters,
    ingest_split,
    create_family_size
)


def test_create_family_size():
    """Test the create_family_size function"""

    # Load in the test configuration & parameters
    config = load_config(".env-test")
    parameters = load_parameters(parameters_path=config["parameters_path"])

    # Unpack the parameters
    uid = parameters["uid"]
    create_family_size_kw_args = (
        parameters["pipeline_parameters"]["create_family_size_kw_args"]
    )
    source_columns = create_family_size_kw_args["source_columns"]
    dest_column = create_family_size_kw_args["dest_column"]

    # Import the data
    X_train, X_test, y_train, y_test, X_holdout = ingest_split(
        train_raw_path=config["train_raw_path"],
        holdout_raw_path=config["holdout_raw_path"],
        target=parameters["target"],
        ingest_split_parameters=parameters["ingest_split_parameters"]
    )

    # Run the function
    X_holdout = create_family_size(
        df=X_holdout,
        source_columns=source_columns,
        dest_column=dest_column
    )

    X_holdout = X_holdout.set_index(uid, drop=True)

    # Run the tests
    assert X_holdout[dest_column].loc[1] == 6
    assert X_holdout[dest_column].loc[9] == 11
    assert X_holdout[dest_column].loc[10] == 1
    assert X_holdout[dest_column].loc[19] == 2
