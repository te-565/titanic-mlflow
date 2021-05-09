from src import (
    load_config,
    load_parameters,
    ingest_split,
    create_title_cat,
    impute_age
)


def test_impute_age():
    """Test the impute_age function"""

    # Load in the test configuration & parameters
    config = load_config(".env-test")
    parameters = load_parameters(parameters_path=config["parameters_path"])

    # Unpack the parameters
    uid = parameters["uid"]

    # create_title_cat parameters
    create_title_cat_kw_args = (
        parameters["pipeline_parameters"]["create_title_cat_kw_args"]
    )
    title_source_column = create_title_cat_kw_args["source_column"]
    title_dest_column = create_title_cat_kw_args["dest_column"]
    title_codes = create_title_cat_kw_args["title_codes"]

    # impute_age parameters
    impute_age_kw_args = (
        parameters["pipeline_parameters"]["impute_age_kw_args"]
    )
    source_column = impute_age_kw_args["source_column"]
    title_column = impute_age_kw_args["title_column"]
    age_codes = impute_age_kw_args["age_codes"]

    # Import the data
    X_train, X_test, y_train, y_test, X_holdout = ingest_split(
        train_raw_path=config["train_raw_path"],
        holdout_raw_path=config["holdout_raw_path"],
        target=parameters["target"],
        ingest_split_parameters=parameters["ingest_split_parameters"]
    )

    # Create the title category column
    X_holdout = create_title_cat(
        df=X_holdout,
        source_column=title_source_column,
        dest_column=title_dest_column,
        title_codes=title_codes
    )

    # Run the function
    X_holdout = impute_age(
        df=X_holdout,
        source_column=source_column,
        title_column=title_column,
        age_codes=age_codes
    )

    X_holdout = X_holdout.set_index(uid, drop=True)

    # Run the tests
    assert X_holdout[source_column].loc[1] == 22
    assert X_holdout[source_column].loc[4] == 35
    assert X_holdout[source_column].loc[5] == 65
    assert X_holdout[source_column].loc[8] == 40
    assert X_holdout[source_column].loc[9] == 35
    assert X_holdout[source_column].loc[10] == 43
