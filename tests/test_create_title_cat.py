from src import (
    load_config,
    load_parameters,
    ingest_split,
    create_title_cat
)


def test_create_title_cat():
    """Test the create_title_cat function"""

    # Load in the test configuration & parameters
    config = load_config(".env-test")
    parameters = load_parameters(parameters_path=config["parameters_path"])

    # Unpack the parameters
    uid = parameters["uid"]
    create_title_cat_kw_args = (
        parameters["pipeline_parameters"]["create_title_cat_kw_args"]
    )
    source_column = create_title_cat_kw_args["source_column"]
    dest_column = create_title_cat_kw_args["dest_column"]
    title_codes = create_title_cat_kw_args["title_codes"]

    # Import the data
    X_train, X_test, y_train, y_test, X_holdout = ingest_split(
        train_raw_path=config["train_raw_path"],
        holdout_raw_path=config["holdout_raw_path"],
        target=parameters["target"],
        ingest_split_parameters=parameters["ingest_split_parameters"]
    )

    # Run the function
    X_holdout = create_title_cat(
        df=X_holdout,
        source_column=source_column,
        dest_column=dest_column,
        title_codes=title_codes
    )

    X_holdout = X_holdout.set_index(uid, drop=True)

    assert X_holdout[dest_column].loc[1] == "gen_male"
    assert X_holdout[dest_column].loc[2] == "young_female"
    assert X_holdout[dest_column].loc[3] == "other_male"
    assert X_holdout[dest_column].loc[4] == "young_male"
    assert X_holdout[dest_column].loc[5] == "gen_female"
    assert X_holdout[dest_column].loc[6] == "gen_female"
    assert X_holdout[dest_column].loc[7] == "other_male"
    assert X_holdout[dest_column].loc[8] == "other_male"
    assert X_holdout[dest_column].loc[9] == "gen_female"
    assert X_holdout[dest_column].loc[10] == "other_male"
    assert X_holdout[dest_column].loc[11] == "other_female"
    assert X_holdout[dest_column].loc[12] == "gen_female"
    assert X_holdout[dest_column].loc[13] == "other_male"
    assert X_holdout[dest_column].loc[14] == "young_female"
    assert X_holdout[dest_column].loc[15] == "other_male"
    assert X_holdout[dest_column].loc[16] == "other_female"
    assert X_holdout[dest_column].loc[17] == "other_female"
    assert X_holdout[dest_column].loc[18] == "other_male"
    assert X_holdout[dest_column].loc[19] == "other_female"
