from src import (
    load_config,
    load_parameters,
    ingest_split,
    set_df_index
)


def test_set_df_index():
    """Test the set_df_index function"""

    # Load in the test configuration & parameters
    config = load_config(".env-test")
    parameters = load_parameters(parameters_path=config["parameters_path"])

    # Unpack the parameters
    set_df_index_kw_args = (
        parameters["pipeline_parameters"]["set_df_index_kw_args"]
    )
    df_index_col = set_df_index_kw_args["df_index_col"]

    # Import the data
    X_train, X_test, y_train, y_test, X_holdout = ingest_split(
        train_raw_path=config["train_raw_path"],
        holdout_raw_path=config["holdout_raw_path"],
        target=parameters["target"],
        ingest_split_parameters=parameters["ingest_split_parameters"]
    )

    # Run the function
    X_holdout = set_df_index(df=X_holdout, df_index_col=df_index_col)

    assert X_holdout.index.name == df_index_col
