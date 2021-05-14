from src.utils.utils import (
    load_config,
    load_parameters
)


def test_load_parameters():
    """Test for the load_parameters function"""

    config = load_config(".env.test")
    parameters = load_parameters(config["parameters_path"])

    assert isinstance(parameters, dict)
    assert isinstance(parameters["ingest_split_parameters"], dict)
    assert isinstance(parameters["pipeline_parameters"], dict)
    assert isinstance(parameters["logreg_hyperparameters"], dict) 
    assert isinstance(
        parameters["pipeline_parameters"]["convert_to_str_kw_args"]
        ["convert_to_str_cols"],
        list
    )
