import pandas as pd 
import numpy as np
from src import (
    load_config,
    load_parameters
)
from src.preprocessing_pipeline import impute_missing_values

def test_impute_missing_values():
    """Test the impute_missing_values function"""

    # Load in the test configuration & parameters
    config = load_config(".env-test")
    parameters = load_parameters(parameters_path=config["parameters_path"])
    strategy = (
        parameters["pipeline_parameters"]["impute_missing_values_kw_args"]
        ["strategy"]
    )

    # Create the data
    data = [
        dict(id=1, col1=10, col2="A", col3=1),
        dict(id=2, col1=10, col2="A", col3=1),
        dict(id=3, col1=10, col2="A", col3=1),
        dict(id=4, col1=9, col2="B", col3=0),
        dict(id=5, col1=None, col2=np.nan, col3=None),
        dict(id=6, col1=np.nan, col2="", col3=None)
    ]

    df = pd.DataFrame(data).set_index("id", drop=True)

    # Run the function
    df_out = impute_missing_values(df=df, strategy=strategy)

    # Run the tests
    assert df_out["col1"].loc[5] == 10
    assert df_out["col1"].loc[6] == 10
    assert df_out["col2"].loc[5] == "A"
    assert df_out["col2"].loc[6] == "A"
    assert df_out["col3"].loc[5] == 1
    assert df_out["col3"].loc[6] == 1
