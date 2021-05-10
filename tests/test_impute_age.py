import pandas as pd
import numpy as np
from src import (
    load_config,
    load_parameters,
    impute_age
)


def test_impute_age():
    """Test the impute_age function"""

    # Load in the test configuration & parameters
    config = load_config(".env-test")
    parameters = load_parameters(parameters_path=config["parameters_path"])
    age_codes = (
        parameters["pipeline_parameters"]["impute_age_kw_args"]
        ["age_codes"]
    )

    # Create the data
    data = [
        dict(id=1, col1="gen_male", col2=np.nan),
        dict(id=2, col1="gen_female", col2=None),
        dict(id=3, col1="young_female", col2=np.nan),
        dict(id=4, col1="young_male", col2=None),
        dict(id=5, col1="other_male", col2=np.nan),
        dict(id=6, col1="other_female", col2=None),
        dict(id=7, col1="gen_male", col2=12),
        dict(id=8, col1="gen_female", col2=22),
        dict(id=9, col1="young_female", col2=32),
        dict(id=10, col1="young_male", col2=42),
        dict(id=11, col1="other_male", col2=52),
        dict(id=12, col1="other_female", col2=62),
    ]

    df = pd.DataFrame(data).set_index("id", drop=True)

    # Run the function
    df_out = impute_age(
        df=df,
        source_column="col2",
        title_cat_column="col1",
        age_codes=age_codes
    )

    # Run the tests
    assert df_out["col2"].loc[1] == 30
    assert df_out["col2"].loc[2] == 35
    assert df_out["col2"].loc[3] == 21
    assert df_out["col2"].loc[4] == 5
    assert df_out["col2"].loc[5] == 40
    assert df_out["col2"].loc[6] == 50
    assert df_out["col2"].loc[7] == 12
    assert df_out["col2"].loc[8] == 22
    assert df_out["col2"].loc[9] == 32
    assert df_out["col2"].loc[10] == 42
    assert df_out["col2"].loc[11] == 52
    assert df_out["col2"].loc[12] == 62
