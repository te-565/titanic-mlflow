import pandas as pd
from src import one_hot_encoder


def test_one_hot_encoder():
    """
    Test the one_hot_encoder function.

   Note that this test is executed using generic data as it would:
        1. Require a lot of pre-processing to test with the contextual test
        data.
        2. The test can ship alongside the function when implemented in
        other projects.
    """

    # Create the data
    data = [
        dict(id=1, col1="foo", col2="wibble"),
        dict(id=2, col1="foo", col2="wibble"),
        dict(id=3, col1="bar", col2="wubble"),
        dict(id=4, col1="bar", col2="wibble"),
        dict(id=5, col1="bar", col2="wubble"),
    ]
    df = pd.DataFrame(data).set_index("id", drop=True)

    # Set the parameters
    one_hot_columns = ["col1", "col2"]

    # Run the function
    df_out = one_hot_encoder(df=df, one_hot_columns=one_hot_columns)
    
    # Run the tests
    assert df_out.columns.tolist() == [
        "col1_bar", "col1_foo", "col2_wibble", "col2_wubble"
    ]
    assert df_out["col1_foo"].tolist() == [1, 1, 0, 0, 0]
    assert df_out["col1_bar"].tolist() == [0, 0, 1, 1, 1]
    assert df_out["col2_wibble"].tolist() == [1, 1, 0, 1, 0]
    assert df_out["col2_wubble"].tolist() == [0, 0, 1, 0, 1]
    
    