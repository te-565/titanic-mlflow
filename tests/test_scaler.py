import pandas as pd
from src.preprocessing_pipeline import scaler


def test_scaler():
    """
    Test the scaler function.

    Note that this test is executed using generic data as it would:
        1. Require a lot of pre-processing to test with the contextual test
        data.
        2. The test can ship alongside the function when implemented in
        other projects.
    """

    data = [
        dict(id=1, value=2),
        dict(id=2, value=3),
        dict(id=3, value=4),
        dict(id=4, value=5),
        dict(id=5, value=6),
    ]

    df = pd.DataFrame(data)

    # Run the function
    df_out = scaler(
        df=df,
        scale_columns=["value"]
    )

    # Run the tests
    assert df_out["value"].min() == 0
    assert df_out["value"].max() == 1
