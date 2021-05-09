import pandas as pd
from src import (
    scaler
)

def test_scaler():
    """Test the scaler function"""

    data = [
        dict(id=1, value=2),
        dict(id=2, value=3),
        dict(id=3, value=4),
        dict(id=3, value=5),
        dict(id=3, value=6),
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
