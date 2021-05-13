import pandas as pd
from src.preprocessing_pipeline import create_family_size


def test_create_family_size():
    """Test the create_family_size function"""

    # Create dummy data
    data = [
        dict(id=1, col1=1, col2=0),
        dict(id=2, col1=3, col2=1),
        dict(id=3, col1=0, col2=2),
        dict(id=4, col1=3, col2=0),
        dict(id=5, col1=0, col2=0),
    ]
    df = pd.DataFrame(data).set_index("id", drop=True)

    # Run the function
    df_out = create_family_size(
        df=df,
        source_columns=["col1", "col2"],
        dest_column="col3"
    )

    # Run the tests
    assert df_out["col3"].loc[1] == 2
    assert df_out["col3"].loc[2] == 5
    assert df_out["col3"].loc[3] == 3
    assert df_out["col3"].loc[4] == 4
    assert df_out["col3"].loc[5] == 1
