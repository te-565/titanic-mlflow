import pandas as pd
from src.preprocessing_pipeline import set_df_index


def test_set_df_index():
    """Test the set_df_index function"""

    # Create the data
    data = [
        dict(id=1, col1=10, col2="A", col3=1),
        dict(id=2, col1=10, col2="A", col3=1),
        dict(id=3, col1=10, col2="A", col3=1),
        dict(id=4, col1=9, col2="B", col3=0),
    ]

    df = pd.DataFrame(data)

    # Run the function
    df_out = set_df_index(df=df, df_index_col="id")

    # Run the test
    assert df_out.index.name == "id"
