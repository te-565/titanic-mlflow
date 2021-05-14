import pandas as pd
from src.preprocessing_pipeline import set_df_index


def test_set_df_index():
    """Test the set_df_index function"""

    # Create the DataFrame data
    data = [
        dict(id=1, col1=10, col2="A", col3=1),
        dict(id=2, col1=10, col2="A", col3=1),
        dict(id=3, col1=10, col2="A", col3=1),
        dict(id=4, col1=9, col2="B", col3=0),
    ]

    # Create a DataFrame and a Series
    df = pd.DataFrame(data)
    sr = pd.Series(data[0])

    # Run the function
    df_out = set_df_index(df=df, df_index_col="id")
    sr_out = set_df_index(df=sr, df_index_col="id")

    # Run the DataFrame test
    assert df_out.index.name == "id"
    assert sr_out.index.name == "id"
    assert df_out.columns.tolist() == ["col1", "col2", "col3"]
    assert sr_out.columns.tolist() == ["col1", "col2", "col3"]
