import pandas as pd
from src.preprocessing_pipeline import drop_columns


def test_drop_columns():
    """Test for the drop_columns function"""

    # Create dummy data
    data = [
        dict(id=0, col1=1, col2=2, col3=3),
        dict(id=1, col1=1, col2=2, col3=3),
        dict(id=2, col1=1, col2=2, col3=3),
        dict(id=3, col1=1, col2=2, col3=3),
        dict(id=4, col1=1, col2=2, col3=3),
    ]
    df = pd.DataFrame(data)

    # Set the columns to be droped
    drop_column_names = ["col2", "col3"]
    df = drop_columns(df=df, drop_column_names=drop_column_names)

    # Run the tests
    assert df.columns.tolist() == ["id", "col1"]
