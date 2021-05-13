import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from loguru import logger


def ingest_split(
    train_test_raw_path: str,
    holdout_raw_path: str,
    target: str,
    ingest_split_parameters: dict,
):
    """
    Description
    -----------
    Ingest the training dataset and holdout datasets and split the training
    dataset into train and test.

    Paramaters
    ----------
    train_test_raw_path: str
        Location of the raw train & test csv file to be ingested

    holdout_raw_path: str
        Location of the raw holdout csv file to be ingested

    target: str
        Name of the target column to be predicted in the datasets

    test_clean: str
        Location to export the processed test_clean.csv file to

    ingest_split_parameters: dict
        Dictionary containing the parameters for the function.

    Returns
    -------
    X_train: pd.core.frame.DataFrame
        The dataframe of features for training the model.

    X_test: pd.core.frame.DataFrame
        The dataframe of features for testing the model.

    y_train: pd.core.frame.DataFrame
        The dataframe containing the target for training the model.

    y_test: pd.core.frame.DataFrame
        The dataframe containing the target for testing the model.

    X_holdout: pd.core.frame.DataFrame
        The dataframe of unseen data to predict values for.


    Raises
    ------
    None

    Examples
    --------

    df_train, df_test = ingest_clean(
        train_test_raw_path="path/to/ingest.csv",
        holdout_raw_path="path/to/export.csv",
        target="target_col",
        ingest_split_parameters=dict(
            random_state=2,
            train_size=0.5,
            test_size=0.5
        )
    )
    """
    try:
        logger.info("Running ingest_split()")

        # Unpack Parameters
        train_size = ingest_split_parameters["train_size"]
        test_size = ingest_split_parameters["test_size"]
        random_state = ingest_split_parameters["random_state"]

        # Log MLflow parameters
        mlflow.log_param("train_size", train_size)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        # Import train & holdout datasets
        df_train = pd.read_csv(train_test_raw_path)
        df_holdout = pd.read_csv(holdout_raw_path)

        # Convert ints to floats
        for column in df_train.columns.tolist():
            if isinstance(df_train[column], np.int64):
                df_train[column] = df_train[column].astype(float)

        for column in df_holdout.columns.tolist():
            if isinstance(df_holdout[column], np.int64):
                df_holdout[column] = df_holdout[column].astype(float)

        # Split the features and target
        X = df_train.drop(target, axis=1)
        y = df_train[[target]]
        X_holdout = df_holdout

        # Train Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=train_size,
            test_size=test_size,
            random_state=random_state
        )

        return X_train, X_test, y_train, y_test, X_holdout

    except Exception:
        logger.exception("Exception in ingest_split()")
