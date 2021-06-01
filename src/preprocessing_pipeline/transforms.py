import re
from loguru import logger
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder
)


def set_df_index(
    df: pd,
    df_index_col: str,
):
    """
    Description
    -----------
    Sets an index of the supplied column name for the supplied DataFrame

    Parameters
    ----------
    df: pandas.core.frame.DataFrame or pandas.core.series.Series
        The dataframe to be processed

    df_index: list
        The names of the column to set as the index

    Returns
    -------
    df_out: pandas.DataFrame
        The processed pandas Dataframe

    Raises
    ------
    Exception: Exception
        Generic exception for logging

    Examples
    --------
    df_out = set_df_index(
        df=df
        df_index_col="col1"
    )
    """
    logger.info("Running set_df_index()")

    try:
        df_out = df.copy()

        # Handle single records which are passed as a series
        if isinstance(df_out, pd.core.series.Series):
            df_out = (
                pd.DataFrame(df_out)
                .transpose()
                .set_index(df_index_col, drop=True)
            )

        # Handle multiple records which are passed as a DataFrame
        else:
            df_out = df_out.set_index(df_index_col, drop=True)

        return df_out

    except Exception:
        logger.exception("Error running set_df_index()")


def convert_to_str(
    df: pd.core.frame.DataFrame,
    convert_to_str_cols: list
):
    """
    Description
    -----------
    Converts the supplied columns to strings for the supplied DataFrame

    Parameters
    ----------
    df: pandas.core.frame.DataFrame
        The dataframe to be processed

    convert_to_str_cols: list
        The names of the columns to convert to strings

    Returns
    -------
    df_out: pandas.DataFrame
        The processed pandas Dataframe

    Raises
    ------
    Exception: Exception
        Generic exception for logging

    Examples
    --------
    df_out = convert_to_str(
        df=df
        convert_to_str_cols="col1"
    )
    """
    logger.info("Running convert_to_str()")

    try:
        df_out = df.copy()

        for column in convert_to_str_cols:
            df_out[column] = df_out[column].astype(str)

        return df_out

    except Exception:
        logger.exception("Error running convert_to_str()")


def drop_columns(
    df: pd.core.frame.DataFrame,
    drop_column_names: list,
):
    """
    Description
    -----------
    Removes the specified columns from the supplied DataFrame

    Parameters
    ----------
    df: pandas.core.frame.DataFrame
        The dataframe to be processed

    drop_column_names: list
        The names of the columns to remove

    Returns
    -------
    df_out: pandas.DataFrame
        The processed pandas Dataframe

    Raises
    ------
    Exception: Exception
        Generic exception for logging

    Examples
    --------
    df_out = drop_columns(
        df=df
        drop_column_names=["col1, "col2]
    )
    """

    logger.info("Running drop_columns()")

    try:
        df_out = df.drop(labels=drop_column_names, axis=1)

        return df_out

    except Exception:
        logger.exception("Error running drop_columns()")


def create_title_cat(
    df: pd.core.frame.DataFrame,
    source_column: str,
    dest_column: str,
    title_codes: dict
):
    """
    Description
    -----------
    Feature Engineers the title column of a pandas Dataframe by extracting the
    title from the source column via regex, coding the values and creating the
    dest_column.

    Contains the extract title sub-function which extracts the blocks of text
    and picks the group containing the title which will always be at index 1.

    Parameters
    ----------
    df: pandas.core.frame.DataFrame
        The dataframe to be processed

    source_column: str
        The coulumn containing the data from which to extract the title.

    dest_column: str
        The new column to create containing the extracted title.

    title_codes: dict
        Dictionary containing the title values as keys (e.g. Mr, Mrs, mme etc.)
        and the corresponding codes as values (e.g. gen_male, other_female etc.)

    Returns
    -------
    df_out: pandas.DataFrame
        The processed pandas Dataframe

    Raises
    ------
    Exception: Exception
        Generic exception for logging

    Examples
    --------
    df_out = create_title_cat(
        df=df
        source_column="col1",
        dest_column="col2"
        title_codes: {
            "Mr": "gen_male".
            "Mrs: "gen_female"s
        }
    )
    """

    logger.info("Running create_title_cat()")

    # Define the extract_title function
    def extract_title(
        row: pd.core.series.Series,
        source_column: str
    ):
        """
        Extracts the title from the supplied specified title_source_column via
        a regex. Applied to a pandas DataFrame
        """

        title_search = re.search(r' ([A-Za-z]+)\.', row[source_column])

        if title_search:
            title = title_search.group(1)

        else:
            title = ""

        return title

    try:
        # Apply the extract_title function to the dataframe
        df_out = df.copy()

        df_out[dest_column] = (
            df_out.apply(
                extract_title,
                args=([source_column]),
                axis=1
            )
            .replace(title_codes)
        )

        return df_out

    except Exception:
        logger.exception("Error running create_title_cat()")


def impute_age(
    df: pd.core.frame.DataFrame,
    source_column: str,
    title_cat_column: str,
    age_codes: dict
):
    """
    If the age of a passenger is missing, infer this based upon the passenger
    title.

    Parameters
    ----------
    df: pandas.DataFrame
        The dataframe to be processed.

    source_column: str
        The column containing the age values.

    title_cat_column: str
        The column containing the title category values.

    age_codes: dict
        Dictionary containing the title category values as keys (e.g. "gen_male"
        "gen_female", and the age to infer as values.

    Returns
    -------
    df: pandas.DataFrame
        The processed dataframe.

    Raises
    ------
    Exception: Exception
        Generic exception for logging

    Examples
    --------
    df = impute_age(
        df=df,
        source_column="Age",
        title_cat_column="TitleCat",
        age_codes=dict(
            gen_male=30,
            gen_female=35
            . . .
        )
    )
    """

    logger.info("Running impute_age()")

    def infer_age(
        row: pd.core.series.Series,
        source_column: str,
        title_cat_column: str,
        age_codes: dict
    ):
        """Infers the age of a passenger based upon the passenger title,
        Applied to a pandas dataframe"""

        if (pd.isnull(row[source_column])):

            # Iterate through the codes and assign an age based upon the title
            for key, value in age_codes.items():
                if row[title_cat_column] == key:
                    age = value

        # Else return the age as an integer
        else:
            age = int(row[source_column])

        return age

    try:
        # Apply the infer_age function to the pandas dataframe
        df_out = df.copy()
        df_out[source_column] = (
            df_out.apply(
                infer_age,
                args=([source_column, title_cat_column, age_codes]),
                axis=1
            )
        )

        return df_out

    except Exception:
        logger.exception("Error running infer_age()")


def create_family_size(
    df: pd.core.frame.DataFrame,
    source_columns: list,
    dest_column: str
):
    """
    Description
    -----------
    Create a column for family_size via summing the source_columns.

    Parameters
    ----------
    df: pd.core.frame.DataFrame
        The dataframe to be processed.

    source_columns: list
        The columns to be summed to calculate the family size.

    dest_column: str
        The destination column to contain the family size values.

    Returns
    -------
    df_out: pd.core.frame.DataFrame.
        The processed dataframe

    Raises
    ------
    Exception: Exception
        Generic exception for logging

    Examples
    --------
    df = create_family_size(
        df=df,
        source_columns=["col1", "col2"],
        dest_column="col3
    )
    """

    logger.info("Running create_family_size()")

    try:
        df_out = df.copy()
        df_out[dest_column] = df_out.apply(
            lambda row: row[source_columns].sum() + 1,
            axis=1
        )

        return df_out

    except Exception:
        logger.exception("Error running create_family_size()")


def impute_missing_values(
    df: pd.core.frame.DataFrame,
    strategy: str
):
    """
    Description
    -----------
    Impute missing values for the dataframe via the specified strategy. Creates
    separate imputers for:
        * np.nan
        * None
        * "" (empty strings)
        * int

    Parameters
    ----------
    df: pd.core.frame.DataFrame
        The dataframe to be processed.

    strategy: str
        The strategy to use for imputation

    Returns
    -------
    df_out: pd.core.frame.DataFrame.
        The processed dataframe

    Raises
    ------
    Exception: Exception
        Generic exception for logging

    Examples
    --------
    df = impute_missing_values(
        df=df,
        strategy="most_frequent"
    )
    """
    logger.info("Running impute_missing_values()")

    try:
        df_out = df.copy()

        # Create imputers for various dtypes
        nan_imputer = SimpleImputer(
            missing_values=np.nan,
            strategy=strategy
        )
        none_imputer = SimpleImputer(
            missing_values=None,
            strategy=strategy
        )
        str_imputer = SimpleImputer(
            missing_values="",
            strategy=strategy
        )
        int_imputer = SimpleImputer(
            missing_values=int,
            strategy=strategy
        )

        df_out[:] = nan_imputer.fit_transform(df_out)
        df_out[:] = none_imputer.fit_transform(df_out)
        df_out[:] = str_imputer.fit_transform(df_out)
        df_out[:] = int_imputer.fit_transform(df_out)

        return df_out

    except Exception:
        logger.exception("Error running impute_missing_values()")


def scaler(
    df: pd.core.frame.DataFrame,
    scale_columns: str
):
    """
    Description
    -----------
    Scale the supplied scale_columns for the dataframe.

    Parameters
    ----------
    df: pd.core.frame.DataFrame
        The dataframe to be processed.

    scale_columns: str
        The columsn to apply scaling to.

    Returns
    -------
    df_out: pd.core.frame.DataFrame.
        The processed dataframe

    Raises
    ------
    Exception: Exception
        Generic exception for logging

    Examples
    --------
    df = scaler(
        df=df,
        scale_columns=["col1", "col2"]
    )
    """
    logger.info("Running scaler()")
    try:
        df_out = df.copy()

        for column in scale_columns:

            scale = MinMaxScaler()
            df_out[column] = scale.fit_transform(
                df_out[column].values.reshape(-1, 1)
            )

        return df_out

    except Exception:
        logger.exception("Error running scaler()")


def one_hot_encoder(
    df: pd.core.frame.DataFrame,
    uid: str,
    one_hot_columns: list
):
    """
    Description
    -----------
    One hot encode the supplied scale_columns for the dataframe.

    Parameters
    ----------
    df: pd.core.frame.DataFrame
        The dataframe to be processed.

    one_hot_columns: str
        The columns to apply one hot encoding to.

    Returns
    -------
    df_out: pd.core.frame.DataFrame.
        The processed dataframe

    Raises
    ------
    Exception: Exception
        Generic exception for logging

    Examples
    --------
    df = one_hot_encoder(
        df=df,
        one_hot_columns=["col1", "col2"]
    )
    """

    logger.info("running one_hot_encoder()")

    try:
        df_out = df.copy()

        # Reset to a generic index to allow merge by position
        df_out.reset_index(inplace=True)

        for column in one_hot_columns:

            # Unpack column dictionary
            col_name = column["col_name"]
            categories = column["categories"]

            # Set the column type as categorical
            df_out[col_name] = df_out[col_name].astype("category")

            # Create the encoder
            oh_enc = OneHotEncoder(
                categories=[categories],
                sparse=False
            )

            # Fit the data to the encoder
            enc_data = oh_enc.fit_transform(df_out[[col_name]].values)
            # Set the column names
            columns = [f"{col_name}_{col}" for col in categories]
            # Create the dataframe of encoded data
            df_oh = pd.DataFrame(enc_data, columns=columns)
            # Join the dataframe on to the output dataframe
            df_out = df_out.join(df_oh).drop(col_name, axis=1)

        # Set the index again
        df_out.set_index(uid, inplace=True)

        return df_out

    except Exception:
        logger.exception("Error running one_hot_encoder()")
