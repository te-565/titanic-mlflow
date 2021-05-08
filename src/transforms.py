from loguru import logger
import pandas as pd
import re


def set_df_index(
    df: pd.core.frame.DataFrame,
    df_index_col: str,
):
    """
    Description
    -----------
    Sets an index of the supplied column name for the supplied DataFrame

    Parameters
    ----------
    df: pandas.core.frame.DataFrame
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
        df_out = df.set_index(df_index_col, drop=True)

    except Exception:
        logger.exception("Error in set_df_index()")

    return df_out


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
        logger.exception("Error in drop_columns()")


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
        row,
        source_column
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

    # Apply the extract_title function to the dataframe
    try:
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
        logger.exception("Error in create_title_cat()")
