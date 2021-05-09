import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler,
    OneHotEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src import (
    set_df_index,
    create_title_cat,
    impute_age,
    create_family_size,
    drop_columns,
    impute_missing_values,
    scaler
)


def create_pipeline(
    pipeline_parameters: dict
):
    """
    Description
    -----------
    Create a scikit learn pipeline as a series of functions applied to the
    dataframe via scikit-learn's FunctionTransformer class.

    Each transformation step has a function assigned with the keyword arguments
    applied through the supplied pipeline_parameters object.

    Parameters
    ----------
    pipeline_parameters: dict
        Parameters containing the metadata associated with the pipeline
        transformations.

    Returns
    -------
    pipeline: sklearn.pipeline.Pipeline
        The scikit-learn pipeline

    Raises:
    -------
    Exception: Exception
        Generic exception for logging

    Examples
    --------
    pipeline = create_pipeline(
        ("Step Description", FunctionTransformer(
            func=my_func_name,
            kw_args={"keyword_name" : "keyword_arg}
        ))
    )
    """

    # Create the pre-processing pipeline
    preprocess_pipeline = Pipeline([
        ("Set dataframe index", FunctionTransformer(
            func=set_df_index,
            kw_args=pipeline_parameters["set_df_index_kw_args"]
        )),
        ("Create title_cat column", FunctionTransformer(
            func=create_title_cat,
            kw_args=pipeline_parameters["create_title_cat_kw_args"]
        )),
        ("Impute missing Age values", FunctionTransformer(
            func=impute_age,
            kw_args=pipeline_parameters["impute_age_kw_args"]
        )),
        ("Create family_size column", FunctionTransformer(
            func=create_family_size,
            kw_args=pipeline_parameters["create_family_size_kw_args"]
        )),
        ("Drop columns", FunctionTransformer(
            func=drop_columns,
            kw_args=pipeline_parameters["drop_columns_kw_args"]
        )),
        ("Impute missing values", FunctionTransformer(
            func=impute_missing_values,
            kw_args=pipeline_parameters["impute_missing_values_kw_args"]
        )),
        ("Scale numeric data", FunctionTransformer(
            func=scaler,
            kw_args=pipeline_parameters["scaler_kw_args"]
        ))
        # One Hot
    ])

    return preprocess_pipeline
