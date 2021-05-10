from loguru import logger
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from src import (
    set_df_index,
    convert_to_str,
    create_title_cat,
    impute_age,
    create_family_size,
    drop_columns,
    impute_missing_values,
    scaler,
    one_hot_encoder
)


def create_preprocessing_pipeline(
    pipeline_parameters: dict
):
    """
    Description
    -----------
    Create a scikit learn pipeline to preprocess the data ready for modelling.
    The pipeline uses a series of functions applied to the dataframe via
    scikit-learn's FunctionTransformer class.

    Each transformation step has a function assigned with the keyword arguments
    applied through the supplied pipeline_parameters object.

    Note that the pipeline works with pandas DataFrames over numpy arrays
    because these are more interpretable and can be logged as artifacts.

    Parameters
    ----------
    pipeline_parameters: dict
        Parameters containing the metadata associated with the pipeline
        transformations.

    Returns
    -------
    preprocessing_pipeline: sklearn.pipeline.Pipeline
        The preprocessing pipeline

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
    try:
        # Create the pre-processing pipeline
        preprocessing_pipeline = Pipeline([
            ("Set dataframe index", FunctionTransformer(
                func=set_df_index,
                kw_args=pipeline_parameters["set_df_index_kw_args"]
            )),
            ("Convert cols to string", FunctionTransformer(
                func=convert_to_str,
                kw_args=pipeline_parameters["convert_to_str_kw_args"]
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
            )),
            ("One hot encode categorical data", FunctionTransformer(
                func=one_hot_encoder,
                kw_args=pipeline_parameters["one_hot_kw_args"]
            ))
        ])

        return preprocessing_pipeline

    except Exception:
        logger.exception("Error in preprocessing_pipeline()")
