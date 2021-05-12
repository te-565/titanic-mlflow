from loguru import logger
import sklearn
import pandas as pd
import mlflow
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    average_precision_score,
    recall_score,
)
from statistics import mean
import matplotlib.pyplot as plt
from yellowbrick.features import Rank1D
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.model_selection import LearningCurve
from scikitplot.metrics import plot_roc


def evaluate_model(
    preprocessing_pipeline: sklearn.pipeline.Pipeline,
    model: sklearn,
    X_train: pd.core.frame.DataFrame,
    y_train: pd.core.frame.DataFrame,
    X_test: pd.core.frame.DataFrame,
    y_test: pd.core.frame.DataFrame,
    artifact_path: str,
    cv: int
):
    """
    TODO
    """
    logger.info("Evaluating Model")

    # Generate train & test features
    X_train_features = preprocessing_pipeline.fit_transform(X_train)
    X_test_features = preprocessing_pipeline.fit_transform(X_test)

    # Fit the model & generate predictions
    model.fit(X=X_train_features, y=y_train.values.ravel())

    # Basic Scores
    train_score = model.score(
        X=X_train_features,
        y=y_train.values.ravel()
    )
    test_score = model.score(
        X=X_test_features,
        y=y_test.values.ravel()
    )

    # CV Scores
    train_cv_score = cross_val_score(
        estimator=model,
        X=X_train_features,
        y=y_train.values.ravel(),
        cv=cv
    )
    test_cv_score = cross_val_score(
        estimator=model,
        X=X_test_features,
        y=y_test.values.ravel(),
        cv=cv
    )

    # Precision
    train_y_score = model.decision_function(X=X_train_features)
    train_precision = average_precision_score(
        y_true=y_train.values.ravel(),
        y_score=train_y_score
    )
    test_y_score = model.decision_function(X=X_test_features)
    test_precision = average_precision_score(
        y_true=y_test.values.ravel(),
        y_score=test_y_score
    )

    # Recall
    train_predictions = model.predict(X=X_train_features)
    train_recall = recall_score(
        y_true=y_train,
        y_pred=train_predictions,
        average="macro"
    )
    test_predictions = model.predict(X=X_test_features)
    test_recall = recall_score(
        y_true=y_test,
        y_pred=test_predictions,
        average="macro"
    )

    # Learning Curve Visualisation
    learning_curve = LearningCurve(
        model,
        scoring='accuracy',
        size=(1080, 720)
    )
    learning_curve.fit(X_train_features, y_train.values.ravel())
    outpath = f"{artifact_path}/learning_curve.png"
    learning_curve.show(outpath=outpath)
    plt.close()

    # Feature Ranking Visualisation
    feature_rank = Rank1D(
        algorithm='shapiro',
        features=X_train_features.columns.tolist(),
        size=(1080, 720)
    )
    feature_rank.fit(X_train_features, y_train.values.ravel())
    feature_rank.transform(X_train_features)
    outpath = f"{artifact_path}/feature_importance.png"
    feature_rank.show(outpath=outpath)
    mlflow.log_artifact(outpath)
    plt.close()

    # Confusion Matrix Visualisation
    confusion_matrix = ConfusionMatrix(
        model,
        classes=[0, 1],
        size=(1080, 720),
        is_fitted=True
    )
    confusion_matrix.score(X_test_features, y_test.values.ravel())
    outpath = f"{artifact_path}/confusion_matrix.png"
    confusion_matrix.show(outpath=outpath)
    mlflow.log_artifact(outpath)
    plt.close()

    # ROCAUC Visualisation
    y_probas = model.predict_proba(X_test_features)
    rocauc = plot_roc(y_test.values.ravel(), y_probas)
    outpath = f"{artifact_path}/roc_auc.png"
    plt.savefig(fname=outpath)
    mlflow.log_artifact(outpath)
    plt.close()

    # Save Train & Test data
    X_train_features.reset_index().to_csv(f"{artifact_path}/X_train.csv")
    X_test_features.reset_index().to_csv(f"{artifact_path}/X_test.csv")
    y_train.to_csv(f"{artifact_path}/y_train.csv")
    y_test.to_csv(f"{artifact_path}/y_test.csv")

    # Log MLFlow Artifacts
    mlflow.log_artifact(artifact_path)

    # Log MLFlow Metrics
    mlflow.log_metric("train_score", round(train_score * 100, 2))
    mlflow.log_metric("test_score", round(test_score * 100, 2))
    mlflow.log_metric("train_cv_score", round(mean(train_cv_score) * 100, 2))
    mlflow.log_metric("test_cv_score", round(mean(test_cv_score) * 100, 2))
    mlflow.log_metric("train_precision", round(train_precision * 100, 2))
    mlflow.log_metric("test_precision", round(test_precision * 100, 2))
    mlflow.log_metric("test_recall", round(test_recall * 100, 2))   
    mlflow.log_metric("train_recall", round(train_recall * 100, 2))

    return model
