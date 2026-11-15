"""
Обучение модели с логированием в MLflow.
Запускается как этап DVC-пайплайна.
"""

import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt

from src.data.validation import validate
from src.features.build_features import add_features, get_feature_names
from src.models.pipeline import build_pipeline

PROCESSED = Path("data/processed")
TARGET = "default.payment.next.month"
MLFLOW_URI = "file:./mlruns"
EXPERIMENT_NAME = "credit_default_gbm"


def load_split():
    train_df = pd.read_csv(PROCESSED / "train.csv")
    test_df = pd.read_csv(PROCESSED / "test.csv")

    validate(train_df)
    validate(test_df)
    return train_df, test_df


def run_search(pipeline, X_train, y_train, n_iter=30):
    param_grid = {
        "clf__n_estimators": [100, 200, 300, 500],
        "clf__max_depth": [3, 4, 5, 6],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.15],
        "clf__subsample": [0.7, 0.8, 0.9, 1.0],
        "clf__min_samples_split": [2, 5, 10],
    }
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=5,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def plot_roc(y_true, y_proba, save_path="models/roc_curve.png"):
    from sklearn.metrics import RocCurveDisplay

    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name="GBM")
    ax.set_title("ROC-AUC (test set)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main():
    train_df, test_df = load_split()

    train_df = add_features(train_df)
    test_df = add_features(test_df)

    num_feats, cat_feats = get_feature_names()
    X_train, y_train = train_df[num_feats + cat_feats], train_df[TARGET]
    X_test, y_test = test_df[num_feats + cat_feats], test_df[TARGET]

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 5 экспериментов
    configs = [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9},
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.01, "subsample": 1.0},
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.7},
    ]

    best_auc = 0

    for i, cfg in enumerate(configs, start=1):
        with mlflow.start_run(run_name=f"gbm_exp_{i}"):
            pipe = build_pipeline(num_feats, cat_feats)
            pipe.set_params(**{f"clf__{k}": v for k, v in cfg.items()})
            pipe.fit(X_train, y_train)

            y_proba = pipe.predict_proba(X_test)[:, 1]
            y_pred = pipe.predict(X_test)

            metrics = {
                "roc_auc": roc_auc_score(y_test, y_proba),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
            }

            mlflow.log_params(cfg)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipe, "model")

            print(f"  Exp {i}: AUC={metrics['roc_auc']:.4f}  F1={metrics['f1']:.4f}")

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]

    # Финальный прогон с RandomizedSearchCV
    with mlflow.start_run(run_name="randomized_search_final"):
        pipe = build_pipeline(num_feats, cat_feats)
        search = run_search(pipe, X_train, y_train, n_iter=40)
        best_pipe = search.best_estimator_
        best_params = search.best_params_

        y_proba = best_pipe.predict_proba(X_test)[:, 1]
        y_pred = best_pipe.predict(X_test)

        mlflow.log_params(best_params)
        mlflow.log_metric("cv_best_auc", search.best_score_)
        mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_proba))
        mlflow.log_metric("test_f1", f1_score(y_test, y_pred))
        mlflow.log_metric("test_precision", precision_score(y_test, y_pred))
        mlflow.log_metric("test_recall", recall_score(y_test, y_pred))

        plot_roc(y_test, y_proba)
        mlflow.log_artifact("models/roc_curve.png")
        mlflow.sklearn.log_model(best_pipe, "best_model", registered_model_name="CreditDefaultGBM")

    # сохраняем модель на диск для DVC
    import joblib

    Path("models").mkdir(exist_ok=True)
    joblib.dump(best_pipe, "models/credit_default_model.pkl")

    # метрики для dvc metrics
    metrics_out = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print("\nГотово. Лучшая модель сохранена в models/credit_default_model.pkl")


if __name__ == "__main__":
    main()
