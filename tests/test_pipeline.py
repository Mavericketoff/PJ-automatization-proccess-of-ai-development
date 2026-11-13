import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.models.pipeline import build_pipeline


def test_pipeline_returns_pipeline():
    pipe = build_pipeline(["LIMIT_BAL", "AGE"], ["SEX"])
    assert isinstance(pipe, Pipeline)


def test_pipeline_fit_predict():
    n = 60
    df = pd.DataFrame({
        "LIMIT_BAL": np.random.uniform(10000, 300000, n),
        "AGE": np.random.randint(20, 65, n),
        "SEX": np.random.choice([1, 2], n),
        "y": np.random.choice([0, 1], n),
    })
    pipe = build_pipeline(["LIMIT_BAL", "AGE"], ["SEX"])
    pipe.fit(df[["LIMIT_BAL", "AGE", "SEX"]], df["y"])
    preds = pipe.predict(df[["LIMIT_BAL", "AGE", "SEX"]])
    assert len(preds) == n
    assert set(np.unique(preds)).issubset({0, 1})
