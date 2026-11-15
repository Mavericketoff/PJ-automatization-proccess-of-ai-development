import pandas as pd
import pytest

from src.features.build_features import add_features


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "LIMIT_BAL": [50000, 120000],
        "AGE": [30, 45],
        "SEX": [1, 2],
        "EDUCATION": [2, 1],
        "MARRIAGE": [1, 2],
        "PAY_0": [0, 2],
        "PAY_2": [-1, 3],
        "PAY_3": [0, 0],
        "PAY_4": [0, -1],
        "PAY_5": [0, 0],
        "PAY_6": [0, 1],
        "BILL_AMT1": [1000, 5000],
        "BILL_AMT2": [800, 4500],
        "BILL_AMT3": [700, 4000],
        "BILL_AMT4": [600, 3500],
        "BILL_AMT5": [500, 3000],
        "BILL_AMT6": [400, 2500],
        "PAY_AMT1": [500, 2000],
        "PAY_AMT2": [400, 1800],
        "PAY_AMT3": [300, 1500],
        "PAY_AMT4": [200, 1200],
        "PAY_AMT5": [100, 1000],
        "PAY_AMT6": [100, 800],
        "default.payment.next.month": [0, 1],
    })


def test_add_features_creates_new_cols(sample_df):
    result = add_features(sample_df)
    for col in ["PAY_MEAN", "PAY_MAX", "PAY_LATE_RATIO", "TOTAL_BILL", "TOTAL_PAY", "AGE_BIN"]:
        assert col in result.columns, f"Не хватает колонки {col}"


def test_pay_late_ratio_range(sample_df):
    result = add_features(sample_df)
    assert result["PAY_LATE_RATIO"].between(0, 1).all()


def test_no_rows_dropped(sample_df):
    result = add_features(sample_df)
    assert len(result) == len(sample_df)
