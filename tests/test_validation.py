import pandas as pd
import pytest

from src.data.validation import validate


def test_valid_data_passes():
    df = pd.DataFrame(
        {
            "LIMIT_BAL": [50000.0],
            "SEX": [1],
            "EDUCATION": [2],
            "MARRIAGE": [1],
            "AGE": [30],
            "PAY_0": [0],
            "PAY_2": [0],
            "PAY_3": [0],
            "PAY_4": [0],
            "PAY_5": [0],
            "PAY_6": [0],
            "BILL_AMT1": [1000.0],
            "BILL_AMT2": [900.0],
            "PAY_AMT1": [500.0],
            "PAY_AMT2": [400.0],
            "default.payment.next.month": [0],
        }
    )
    # не должно бросать исключение
    validate(df)


def test_invalid_age_fails():
    df = pd.DataFrame(
        {
            "LIMIT_BAL": [50000.0],
            "SEX": [1],
            "EDUCATION": [2],
            "MARRIAGE": [1],
            "AGE": [15],
            "PAY_0": [0],
            "PAY_2": [0],
            "PAY_3": [0],
            "PAY_4": [0],
            "PAY_5": [0],
            "PAY_6": [0],
            "BILL_AMT1": [1000.0],
            "BILL_AMT2": [900.0],
            "PAY_AMT1": [500.0],
            "PAY_AMT2": [400.0],
            "default.payment.next.month": [0],
        }
    )
    with pytest.raises(Exception):
        validate(df)
