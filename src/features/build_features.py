import pandas as pd

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
PAY_AMT_COLS = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет агрегированные фичи и возвращает копию."""
    out = df.copy()

    # среднее и максимальное число месяцев просрочки
    out["PAY_MEAN"] = out[PAY_COLS].mean(axis=1)
    out["PAY_MAX"] = out[PAY_COLS].max(axis=1)

    out["PAY_LATE_RATIO"] = (out[PAY_COLS] >= 2).sum(axis=1) / len(PAY_COLS)

    # суммарный счёт и суммарные выплаты
    out["TOTAL_BILL"] = out[BILL_COLS].sum(axis=1)
    out["TOTAL_PAY"] = out[PAY_AMT_COLS].sum(axis=1)

    # отношение выплат к лимиту
    out["PAY_TO_LIMIT"] = out["TOTAL_PAY"] / (out["LIMIT_BAL"] + 1)

    # бинаризируем возраст
    out["AGE_BIN"] = pd.cut(out["AGE"], bins=[17, 25, 35, 45, 55, 100], labels=False)

    return out


def get_feature_names() -> tuple[list[str], list[str]]:
    """Возвращает фичи для pipeline."""
    numeric = [
        "LIMIT_BAL", "AGE", "PAY_MEAN", "PAY_MAX", "PAY_LATE_RATIO",
        "TOTAL_BILL", "TOTAL_PAY", "PAY_TO_LIMIT",
        "BILL_AMT1", "PAY_AMT1",
    ]
    categorical = ["SEX", "EDUCATION", "MARRIAGE", "AGE_BIN"]
    return numeric, categorical
