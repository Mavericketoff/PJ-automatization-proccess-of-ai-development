import pandera.pandas as pa

credit_schema = pa.DataFrameSchema(
    columns={
        "LIMIT_BAL": pa.Column(float, pa.Check.ge(0), nullable=False),
        "SEX": pa.Column(int, pa.Check.isin([1, 2])),
        "EDUCATION": pa.Column(int, pa.Check.isin([0, 1, 2, 3, 4, 5, 6])),
        "MARRIAGE": pa.Column(int, pa.Check.isin([0, 1, 2, 3])),
        "AGE": pa.Column(int, pa.Check.in_range(18, 100)),
        "PAY_0": pa.Column(int, pa.Check.in_range(-2, 8)),
        "PAY_2": pa.Column(int, pa.Check.in_range(-2, 8)),
        "PAY_3": pa.Column(int, pa.Check.in_range(-2, 8)),
        "PAY_4": pa.Column(int, pa.Check.in_range(-2, 8)),
        "PAY_5": pa.Column(int, pa.Check.in_range(-2, 8)),
        "PAY_6": pa.Column(int, pa.Check.in_range(-2, 8)),
        "BILL_AMT1": pa.Column(float, nullable=True),
        "BILL_AMT2": pa.Column(float, nullable=True),
        "PAY_AMT1": pa.Column(float, pa.Check.ge(0)),
        "PAY_AMT2": pa.Column(float, pa.Check.ge(0)),
        "default.payment.next.month": pa.Column(int, pa.Check.isin([0, 1])),
    },
    strict=False,  # допускаем доп. колонки после feature engineering
    coerce=True,
)


def validate(df):
    """Валидирует датафрейм. Генерирует исключение при ошибке валидации"""
    return credit_schema.validate(df, lazy=True)
