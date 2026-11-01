from pandera import Column, Check, DataFrameSchema

credit_schema = DataFrameSchema(
    columns={
        "LIMIT_BAL": Column(float, Check.ge(0), nullable=False),
        "SEX": Column(int, Check.isin([1, 2])),
        "EDUCATION": Column(int, Check.isin([0, 1, 2, 3, 4, 5, 6])),
        "MARRIAGE": Column(int, Check.isin([0, 1, 2, 3])),
        "AGE": Column(int, Check.in_range(18, 100)),
        "PAY_0": Column(int, Check.in_range(-2, 8)),
        "PAY_2": Column(int, Check.in_range(-2, 8)),
        "PAY_3": Column(int, Check.in_range(-2, 8)),
        "PAY_4": Column(int, Check.in_range(-2, 8)),
        "PAY_5": Column(int, Check.in_range(-2, 8)),
        "PAY_6": Column(int, Check.in_range(-2, 8)),
        "BILL_AMT1": Column(float, nullable=True),
        "BILL_AMT2": Column(float, nullable=True),
        "PAY_AMT1": Column(float, Check.ge(0)),
        "PAY_AMT2": Column(float, Check.ge(0)),
        "default.payment.next.month": Column(int, Check.isin([0, 1])),
    },
    strict=False,  # допускаем доп. колонки после feature engineering
    coerce=True,
)


def validate(df):
    """Валидирует датафрейм. Генерирует исключение при ошибке валидации"""
    return credit_schema.validate(df, lazy=True)
