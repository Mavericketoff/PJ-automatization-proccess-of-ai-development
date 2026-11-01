import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw/UCI_Credit_Card.csv")
OUT_DIR = Path("data/processed")

COLUMNS_TO_DROP = ["ID"]
TARGET = "default.payment.next.month"


def load_raw(path: Path) -> pd.DataFrame:
    """Читаем CSV и сразу убираем служебные колонки"""
    df = pd.read_csv(path)
    df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns], inplace=True)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Удаление дубликатов и подозрительный (по возрасту) значений"""
    df = df.drop_duplicates().reset_index(drop=True)
    # возраст не может быть < 18 или > 100
    df = df[(df["AGE"] >= 18) & (df["AGE"] <= 100)]
    return df


def main() -> None:
    if not RAW_PATH.exists():
        sys.exit(f"[ERROR] Файл {RAW_PATH} не найден")

    df = load_raw(RAW_PATH)
    print(f"Загружено строк: {len(df)}")

    df = clean(df)
    print(f"После очистки: {len(df)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[TARGET]
    )
    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)
    print(f"Train: {len(train_df)}, Test: {len(test_df)} -> сохранено в {OUT_DIR}/")


if __name__ == "__main__":
    main()
