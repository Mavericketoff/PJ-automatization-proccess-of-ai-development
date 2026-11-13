import numpy as np
import pandas as pd
import requests

API_URL = "http://localhost:8000/predict"
PSI_BINS = 10


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index между двумя распределениями."""
    expected_pct, edges = np.histogram(expected, bins=bins, density=False)
    actual_pct, _ = np.histogram(actual, bins=edges, density=False)

    expected_pct = expected_pct / expected_pct.sum() + 1e-6
    actual_pct = actual_pct / actual_pct.sum() + 1e-6

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def simulate_new_data(test_path="data/processed/test.csv", n=200):
    """Берём часть тестовой выборки как «новые» данные."""
    df = pd.read_csv(test_path)
    return df.sample(n=min(n, len(df)), random_state=99)


def main():
    new_data = simulate_new_data()
    train_data = pd.read_csv("data/processed/train.csv")

    print(f"Имитируем поступление {len(new_data)} записей...")

    # отправляем на API
    feature_cols = [
        "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    ]
    predictions = []
    for _, row in new_data[feature_cols].iterrows():
        try:
            resp = requests.post(API_URL, json=row.to_dict(), timeout=5)
            if resp.status_code == 200:
                predictions.append(resp.json()["default_probability"])
        except requests.ConnectionError:
            print("[WARN] API недоступен, пропускаю.")
            return

    if not predictions:
        return

    pred_array = np.array(predictions)
    train_target = train_data["default.payment.next.month"].values.astype(float)

    psi = calculate_psi(train_target, pred_array)
    print(f"PSI (default probability): {psi:.4f}")
    if psi < 0.1:
        print("Дрейф не обнаружен")
    elif psi < 0.25:
        print("Умеренный дрейф")
    else:
        print("Сильный дрейф")

    # PSI по ключевому признаку
    psi_age = calculate_psi(train_data["AGE"].values, new_data["AGE"].values)
    print(f"PSI (AGE): {psi_age:.4f}")


if __name__ == "__main__":
    main()
