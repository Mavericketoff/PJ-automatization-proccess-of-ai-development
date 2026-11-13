from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/credit_default_model.pkl")

app = FastAPI(title="CDP API", version="1.0")

_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError("Модель не найдена. Запустите `dvc repro` сначала.")
        _model = joblib.load(MODEL_PATH)
    return _model


class ClientFeatures(BaseModel):
    LIMIT_BAL: float = Field(..., gt=0, description="Кредитный лимит")
    SEX: int = Field(..., ge=1, le=2)
    EDUCATION: int = Field(..., ge=0, le=6)
    MARRIAGE: int = Field(..., ge=0, le=3)
    AGE: int = Field(..., ge=18, le=100)
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float = 0.0
    BILL_AMT4: float = 0.0
    BILL_AMT5: float = 0.0
    BILL_AMT6: float = 0.0
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float = 0.0
    PAY_AMT4: float = 0.0
    PAY_AMT5: float = 0.0
    PAY_AMT6: float = 0.0


@app.get("/")
def health():
    return {"status": "ok", "service": "cdp"}


@app.post("/predict")
def predict(features: ClientFeatures):
    try:
        model = get_model()
        row = pd.DataFrame([features.model_dump()])

        from src.features.build_features import add_features
        row = add_features(row)

        proba = float(model.predict_proba(row)[:, 1][0])
        label = int(model.predict(row)[0])
        return {"default_prediction": label, "default_probability": round(proba, 4)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
