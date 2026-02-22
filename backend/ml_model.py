"""
ML model loading and prediction utilities for the backend.
"""

from typing import Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import config
from backend.schemas import PatientData

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]

_model: Optional[Pipeline] = None
_model_loaded: bool = False


class _FallbackModel:
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.array([[0.5, 0.5]], dtype=float)


def _get_model_path() -> Path:
    return Path(config.MODEL_PATH)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path)


def _find_target_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["target", "label", "output", "heart_disease", "disease"]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _train_fallback_model(model_path: Path) -> Optional[Pipeline]:
    dataset_path = config.get_dataset_path()
    if not dataset_path or not Path(dataset_path).exists():
        print("[WARNING] Dataset not found. Skipping fallback training.")
        return None

    df = pd.read_csv(dataset_path)
    target_col = _find_target_column(df)
    if target_col is None:
        print("[WARNING] Target column not found in dataset. Skipping fallback training.")
        return None

    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        print(f"[WARNING] Dataset missing required features: {missing}")
        return None

    X = df[FEATURE_COLUMNS].astype(float)
    y = df[target_col].astype(int)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X, y)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"[OK] Trained fallback logistic regression model and saved to: {_display_path(model_path)}")
    return pipeline


def load_model() -> None:
    global _model, _model_loaded

    if _model_loaded:
        return

    _model_loaded = True
    model_path = _get_model_path()
    if model_path.exists():
        try:
            _model = joblib.load(model_path)
            print(f"[OK] Model loaded from: {_display_path(model_path)}")
            return
        except Exception as exc:
            print(f"[WARNING] Failed to load model from {_display_path(model_path)}: {exc}")

    print("[WARNING] No saved model found.")
    print(f"[WARNING] Expected model at: {_display_path(model_path)}")
    _model = _train_fallback_model(model_path)

    if _model is None:
        print("[WARNING] Using fallback model with neutral probability output.")
        _model = _FallbackModel()


def is_model_loaded() -> bool:
    return _model is not None


def preprocess_patient_data(patient_data: PatientData) -> np.ndarray:
    return np.array([[
        patient_data.age,
        patient_data.sex,
        patient_data.cp,
        patient_data.trestbps,
        patient_data.chol,
        patient_data.fbs,
        patient_data.restecg,
        patient_data.thalach,
        patient_data.exang,
        patient_data.oldpeak,
        patient_data.slope,
        patient_data.ca,
        patient_data.thal
    ]], dtype=float)


def predict_heart_disease(patient_data: PatientData) -> Tuple[int, float]:
    if _model is None:
        load_model()

    if _model is None:
        raise RuntimeError("Model not loaded")

    features = preprocess_patient_data(patient_data)

    try:
        proba = _model.predict_proba(features)[0][1]
    except Exception:
        proba = 0.5

    probability = float(proba)
    prediction = 1 if probability > 0.5 else 0
    return prediction, probability
