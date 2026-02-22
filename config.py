"""
System Configuration
Centralized configuration for the entire system
"""

import os
from pathlib import Path
from typing import Optional

PROJECT_ROOT: Path = Path(__file__).resolve().parent


def _normalize_host(host: str) -> str:
    if host in ("0.0.0.0", "::"):
        return "127.0.0.1"
    return host


def _resolve_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def _resolve_sqlite_url(url: str) -> str:
    if url.startswith("sqlite:///"):
        raw_path = url.replace("sqlite:///", "", 1)
        resolved_path = Path(_resolve_path(raw_path)).resolve().as_posix()
        return f"sqlite:///{resolved_path}"
    return url

# API Configuration
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
API_PUBLIC_HOST: str = os.getenv("API_PUBLIC_HOST", _normalize_host(API_HOST))
API_BASE_URL: str = os.getenv("API_BASE_URL", f"http://{API_PUBLIC_HOST}:{API_PORT}")
API_TIMEOUT: int = int(os.getenv("API_TIMEOUT", "10"))

# Frontend Configuration
FRONTEND_PORT: int = int(os.getenv("FRONTEND_PORT", "8501"))
FRONTEND_HOST: str = os.getenv("FRONTEND_HOST", "localhost")

# Database Configuration
DATABASE_URL: str = _resolve_sqlite_url(
    os.getenv("DATABASE_URL", f"sqlite:///heart_disease_predictions.db")
)

# Model Configuration
MODEL_DIR: str = _resolve_path(os.getenv("MODEL_DIR", str(Path("backend") / "models")))
MODEL_PATH: str = _resolve_path(
    os.getenv("MODEL_PATH", str(Path("backend") / "models" / "heart_model.pkl"))
)

# Feature Configuration
NUM_FEATURES: int = 13  # Number of features in heart disease dataset

# Logging Configuration
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

def get_api_url() -> str:
    """Get the full API base URL"""
    return API_BASE_URL


def get_public_api_url() -> str:
    """Get the public API base URL for browsers"""
    return f"http://{API_PUBLIC_HOST}:{API_PORT}"

def get_health_endpoint() -> str:
    """Get the health check endpoint URL"""
    return f"{API_BASE_URL}/health"

def get_predict_endpoint() -> str:
    """Get the prediction endpoint URL"""
    return f"{API_BASE_URL}/predict"


def get_dataset_path() -> Optional[str]:
    """Return the first existing dataset path, if any"""
    candidates = [
        _resolve_path("dataset/raw/heart.csv"),
        _resolve_path("dataset/heart.csv")
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]

