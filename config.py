"""
System Configuration
Centralized configuration for the entire system
"""

import os
from typing import Optional

# API Configuration
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
API_BASE_URL: str = os.getenv("API_BASE_URL", f"http://localhost:{API_PORT}")
API_TIMEOUT: int = int(os.getenv("API_TIMEOUT", "10"))

# Frontend Configuration
FRONTEND_PORT: int = int(os.getenv("FRONTEND_PORT", "8501"))
FRONTEND_HOST: str = os.getenv("FRONTEND_HOST", "localhost")

# Database Configuration
DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./heart_disease_predictions.db")

# Model Configuration
MODEL_DIR: str = os.getenv("MODEL_DIR", "models")
MODEL_PATHS: list = [
    f"{MODEL_DIR}/federated_model.pth",
    f"{MODEL_DIR}/client_1_model.pth",
    f"{MODEL_DIR}/client_2_model.pth",
    f"{MODEL_DIR}/client_3_model.pth"
]

# Feature Configuration
NUM_FEATURES: int = 13  # Number of features in heart disease dataset

# Logging Configuration
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

def get_api_url() -> str:
    """Get the full API base URL"""
    return API_BASE_URL

def get_health_endpoint() -> str:
    """Get the health check endpoint URL"""
    return f"{API_BASE_URL}/health"

def get_predict_endpoint() -> str:
    """Get the prediction endpoint URL"""
    return f"{API_BASE_URL}/predict"

