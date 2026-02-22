"""
Pydantic Schemas for Request/Response Validation
Defines data structures for API requests and responses
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional
from datetime import datetime


class PatientData(BaseModel):
    """
    Schema for patient medical data input
    
    Validates all 13 features required for heart disease prediction
    """
    age: int = Field(..., ge=0, le=120, description="Patient age in years (0-120)")
    sex: int = Field(..., ge=0, le=1, description="Sex: 0=female, 1=male")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type: 0-3")
    trestbps: int = Field(..., ge=0, le=300, description="Resting blood pressure in mm Hg")
    chol: int = Field(..., ge=0, le=600, description="Serum cholesterol in mg/dl")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120: 0=no, 1=yes")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results: 0-2")
    thalach: int = Field(..., ge=0, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina: 0=no, 1=yes")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression (0.0-10.0)")
    slope: int = Field(..., ge=0, le=2, description="ST slope: 0-2")
    ca: int = Field(..., ge=0, le=3, description="Number of major vessels: 0-3")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia type: 0-3")
    
    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        """Validate age is realistic for medical data"""
        if v < 1 or v > 120:
            raise ValueError("Age must be between 1 and 120 years")
        return v
    
    @field_validator('trestbps')
    @classmethod
    def validate_blood_pressure(cls, v):
        """Validate blood pressure is in medical range"""
        if v < 50 or v > 250:
            raise ValueError("Blood pressure must be between 50 and 250 mm Hg")
        return v
    
    @field_validator('chol')
    @classmethod
    def validate_cholesterol(cls, v):
        """Validate cholesterol is in medical range"""
        if v < 100 or v > 600:
            raise ValueError("Cholesterol must be between 100 and 600 mg/dl")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1
            }
        }
    )


class PredictionResponse(BaseModel):
    """
    Schema for prediction API response
    """
    prediction: int = Field(..., description="Prediction: 0=no disease, 1=heart disease")
    probability: float = Field(..., ge=0.0, le=1.0, description="Prediction probability (0.0-1.0)")
    prediction_text: str = Field(..., description="Human-readable prediction")
    risk_level: str = Field(..., description="Risk level: Low, Medium, High")
    record_id: Optional[int] = Field(None, description="Database record ID")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": 1,
                "probability": 0.85,
                "prediction_text": "Heart disease detected",
                "risk_level": "High",
                "record_id": 1
            }
        }
    )


class PredictionRecord(BaseModel):
    """
    Schema for stored prediction record (database model representation)
    """
    id: int
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int
    prediction: int
    probability: float
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class HealthResponse(BaseModel):
    """
    Schema for health check response
    """
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    database_connected: bool = Field(..., description="Whether database is connected")
    message: str = Field(..., description="Status message")

