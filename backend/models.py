"""
SQLAlchemy Database Models
Defines database tables and relationships
"""

from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.sql import func
from backend.database import Base


class Prediction(Base):
    """
    Prediction table - stores patient data and prediction results
    
    This table stores:
    - Patient medical features (age, blood pressure, cholesterol, etc.)
    - Model prediction (0 = no disease, 1 = heart disease)
    - Prediction probability (confidence score)
    - Timestamp of prediction
    """
    __tablename__ = "predictions"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Patient medical features (13 features from heart disease dataset)
    age = Column(Integer, nullable=False, comment="Patient age in years")
    sex = Column(Integer, nullable=False, comment="Sex (0=female, 1=male)")
    cp = Column(Integer, nullable=False, comment="Chest pain type (0-3)")
    trestbps = Column(Integer, nullable=False, comment="Resting blood pressure (mm Hg)")
    chol = Column(Integer, nullable=False, comment="Serum cholesterol (mg/dl)")
    fbs = Column(Integer, nullable=False, comment="Fasting blood sugar > 120 (0/1)")
    restecg = Column(Integer, nullable=False, comment="Resting ECG results (0-2)")
    thalach = Column(Integer, nullable=False, comment="Maximum heart rate achieved")
    exang = Column(Integer, nullable=False, comment="Exercise induced angina (0/1)")
    oldpeak = Column(Float, nullable=False, comment="ST depression induced by exercise")
    slope = Column(Integer, nullable=False, comment="Slope of peak exercise ST segment (0-2)")
    ca = Column(Integer, nullable=False, comment="Number of major vessels (0-3)")
    thal = Column(Integer, nullable=False, comment="Thalassemia type (0-3)")
    
    # Prediction results
    prediction = Column(Integer, nullable=False, comment="Prediction: 0=no disease, 1=heart disease")
    probability = Column(Float, nullable=False, comment="Prediction probability (0.0-1.0)")
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<Prediction(id={self.id}, prediction={self.prediction}, probability={self.probability:.2f})>"


class User(Base):
    """
    User table - stores user information (optional, for future use)
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username})>"

