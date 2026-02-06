"""
CRUD Operations for Database
Create, Read, Update, Delete operations for predictions
"""

from sqlalchemy.orm import Session
from typing import List, Optional
from backend.models import Prediction, User
from backend.schemas import PatientData, PredictionResponse
from datetime import datetime


def create_prediction(
    db: Session,
    patient_data: PatientData,
    prediction: int,
    probability: float
) -> Prediction:
    """
    Create a new prediction record in database
    
    Args:
        db: Database session
        patient_data: Patient medical data
        prediction: Model prediction (0 or 1)
        probability: Prediction probability (0.0-1.0)
        
    Returns:
        Created Prediction object
    """
    db_prediction = Prediction(
        age=patient_data.age,
        sex=patient_data.sex,
        cp=patient_data.cp,
        trestbps=patient_data.trestbps,
        chol=patient_data.chol,
        fbs=patient_data.fbs,
        restecg=patient_data.restecg,
        thalach=patient_data.thalach,
        exang=patient_data.exang,
        oldpeak=patient_data.oldpeak,
        slope=patient_data.slope,
        ca=patient_data.ca,
        thal=patient_data.thal,
        prediction=prediction,
        probability=probability
    )
    
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    
    return db_prediction


def get_prediction(db: Session, prediction_id: int) -> Optional[Prediction]:
    """
    Get a prediction record by ID
    
    Args:
        db: Database session
        prediction_id: Prediction record ID
        
    Returns:
        Prediction object or None if not found
    """
    return db.query(Prediction).filter(Prediction.id == prediction_id).first()


def get_all_predictions(
    db: Session,
    skip: int = 0,
    limit: int = 100
) -> List[Prediction]:
    """
    Get all prediction records with pagination
    
    Args:
        db: Database session
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        List of Prediction objects
    """
    return db.query(Prediction).offset(skip).limit(limit).all()


def get_predictions_by_result(
    db: Session,
    prediction_value: int,
    skip: int = 0,
    limit: int = 100
) -> List[Prediction]:
    """
    Get predictions filtered by result (0=no disease, 1=disease)
    
    Args:
        db: Database session
        prediction_value: Prediction value to filter (0 or 1)
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        List of Prediction objects
    """
    return db.query(Prediction).filter(
        Prediction.prediction == prediction_value
    ).offset(skip).limit(limit).all()


def get_prediction_statistics(db: Session) -> dict:
    """
    Get statistics about stored predictions
    
    Args:
        db: Database session
        
    Returns:
        Dictionary with prediction statistics
    """
    total = db.query(Prediction).count()
    disease_count = db.query(Prediction).filter(Prediction.prediction == 1).count()
    no_disease_count = db.query(Prediction).filter(Prediction.prediction == 0).count()
    
    avg_probability = db.query(
        db.func.avg(Prediction.probability)
    ).scalar() or 0.0
    
    return {
        "total_predictions": total,
        "disease_detected": disease_count,
        "no_disease": no_disease_count,
        "disease_rate": (disease_count / total * 100) if total > 0 else 0.0,
        "average_probability": float(avg_probability)
    }


def delete_prediction(db: Session, prediction_id: int) -> bool:
    """
    Delete a prediction record
    
    Args:
        db: Database session
        prediction_id: Prediction record ID to delete
        
    Returns:
        True if deleted, False if not found
    """
    prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if prediction:
        db.delete(prediction)
        db.commit()
        return True
    return False

