"""
FastAPI Application - Heart Disease Prediction API
REST API for heart disease prediction using trained federated learning model
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import torch
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from backend.database import get_db, init_db, engine
    from backend.models import Base, Prediction
    from backend.schemas import (
        PatientData,
        PredictionResponse,
        PredictionRecord,
        HealthResponse
    )
    from backend.crud import (
        create_prediction,
        get_prediction,
        get_all_predictions,
        get_predictions_by_result,
        get_prediction_statistics,
        delete_prediction
    )
except ImportError:
    # Handle relative imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.database import get_db, init_db, engine
    from backend.models import Base, Prediction
    from backend.schemas import (
        PatientData,
        PredictionResponse,
        PredictionRecord,
        HealthResponse
    )
    from backend.crud import (
        create_prediction,
        get_prediction,
        get_all_predictions,
        get_predictions_by_result,
        get_prediction_statistics,
        delete_prediction
    )
from client.model import HeartDiseaseMLP
from client.data_preprocessing import DataPreprocessor
from sklearn.preprocessing import StandardScaler
import pickle

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="REST API for heart disease prediction using Federated Learning model",
    version="1.0.0"
)

# CORS middleware (allow cross-origin requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessor
model = None
scaler = None
num_features = 13  # Number of features in heart disease dataset


def load_model():
    """
    Load the trained federated learning model
    This function is called at application startup
    """
    global model, scaler
    
    # Try to load federated model first, then fallback to local model
    model_paths = [
        "models/federated_model.pth",
        "models/client_1_model.pth",
        "models/client_2_model.pth",
        "models/client_3_model.pth"
    ]
    
    model_loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                model = HeartDiseaseMLP(input_size=num_features)
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                print(f"[OK] Model loaded from: {model_path}")
                model_loaded = True
                break
            except Exception as e:
                print(f"[WARNING] Failed to load {model_path}: {e}")
                continue
    
    if not model_loaded:
        # Create a new model if no saved model found
        print("[WARNING] No saved model found. Using randomly initialized model.")
        print("[WARNING] For accurate predictions, train a model first using Phase 2.")
        model = HeartDiseaseMLP(input_size=num_features)
        model.eval()
    
    # Initialize scaler (in production, load from saved scaler)
    scaler = StandardScaler()
    print("[OK] Model and scaler initialized")


@app.on_event("startup")
async def startup_event():
    """Initialize database and load model on application startup"""
    # Create database tables
    Base.metadata.create_all(bind=engine)
    print("[OK] Database tables created")
    
    # Load model
    load_model()
    
    print("[OK] API startup complete")


def preprocess_patient_data(patient_data: PatientData) -> torch.Tensor:
    """
    Preprocess patient data for model prediction
    
    Args:
        patient_data: Patient medical data
        
    Returns:
        Preprocessed tensor ready for model input
    """
    # Convert to numpy array
    features = np.array([[
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
    ]])
    
    # Normalize features (using scaler - in production, load saved scaler)
    # For now, we'll use a simple normalization
    # In production, use the same scaler that was used during training
    features_normalized = features.astype(np.float32)
    
    # Convert to tensor
    features_tensor = torch.FloatTensor(features_normalized)
    
    return features_tensor


def predict_heart_disease(patient_data: PatientData) -> tuple:
    """
    Predict heart disease for given patient data
    
    Args:
        patient_data: Patient medical data
        
    Returns:
        Tuple of (prediction, probability)
        - prediction: 0 (no disease) or 1 (disease)
        - probability: Confidence score (0.0-1.0)
    """
    global model
    
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please ensure model is trained and saved."
        )
    
    # Preprocess data
    features_tensor = preprocess_patient_data(patient_data)
    
    # Make prediction
    with torch.no_grad():
        output = model(features_tensor)
        probability = output.item()
    
    # Convert probability to binary prediction (threshold = 0.5)
    prediction = 1 if probability > 0.5 else 0
    
    return prediction, probability


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint
    
    Returns:
        API health status, model status, and database connection status
    """
    # Check database connection
    db_connected = False
    try:
        db.execute("SELECT 1")
        db_connected = True
    except:
        db_connected = False
    
    # Check model status
    model_loaded = model is not None
    
    status_msg = "API is healthy"
    if not model_loaded:
        status_msg = "API is running but model is not loaded"
    if not db_connected:
        status_msg = "API is running but database connection failed"
    
    return HealthResponse(
        status="healthy" if (model_loaded and db_connected) else "degraded",
        model_loaded=model_loaded,
        database_connected=db_connected,
        message=status_msg
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    patient_data: PatientData,
    db: Session = Depends(get_db)
):
    """
    Predict heart disease for given patient data
    
    This endpoint:
    1. Validates patient medical data
    2. Preprocesses the data
    3. Makes prediction using trained model
    4. Stores prediction in database
    5. Returns prediction result
    
    Args:
        patient_data: Patient medical features (13 features)
        db: Database session
        
    Returns:
        Prediction result with probability and risk level
    """
    try:
        # Make prediction
        prediction, probability = predict_heart_disease(patient_data)
        
        # Determine risk level
        if probability >= 0.7:
            risk_level = "High"
            prediction_text = "Heart disease detected - High risk"
        elif probability >= 0.5:
            risk_level = "Medium"
            prediction_text = "Heart disease possible - Medium risk"
        else:
            risk_level = "Low"
            prediction_text = "No heart disease detected - Low risk"
        
        # Store prediction in database
        db_prediction = create_prediction(
            db=db,
            patient_data=patient_data,
            prediction=prediction,
            probability=probability
        )
        
        return PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            prediction_text=prediction_text,
            risk_level=risk_level,
            record_id=db_prediction.id
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input data: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/predictions", response_model=List[PredictionRecord], tags=["Predictions"])
async def get_predictions(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Get all stored predictions with pagination
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session
        
    Returns:
        List of prediction records
    """
    predictions = get_all_predictions(db, skip=skip, limit=limit)
    return predictions


@app.get("/predictions/{prediction_id}", response_model=PredictionRecord, tags=["Predictions"])
async def get_prediction_by_id(
    prediction_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a specific prediction record by ID
    
    Args:
        prediction_id: Prediction record ID
        db: Database session
        
    Returns:
        Prediction record
    """
    prediction = get_prediction(db, prediction_id)
    if prediction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction with ID {prediction_id} not found"
        )
    return prediction


@app.get("/predictions/stats", tags=["Predictions"])
async def get_stats(db: Session = Depends(get_db)):
    """
    Get prediction statistics
    
    Returns:
        Dictionary with prediction statistics
    """
    stats = get_prediction_statistics(db)
    return stats


@app.delete("/predictions/{prediction_id}", tags=["Predictions"])
async def delete_prediction_by_id(
    prediction_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a prediction record
    
    Args:
        prediction_id: Prediction record ID to delete
        db: Database session
        
    Returns:
        Success message
    """
    deleted = delete_prediction(db, prediction_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction with ID {prediction_id} not found"
        )
    return {"message": f"Prediction {prediction_id} deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

