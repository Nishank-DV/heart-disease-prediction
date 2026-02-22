"""
FastAPI Application - Heart Disease Prediction API
REST API for heart disease prediction using trained federated learning model
"""

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import List
from contextlib import asynccontextmanager
import os
import sys
from pydantic import ValidationError

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.database import get_db, init_db
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
    get_prediction_statistics,
    delete_prediction
)
from backend.ml_model import load_model, is_model_loaded, predict_heart_disease


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan event handler for startup and shutdown."""
    init_db()
    print("[OK] Database tables created")

    load_model()
    print("[OK] API startup complete")
    yield


app = FastAPI(
    title="Heart Disease Prediction API",
    description="REST API for heart disease prediction using Federated Learning model",
    version="1.0.0",
    lifespan=lifespan
)

templates = Jinja2Templates(directory=os.path.join(project_root, "frontend", "templates"))
static_dir = os.path.join(project_root, "frontend", "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# CORS middleware (allow cross-origin requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/ui/predict", response_class=HTMLResponse, tags=["UI"])
async def ui_predict(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})


@app.post("/ui/predict", response_class=HTMLResponse, tags=["UI"])
async def ui_predict_submit(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    data = {key: form.get(key) for key in form.keys()}

    try:
        patient_data = PatientData(**data)
        prediction, probability = predict_heart_disease(patient_data)

        if probability >= 0.7:
            risk_level = "High"
            prediction_text = "Heart disease detected - High risk"
        elif probability >= 0.5:
            risk_level = "Medium"
            prediction_text = "Heart disease possible - Medium risk"
        else:
            risk_level = "Low"
            prediction_text = "No heart disease detected - Low risk"

        db_prediction = create_prediction(
            db=db,
            patient_data=patient_data,
            prediction=prediction,
            probability=probability
        )

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "prediction_text": prediction_text,
                "risk_level": risk_level,
                "probability": round(probability * 100, 2),
                "record_id": db_prediction.id
            }
        )
    except ValidationError:
        return templates.TemplateResponse(
            "predict.html",
            {
                "request": request,
                "error": "Invalid input. Please check the highlighted fields."
            },
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    except Exception as exc:
        return templates.TemplateResponse(
            "predict.html",
            {
                "request": request,
                "error": f"Prediction failed: {exc}"
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


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
        db.execute(text("SELECT 1"))
        db_connected = True
    except Exception:
        db_connected = False
    
    # Check model status
    model_loaded = is_model_loaded()
    
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
        
    except HTTPException as e:
        raise e
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


@app.get("/predictions/stats", tags=["Predictions"])
async def get_stats(db: Session = Depends(get_db)):
    """
    Get prediction statistics
    
    Returns:
        Dictionary with prediction statistics
    """
    stats = get_prediction_statistics(db)
    return stats


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


