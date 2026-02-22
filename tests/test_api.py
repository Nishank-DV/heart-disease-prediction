"""
API Integration Tests
Tests for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app import app
from backend.database import Base, get_db

# Create test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_predictions.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)


def override_get_db():
    """Override database dependency for testing"""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

# Create test client
client = TestClient(app)


class TestAPI:
    """Test cases for API endpoints"""
    
    def test_root_endpoint(self):
        """Test 1: Root endpoint returns UI HTML"""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "Heart Disease" in response.text
    
    def test_health_check(self):
        """Test 2: Health check endpoint works"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "database_connected" in data
    
    def test_predict_success(self):
        """Test 3: Prediction endpoint with valid data"""
        patient_data = {
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
        
        response = client.post("/predict", json=patient_data)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "prediction_text" in data
        assert "risk_level" in data
        assert data["prediction"] in [0, 1]
        assert 0.0 <= data["probability"] <= 1.0
    
    def test_predict_invalid_age(self):
        """Test 4: Prediction with invalid age (too high)"""
        patient_data = {
            "age": 150,  # Invalid: too high
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
        
        response = client.post("/predict", json=patient_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_invalid_blood_pressure(self):
        """Test 5: Prediction with invalid blood pressure"""
        patient_data = {
            "age": 63,
            "sex": 1,
            "cp": 3,
            "trestbps": 500,  # Invalid: too high
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
        
        response = client.post("/predict", json=patient_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_missing_field(self):
        """Test 6: Prediction with missing required field"""
        patient_data = {
            "age": 63,
            "sex": 1,
            # Missing 'cp' field
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
        
        response = client.post("/predict", json=patient_data)
        assert response.status_code == 422  # Validation error
    
    def test_get_predictions(self):
        """Test 7: Get all predictions"""
        response = client.get("/predictions")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_prediction_by_id(self):
        """Test 8: Get prediction by ID"""
        # First create a prediction
        patient_data = {
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
        
        create_response = client.post("/predict", json=patient_data)
        assert create_response.status_code == 200
        record_id = create_response.json()["record_id"]
        
        # Get the prediction
        response = client.get(f"/predictions/{record_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == record_id
    
    def test_get_prediction_not_found(self):
        """Test 9: Get non-existent prediction"""
        response = client.get("/predictions/99999")
        assert response.status_code == 404
    
    def test_get_statistics(self):
        """Test 10: Get prediction statistics"""
        response = client.get("/predictions/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "disease_detected" in data
        assert "no_disease" in data
    
    def test_delete_prediction(self):
        """Test 11: Delete prediction"""
        # Create a prediction first
        patient_data = {
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
        
        create_response = client.post("/predict", json=patient_data)
        record_id = create_response.json()["record_id"]
        
        # Delete the prediction
        response = client.delete(f"/predictions/{record_id}")
        assert response.status_code == 200
        
        # Verify it's deleted
        get_response = client.get(f"/predictions/{record_id}")
        assert get_response.status_code == 404
    
    def test_medical_edge_cases(self):
        """Test 12: Medical edge cases"""
        edge_cases = [
            {
                "name": "Very young patient",
                "data": {
                    "age": 25,
                    "sex": 0,
                    "cp": 0,
                    "trestbps": 110,
                    "chol": 180,
                    "fbs": 0,
                    "restecg": 0,
                    "thalach": 200,
                    "exang": 0,
                    "oldpeak": 0.0,
                    "slope": 0,
                    "ca": 0,
                    "thal": 0
                }
            },
            {
                "name": "Very old patient",
                "data": {
                    "age": 80,
                    "sex": 1,
                    "cp": 2,
                    "trestbps": 180,
                    "chol": 300,
                    "fbs": 1,
                    "restecg": 1,
                    "thalach": 120,
                    "exang": 1,
                    "oldpeak": 4.0,
                    "slope": 2,
                    "ca": 3,
                    "thal": 2
                }
            },
            {
                "name": "High risk patient",
                "data": {
                    "age": 65,
                    "sex": 1,
                    "cp": 3,
                    "trestbps": 200,
                    "chol": 400,
                    "fbs": 1,
                    "restecg": 2,
                    "thalach": 100,
                    "exang": 1,
                    "oldpeak": 6.0,
                    "slope": 2,
                    "ca": 3,
                    "thal": 3
                }
            }
        ]
        
        for case in edge_cases:
            response = client.post("/predict", json=case["data"])
            assert response.status_code == 200, f"Failed for {case['name']}"
            data = response.json()
            assert "prediction" in data
            assert "probability" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

