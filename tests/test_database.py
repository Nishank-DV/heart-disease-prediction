"""
Database Tests
Tests for database operations and data integrity
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import Base, get_db
from backend.models import Prediction
from backend.schemas import PatientData
from backend.crud import (
    create_prediction,
    get_prediction,
    get_all_predictions,
    get_prediction_statistics,
    delete_prediction
)


# Create test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_db.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def db_session():
    """Create a test database session"""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


class TestDatabase:
    """Test cases for database operations"""
    
    def test_create_prediction(self, db_session):
        """Test 1: Create prediction record"""
        patient_data = PatientData(
            age=63,
            sex=1,
            cp=3,
            trestbps=145,
            chol=233,
            fbs=1,
            restecg=0,
            thalach=150,
            exang=0,
            oldpeak=2.3,
            slope=0,
            ca=0,
            thal=1
        )
        
        prediction = create_prediction(
            db=db_session,
            patient_data=patient_data,
            prediction=1,
            probability=0.85
        )
        
        assert prediction.id is not None
        assert prediction.prediction == 1
        assert prediction.probability == 0.85
        assert prediction.age == 63
    
    def test_get_prediction(self, db_session):
        """Test 2: Retrieve prediction by ID"""
        # Create a prediction
        patient_data = PatientData(
            age=63, sex=1, cp=3, trestbps=145, chol=233, fbs=1,
            restecg=0, thalach=150, exang=0, oldpeak=2.3, slope=0, ca=0, thal=1
        )
        
        created = create_prediction(db_session, patient_data, 1, 0.85)
        
        # Retrieve it
        retrieved = get_prediction(db_session, created.id)
        
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.prediction == 1
    
    def test_get_all_predictions(self, db_session):
        """Test 3: Get all predictions with pagination"""
        # Create multiple predictions
        for i in range(5):
            patient_data = PatientData(
                age=50+i, sex=1, cp=1, trestbps=140, chol=200, fbs=0,
                restecg=0, thalach=150, exang=0, oldpeak=1.0, slope=0, ca=0, thal=1
            )
            create_prediction(db_session, patient_data, i % 2, 0.5 + i * 0.1)
        
        # Get all
        predictions = get_all_predictions(db_session, skip=0, limit=10)
        assert len(predictions) == 5
        
        # Test pagination
        predictions_page1 = get_all_predictions(db_session, skip=0, limit=2)
        assert len(predictions_page1) == 2
        
        predictions_page2 = get_all_predictions(db_session, skip=2, limit=2)
        assert len(predictions_page2) == 2
    
    def test_get_predictions_by_result(self, db_session):
        """Test 4: Filter predictions by result"""
        # Create predictions with different results
        for i in range(5):
            patient_data = PatientData(
                age=50, sex=1, cp=1, trestbps=140, chol=200, fbs=0,
                restecg=0, thalach=150, exang=0, oldpeak=1.0, slope=0, ca=0, thal=1
            )
            create_prediction(db_session, patient_data, i % 2, 0.5)
        
        # Get only disease predictions
        disease_predictions = get_predictions_by_result(db_session, prediction_value=1)
        assert len(disease_predictions) == 3  # 0, 2, 4 are disease
        
        # Get only no-disease predictions
        no_disease = get_predictions_by_result(db_session, prediction_value=0)
        assert len(no_disease) == 2  # 1, 3 are no disease
    
    def test_prediction_statistics(self, db_session):
        """Test 5: Get prediction statistics"""
        # Create predictions
        for i in range(10):
            patient_data = PatientData(
                age=50, sex=1, cp=1, trestbps=140, chol=200, fbs=0,
                restecg=0, thalach=150, exang=0, oldpeak=1.0, slope=0, ca=0, thal=1
            )
            create_prediction(db_session, patient_data, i % 2, 0.5)
        
        stats = get_prediction_statistics(db_session)
        
        assert stats["total_predictions"] == 10
        assert stats["disease_detected"] == 5
        assert stats["no_disease"] == 5
        assert stats["disease_rate"] == 50.0
    
    def test_delete_prediction(self, db_session):
        """Test 6: Delete prediction record"""
        # Create a prediction
        patient_data = PatientData(
            age=63, sex=1, cp=3, trestbps=145, chol=233, fbs=1,
            restecg=0, thalach=150, exang=0, oldpeak=2.3, slope=0, ca=0, thal=1
        )
        
        created = create_prediction(db_session, patient_data, 1, 0.85)
        prediction_id = created.id
        
        # Delete it
        deleted = delete_prediction(db_session, prediction_id)
        assert deleted is True
        
        # Verify it's deleted
        retrieved = get_prediction(db_session, prediction_id)
        assert retrieved is None
    
    def test_data_integrity(self, db_session):
        """Test 7: Data integrity - all fields stored correctly"""
        patient_data = PatientData(
            age=63,
            sex=1,
            cp=3,
            trestbps=145,
            chol=233,
            fbs=1,
            restecg=0,
            thalach=150,
            exang=0,
            oldpeak=2.3,
            slope=0,
            ca=0,
            thal=1
        )
        
        prediction = create_prediction(db_session, patient_data, 1, 0.85)
        
        # Verify all fields
        assert prediction.age == 63
        assert prediction.sex == 1
        assert prediction.cp == 3
        assert prediction.trestbps == 145
        assert prediction.chol == 233
        assert prediction.fbs == 1
        assert prediction.restecg == 0
        assert prediction.thalach == 150
        assert prediction.exang == 0
        assert prediction.oldpeak == 2.3
        assert prediction.slope == 0
        assert prediction.ca == 0
        assert prediction.thal == 1
        assert prediction.prediction == 1
        assert prediction.probability == 0.85
        assert prediction.created_at is not None
    
    def test_timestamp_auto_generated(self, db_session):
        """Test 8: Timestamp is automatically generated"""
        patient_data = PatientData(
            age=63, sex=1, cp=3, trestbps=145, chol=233, fbs=1,
            restecg=0, thalach=150, exang=0, oldpeak=2.3, slope=0, ca=0, thal=1
        )
        
        prediction = create_prediction(db_session, patient_data, 1, 0.85)
        
        assert prediction.created_at is not None
        from datetime import datetime
        assert isinstance(prediction.created_at, datetime)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

