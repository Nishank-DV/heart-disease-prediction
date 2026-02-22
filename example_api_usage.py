"""
Example: Using the Heart Disease Prediction API
Demonstrates how to interact with the REST API
"""

import requests
import json
from typing import Dict, Any
import config

# API base URL
API_BASE_URL = config.get_public_api_url()


def check_api_health() -> bool:
    """Check if API is running and healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("=" * 60)
            print("API HEALTH CHECK")
            print("=" * 60)
            print(f"Status: {data['status']}")
            print(f"Model Loaded: {data['model_loaded']}")
            print(f"Database Connected: {data['database_connected']}")
            print(f"Message: {data['message']}")
            print("=" * 60)
            return True
        else:
            print(f"API returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to API.")
        print("   Make sure the API server is running:")
        print("   python run_api.py")
        return False


def predict_heart_disease(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make a heart disease prediction
    
    Args:
        patient_data: Dictionary with patient medical features
        
    Returns:
        Prediction result dictionary
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=patient_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   {response.text}")
            return {}
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return {}


def get_all_predictions(limit: int = 10) -> list:
    """Get all stored predictions"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/predictions",
            params={"skip": 0, "limit": limit}
        )
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        print(f"❌ Error getting predictions: {e}")
        return []


def get_prediction_by_id(prediction_id: int) -> Dict[str, Any]:
    """Get a specific prediction by ID"""
    try:
        response = requests.get(f"{API_BASE_URL}/predictions/{prediction_id}")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Prediction {prediction_id} not found")
            return {}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {}


def get_statistics() -> Dict[str, Any]:
    """Get prediction statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/predictions/stats")
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {}


def main():
    """Main example function"""
    print("\n" + "=" * 60)
    print("HEART DISEASE PREDICTION API - EXAMPLE USAGE")
    print("=" * 60 + "\n")
    
    # 1. Check API health
    if not check_api_health():
        return
    
    print("\n")
    
    # 2. Example patient data (high risk)
    print("=" * 60)
    print("EXAMPLE 1: High Risk Patient")
    print("=" * 60)
    
    high_risk_patient = {
        "age": 70,
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
    
    result = predict_heart_disease(high_risk_patient)
    if result:
        print(f"\nPrediction Result:")
        print(f"  Prediction: {result['prediction']} ({'Disease' if result['prediction'] == 1 else 'No Disease'})")
        print(f"  Probability: {result['probability']:.2%}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Message: {result['prediction_text']}")
        print(f"  Record ID: {result['record_id']}")
    
    print("\n")
    
    # 3. Example patient data (low risk)
    print("=" * 60)
    print("EXAMPLE 2: Low Risk Patient")
    print("=" * 60)
    
    low_risk_patient = {
        "age": 35,
        "sex": 0,
        "cp": 0,
        "trestbps": 120,
        "chol": 180,
        "fbs": 0,
        "restecg": 0,
        "thalach": 180,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 0,
        "ca": 0,
        "thal": 0
    }
    
    result = predict_heart_disease(low_risk_patient)
    if result:
        print(f"\nPrediction Result:")
        print(f"  Prediction: {result['prediction']} ({'Disease' if result['prediction'] == 1 else 'No Disease'})")
        print(f"  Probability: {result['probability']:.2%}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Message: {result['prediction_text']}")
        print(f"  Record ID: {result['record_id']}")
    
    print("\n")
    
    # 4. Get statistics
    print("=" * 60)
    print("PREDICTION STATISTICS")
    print("=" * 60)
    
    stats = get_statistics()
    if stats:
        print(f"\nTotal Predictions: {stats['total_predictions']}")
        print(f"Disease Detected: {stats['disease_detected']}")
        print(f"No Disease: {stats['no_disease']}")
        print(f"Disease Rate: {stats['disease_rate']:.2f}%")
        print(f"Average Probability: {stats['average_probability']:.4f}")
    
    print("\n")
    
    # 5. Get recent predictions
    print("=" * 60)
    print("RECENT PREDICTIONS")
    print("=" * 60)
    
    predictions = get_all_predictions(limit=5)
    if predictions:
        print(f"\nFound {len(predictions)} predictions:")
        for pred in predictions:
            print(f"\n  ID: {pred['id']}")
            print(f"  Age: {pred['age']}, Prediction: {pred['prediction']}")
            print(f"  Probability: {pred['probability']:.2%}")
            print(f"  Created: {pred['created_at']}")
    else:
        print("\nNo predictions found.")
    
    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

