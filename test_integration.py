"""
Integration Test Script
Tests the complete system: Frontend → Backend → Database → Model
"""

import requests
import time
import sys
import os
from config import API_BASE_URL, get_health_endpoint, get_predict_endpoint

def test_api_health():
    """Test 1: API Health Check"""
    print("=" * 60)
    print("TEST 1: API Health Check")
    print("=" * 60)
    
    try:
        response = requests.get(get_health_endpoint(), timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is healthy")
            print(f"   Status: {data.get('status')}")
            print(f"   Model Loaded: {data.get('model_loaded')}")
            print(f"   Database Connected: {data.get('database_connected')}")
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is the backend running?")
        print(f"   Expected URL: {get_health_endpoint()}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_valid_prediction():
    """Test 2: Valid Input → Correct Prediction"""
    print("\n" + "=" * 60)
    print("TEST 2: Valid Input → Correct Prediction")
    print("=" * 60)
    
    # High risk patient data
    patient_data = {
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
    
    try:
        print("Sending prediction request...")
        response = requests.post(
            get_predict_endpoint(),
            json=patient_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"   Prediction: {result.get('prediction')} ({'Disease' if result.get('prediction') == 1 else 'No Disease'})")
            print(f"   Probability: {result.get('probability'):.2%}")
            print(f"   Risk Level: {result.get('risk_level')}")
            print(f"   Record ID: {result.get('record_id')}")
            
            # Verify result structure
            required_fields = ['prediction', 'probability', 'risk_level', 'prediction_text', 'record_id']
            if all(field in result for field in required_fields):
                print("✅ All required fields present")
                return True
            else:
                print("❌ Missing required fields")
                return False
        else:
            print(f"❌ API returned status code: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_missing_input():
    """Test 3: Missing Input → Validation Error"""
    print("\n" + "=" * 60)
    print("TEST 3: Missing Input → Validation Error")
    print("=" * 60)
    
    # Incomplete patient data (missing 'age')
    incomplete_data = {
        "sex": 1,
        "cp": 3,
        "trestbps": 200,
        # Missing 'age' field
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
    
    try:
        print("Sending incomplete data...")
        response = requests.post(
            get_predict_endpoint(),
            json=incomplete_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 422:
            print("✅ Validation error correctly returned (422)")
            error_data = response.json()
            print(f"   Error detail: {error_data.get('detail', 'N/A')}")
            return True
        else:
            print(f"❌ Expected 422, got {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_invalid_input():
    """Test 4: Invalid Input → Validation Error"""
    print("\n" + "=" * 60)
    print("TEST 4: Invalid Input → Validation Error")
    print("=" * 60)
    
    # Invalid patient data (age out of range)
    invalid_data = {
        "age": 200,  # Invalid: too high
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
    
    try:
        print("Sending invalid data (age=200)...")
        response = requests.post(
            get_predict_endpoint(),
            json=invalid_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 422:
            print("✅ Validation error correctly returned (422)")
            error_data = response.json()
            print(f"   Error detail: {error_data.get('detail', 'N/A')}")
            return True
        else:
            print(f"❌ Expected 422, got {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_database_storage():
    """Test 5: Database Storage"""
    print("\n" + "=" * 60)
    print("TEST 5: Database Storage")
    print("=" * 60)
    
    # Make a prediction
    patient_data = {
        "age": 50,
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
    
    try:
        response = requests.post(
            get_predict_endpoint(),
            json=patient_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            record_id = result.get('record_id')
            
            if record_id:
                print(f"✅ Prediction stored in database")
                print(f"   Record ID: {record_id}")
                
                # Try to retrieve the record
                get_response = requests.get(
                    f"{API_BASE_URL}/predictions/{record_id}",
                    timeout=5
                )
                
                if get_response.status_code == 200:
                    print("✅ Record retrieved from database")
                    return True
                else:
                    print(f"❌ Failed to retrieve record: {get_response.status_code}")
                    return False
            else:
                print("❌ No record ID returned")
                return False
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all integration tests"""
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUITE")
    print("=" * 60)
    print(f"Testing API at: {API_BASE_URL}")
    print("=" * 60 + "\n")
    
    tests = [
        ("API Health Check", test_api_health),
        ("Valid Input → Prediction", test_valid_prediction),
        ("Missing Input → Validation Error", test_missing_input),
        ("Invalid Input → Validation Error", test_invalid_input),
        ("Database Storage", test_database_storage),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
        time.sleep(1)  # Small delay between tests
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print("=" * 60)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n✅ All integration tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

