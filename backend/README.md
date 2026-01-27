# Backend API Documentation

## Heart Disease Prediction REST API

FastAPI-based REST API for heart disease prediction using the trained federated learning model.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API Server
```bash
python run_api.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

---

## 📡 API Endpoints

### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "database_connected": true,
  "message": "API is healthy"
}
```

### 2. Predict Heart Disease
```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
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
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.8500,
  "prediction_text": "Heart disease detected - High risk",
  "risk_level": "High",
  "record_id": 1
}
```

### 3. Get All Predictions
```http
GET /predictions?skip=0&limit=100
```

### 4. Get Prediction by ID
```http
GET /predictions/{prediction_id}
```

### 5. Get Statistics
```http
GET /predictions/stats
```

### 6. Delete Prediction
```http
DELETE /predictions/{prediction_id}
```

---

## 🗄️ Database Schema

### Predictions Table

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key (auto-increment) |
| age | Integer | Patient age |
| sex | Integer | Sex (0=female, 1=male) |
| cp | Integer | Chest pain type (0-3) |
| trestbps | Integer | Resting blood pressure |
| chol | Integer | Cholesterol |
| fbs | Integer | Fasting blood sugar |
| restecg | Integer | Resting ECG |
| thalach | Integer | Max heart rate |
| exang | Integer | Exercise angina |
| oldpeak | Float | ST depression |
| slope | Integer | ST slope |
| ca | Integer | Major vessels |
| thal | Integer | Thalassemia |
| prediction | Integer | Prediction (0/1) |
| probability | Float | Probability (0.0-1.0) |
| created_at | DateTime | Timestamp |

---

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Tests
```bash
# API tests
pytest tests/test_api.py -v

# Database tests
pytest tests/test_database.py -v

# Model tests
pytest tests/test_model.py -v
```

---

## 📝 Example Usage

### Using cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Using Python
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
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
)

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']}")
print(f"Risk Level: {result['risk_level']}")
```

---

## 🔧 Configuration

### Database
- **Type**: SQLite
- **File**: `heart_disease_predictions.db`
- **Location**: Project root directory

### Model Loading
The API tries to load models in this order:
1. `models/federated_model.pth`
2. `models/client_1_model.pth`
3. `models/client_2_model.pth`
4. `models/client_3_model.pth`

If no model is found, a randomly initialized model is used (for testing only).

---

## 📊 API Response Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request (validation error) |
| 404 | Not Found |
| 422 | Validation Error |
| 500 | Internal Server Error |
| 503 | Service Unavailable (model not loaded) |

---

**Backend API is ready for use!**

