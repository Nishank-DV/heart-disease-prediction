# API Usage Guide

## Complete Guide for Using the Heart Disease Prediction API

---

## 🚀 Starting the API

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python run_api.py
```

**Server starts at**: http://localhost:8000

---

## 📡 API Endpoints

### 1. Health Check

**Endpoint**: `GET /health`

**Example:**
```bash
curl http://localhost:8000/health
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

---

### 2. Predict Heart Disease

**Endpoint**: `POST /predict`

**Request:**
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

---

### 3. Get All Predictions

**Endpoint**: `GET /predictions?skip=0&limit=100`

**Example:**
```bash
curl http://localhost:8000/predictions?skip=0&limit=10
```

---

### 4. Get Prediction by ID

**Endpoint**: `GET /predictions/{id}`

**Example:**
```bash
curl http://localhost:8000/predictions/1
```

---

### 5. Get Statistics

**Endpoint**: `GET /predictions/stats`

**Example:**
```bash
curl http://localhost:8000/predictions/stats
```

**Response:**
```json
{
  "total_predictions": 100,
  "disease_detected": 45,
  "no_disease": 55,
  "disease_rate": 45.0,
  "average_probability": 0.5234
}
```

---

## 🧪 Testing the API

### Using Python requests

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Make prediction
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

response = requests.post(
    "http://localhost:8000/predict",
    json=patient_data
)

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.2%}")
print(f"Risk Level: {result['risk_level']}")
```

---

## 📊 Interactive API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive documentation where you can test the API directly in your browser!

---

**API is ready to use!**

