# Complete System Integration Guide

## Full Stack Integration: Frontend + Backend + Database + AI Model

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER INTERFACE                        │
│              (Streamlit Frontend - Port 8501)            │
│  - Patient Data Input Form                              │
│  - Result Display with Color-Coded Risk Levels           │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP POST /predict
                     │ JSON: {age, sex, cp, ...}
                     ▼
┌─────────────────────────────────────────────────────────┐
│                    API LAYER                             │
│            (FastAPI Backend - Port 8000)                 │
│  - Request Validation (Pydantic)                         │
│  - Data Preprocessing                                    │
│  - Model Inference                                       │
└────┬──────────────────────────────┬──────────────────────┘
     │                              │
     ▼                              ▼
┌──────────────┐          ┌──────────────────┐
│   DATABASE   │          │   AI MODEL        │
│   (SQLite)   │          │   (PyTorch MLP)   │
│              │          │                   │
│ - Store      │          │ - Load trained   │
│   predictions│          │   model weights   │
│ - Track      │          │ - Make prediction│
│   statistics │          │ - Return          │
│              │          │   probability     │
└──────────────┘          └──────────────────┘
```

---

## 🔄 Data Flow

### 1. User Input (Frontend)
```python
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
```

### 2. API Request (Frontend → Backend)
```python
POST http://localhost:8000/predict
Content-Type: application/json
Body: {patient_data}
```

### 3. Backend Processing
1. **Validation**: Pydantic schema validates input
2. **Preprocessing**: Convert to tensor format
3. **Model Inference**: PyTorch model makes prediction
4. **Database Storage**: Save prediction to SQLite
5. **Response**: Return JSON with results

### 4. API Response (Backend → Frontend)
```json
{
    "prediction": 1,
    "probability": 0.8500,
    "prediction_text": "Heart disease detected - High risk",
    "risk_level": "High",
    "record_id": 1
}
```

### 5. UI Display (Frontend)
- Color-coded result card (Red/Yellow/Green)
- Large prediction text
- Probability percentage
- Risk level badge

---

## 🚀 Quick Start

### Option 1: Unified Startup (Recommended)
```bash
python run_full_system.py
```

This starts both backend and frontend automatically.

### Option 2: Manual Startup
```bash
# Terminal 1: Backend
python run_api.py

# Terminal 2: Frontend
python run_frontend.py
```

---

## 🧪 Testing Integration

### Run Integration Tests
```bash
python test_integration.py
```

### Test Scenarios

1. **Valid Input → Correct Prediction**
   - Submit complete patient data
   - Receive prediction with probability
   - Verify database storage

2. **Missing Input → Validation Error**
   - Submit incomplete data
   - Receive 422 validation error
   - See user-friendly error message

3. **Invalid Input → Validation Error**
   - Submit out-of-range values
   - Receive 422 validation error
   - See field-specific error

4. **Backend Down → User-Friendly Message**
   - Stop backend server
   - Frontend shows connection error
   - Provides instructions to fix

---

## 📁 File Structure

```
project/
├── config.py                 # Centralized configuration
├── run_full_system.py        # Unified startup script
├── test_integration.py       # Integration tests
│
├── frontend/
│   └── app.py               # Streamlit frontend
│
├── backend/
│   ├── app.py               # FastAPI backend
│   ├── database.py          # Database config
│   ├── models.py            # SQLAlchemy models
│   ├── schemas.py           # Pydantic schemas
│   └── crud.py              # Database operations
│
├── client/
│   └── model.py             # PyTorch MLP model
│
└── tests/
    ├── test_api.py          # API tests
    ├── test_database.py     # Database tests
    └── test_model.py         # Model tests
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# API Configuration
export API_HOST=0.0.0.0
export API_PORT=8000
export API_BASE_URL=http://localhost:8000
export API_TIMEOUT=10

# Frontend Configuration
export FRONTEND_PORT=8501
export FRONTEND_HOST=localhost

# Database Configuration
export DATABASE_URL=sqlite:///./heart_disease_predictions.db
```

### Config File (`config.py`)

All configuration is centralized in `config.py`:
- API URLs and ports
- Database settings
- Model paths
- Feature counts

---

## 🔌 API Integration Details

### Frontend API Calls

```python
# Health Check
GET /health
→ Returns: {status, model_loaded, database_connected}

# Prediction
POST /predict
Body: {patient_data}
→ Returns: {prediction, probability, risk_level, record_id}
```

### Error Handling

| Error Type | HTTP Code | Frontend Display |
|------------|-----------|------------------|
| Validation Error | 422 | Field-specific error messages |
| Service Unavailable | 503 | "Model not loaded" message |
| Connection Error | - | "Cannot connect to API" with instructions |
| Timeout | - | "Request timed out" message |
| Server Error | 500 | Generic error message |

---

## 💾 Database Integration

### Storage Flow

1. **Prediction Made** → Backend receives request
2. **Model Inference** → Prediction and probability calculated
3. **Database Write** → `create_prediction()` saves to SQLite
4. **Record ID Returned** → Frontend receives `record_id`
5. **Verification** → Can retrieve record via `GET /predictions/{id}`

### Database Schema

```python
Predictions Table:
- id (Primary Key)
- age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
- prediction (0 or 1)
- probability (0.0 to 1.0)
- created_at (Timestamp)
```

---

## 🤖 Model Integration

### Model Loading

1. **Startup**: Backend loads model on startup
2. **Priority Order**:
   - `models/federated_model.pth` (preferred)
   - `models/client_1_model.pth`
   - `models/client_2_model.pth`
   - `models/client_3_model.pth`
   - Random initialization (fallback)

### Prediction Process

```python
1. Preprocess patient data → Tensor format
2. Model forward pass → Probability output
3. Threshold (0.5) → Binary prediction
4. Risk level calculation → Low/Medium/High
5. Database storage → Save all data
```

---

## 🎨 Frontend-Backend Communication

### Request Format

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

### Response Format

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

## ✅ Integration Checklist

- [x] Frontend connects to backend API
- [x] Form data sent as JSON
- [x] Backend validates input
- [x] Model makes prediction
- [x] Prediction stored in database
- [x] Result displayed in UI
- [x] Error handling for all scenarios
- [x] Loading states during API calls
- [x] Configuration centralized
- [x] Integration tests pass

---

## 🐛 Troubleshooting

### Frontend can't connect to backend
1. Check backend is running: `python run_api.py`
2. Verify API health: `curl http://localhost:8000/health`
3. Check API URL in `config.py`

### Model not loading
1. Ensure model files exist in `models/` directory
2. Check model path in `backend/app.py`
3. Train model first: `python phase2_main.py`

### Database errors
1. Check database file permissions
2. Verify SQLite is installed
3. Check database path in `config.py`

### Validation errors
1. Check input ranges match schema
2. Verify all 13 fields are provided
3. Check data types (int vs float)

---

## 📊 Performance

- **API Response Time**: < 1 second
- **Model Inference**: < 500ms
- **Database Write**: < 100ms
- **Frontend Load**: < 2 seconds

---

## 🔒 Security Considerations

1. **Input Validation**: All inputs validated by Pydantic
2. **SQL Injection**: Prevented by SQLAlchemy ORM
3. **Error Messages**: Don't expose sensitive information
4. **CORS**: Configured for cross-origin requests

---

## 📚 Related Documentation

- `COMPLETE_SYSTEM_GUIDE.md` - Full system overview
- `FRONTEND_GUIDE.md` - Frontend details
- `backend/README.md` - Backend API documentation
- `API_USAGE.md` - API usage examples

---

**System is fully integrated and ready for use!** 🎉

