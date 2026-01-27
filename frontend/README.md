# Frontend Application - Heart Disease Prediction

## Streamlit Web Interface

A clean, minimal, medical-grade user interface for the Heart Disease Prediction system.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Backend API (Required)
```bash
python run_api.py
```

The API must be running on `http://localhost:8000`

### 3. Start Frontend
```bash
python run_frontend.py
```

Or directly:
```bash
streamlit run frontend/app.py
```

The application will open at: **http://localhost:8501**

---

## 🎨 Features

### 1. Clean Landing Page
- Large, readable title
- Professional medical-grade styling
- Clear project description

### 2. Patient Data Input Form
All 13 medical features with:
- **Basic Information:**
  - Age (years)
  - Gender (Male/Female)
  - Chest Pain Type
  - Resting Blood Pressure
  - Serum Cholesterol
  - Fasting Blood Sugar

- **Medical Test Results:**
  - Resting ECG Results
  - Maximum Heart Rate
  - Exercise Induced Angina
  - ST Depression
  - ST Slope
  - Number of Major Vessels
  - Thalassemia Type

### 3. Prediction Results
- **Visual Indicators:**
  - Color-coded risk levels (Green/Yellow/Red)
  - Large, readable prediction text
  - Probability percentage
  - Risk level badge

- **Risk Levels:**
  - **Low Risk** (Green): Probability < 0.5
  - **Medium Risk** (Yellow): Probability 0.5-0.7
  - **High Risk** (Red): Probability > 0.7

### 4. User Experience
- ✅ Loading spinner during prediction
- ✅ API health check
- ✅ Clear error messages
- ✅ Form validation
- ✅ Responsive layout
- ✅ Large, readable fonts

---

## 📱 UI Components

### Color Scheme
- **Primary Blue**: #1f77b4 (Medical professional)
- **Success Green**: #28a745 (Low risk)
- **Warning Yellow**: #ffc107 (Medium risk)
- **Danger Red**: #dc3545 (High risk)

### Typography
- **Title**: 3rem, Bold
- **Section Headers**: 1.5rem, Semi-bold
- **Form Labels**: 1.1rem
- **Result Text**: 2rem, Bold

---

## 🔧 Configuration

### API Endpoint
The frontend connects to the backend API. To change the API URL, edit:
```python
API_BASE_URL = "http://localhost:8000"
```

In `frontend/app.py`

---

## 📊 Example Usage

1. **Open the application** in your browser
2. **Fill in patient data** using the form
3. **Click "Predict Heart Disease"**
4. **View results** with color-coded risk level

---

## 🎯 Design Principles

1. **Minimalism**: Clean, uncluttered interface
2. **Readability**: Large fonts, clear labels
3. **Medical Grade**: Professional healthcare appearance
4. **Accessibility**: High contrast, clear indicators
5. **Responsiveness**: Works on different screen sizes

---

## 🐛 Troubleshooting

### API Connection Error
If you see "Cannot connect to API server":
1. Ensure backend is running: `python run_api.py`
2. Check API is accessible: `http://localhost:8000/health`

### Port Already in Use
If port 8501 is busy:
```bash
streamlit run frontend/app.py --server.port=8502
```

---

**Frontend is ready to use!** 🎉

