# Frontend Development Guide

## Complete Guide for the Streamlit Frontend Application

---

## 📋 Overview

The frontend is a Streamlit web application that provides a user-friendly interface for the Heart Disease Prediction system. It features a clean, medical-grade design optimized for healthcare professionals and examiners.

---

## 🏗️ Architecture

```
frontend/
└── app.py              # Main Streamlit application

run_frontend.py         # Frontend startup script
```

---

## 🎨 UI Components

### 1. Landing Page
- **Title**: Large, centered heading
- **Subtitle**: Project description
- **API Status**: Health check indicator

### 2. Patient Form
- **Two-column layout** for better organization
- **Basic Information** (Left column):
  - Age, Gender, Chest Pain Type
  - Blood Pressure, Cholesterol, Blood Sugar
  
- **Medical Test Results** (Right column):
  - ECG, Heart Rate, Exercise Angina
  - ST Depression, Slope, Vessels, Thalassemia

### 3. Result Display
- **Color-coded cards**:
  - Green: Low risk
  - Yellow: Medium risk
  - Red: High risk
  
- **Information shown**:
  - Prediction (Disease/No Disease)
  - Probability percentage
  - Risk level badge
  - Detailed JSON (expandable)

---

## 🎯 Key Features

### 1. API Integration
```python
def make_prediction(patient_data):
    response = requests.post(
        API_PREDICT_ENDPOINT,
        json=patient_data,
        timeout=10
    )
    return response.json()
```

### 2. Loading States
- Spinner during API calls
- Health check on startup
- Clear error messages

### 3. Validation
- Input range validation
- Required field checking
- Type validation

### 4. Styling
- Custom CSS for medical-grade appearance
- Responsive design
- Color-coded risk indicators

---

## 🔧 Customization

### Change Colors
Edit CSS in `load_custom_css()`:
```python
.risk-badge.high {
    background-color: #dc3545;  # Change this
}
```

### Add Features
1. Add new form fields in the form section
2. Update `patient_data` dictionary
3. Modify result display as needed

### Change API URL
```python
API_BASE_URL = "http://your-api-url:8000"
```

---

## 📱 Responsive Design

The layout adapts to different screen sizes:
- **Desktop**: Two-column form layout
- **Tablet**: Stacked columns
- **Mobile**: Single column

---

## 🧪 Testing

### Manual Testing Checklist
- [ ] Form submission works
- [ ] API connection successful
- [ ] Results display correctly
- [ ] Error messages show properly
- [ ] Loading spinner appears during prediction
- [ ] Color coding matches risk levels

### Test Scenarios
1. **Valid Input**: Submit complete form
2. **Invalid Input**: Submit with out-of-range values
3. **API Down**: Test error handling
4. **Slow Connection**: Test timeout handling

---

## 🎨 Design System

### Colors
- **Primary**: #1f77b4 (Blue)
- **Success**: #28a745 (Green)
- **Warning**: #ffc107 (Yellow)
- **Danger**: #dc3545 (Red)
- **Background**: White
- **Text**: #333 (Dark Gray)

### Typography
- **Title**: 3rem, Bold
- **Headers**: 1.5rem, Semi-bold
- **Body**: 1rem, Regular
- **Labels**: 1.1rem, Medium

### Spacing
- **Section Margin**: 2rem
- **Card Padding**: 2rem
- **Button Height**: 3.5rem

---

## 🚀 Deployment

### Local Development
```bash
streamlit run frontend/app.py
```

### Production Deployment
1. Use Streamlit Cloud
2. Or deploy on server with:
   ```bash
   streamlit run frontend/app.py --server.port=8501
   ```

---

## 📝 Code Structure

### Main Functions
1. `load_custom_css()` - Loads styling
2. `check_api_health()` - Checks API status
3. `make_prediction()` - Calls API
4. `display_result()` - Shows results
5. `main()` - Main application logic

### State Management
- Uses Streamlit's session state
- Form data stored in variables
- Results displayed after submission

---

## 🔒 Security Considerations

1. **Input Validation**: All inputs validated before submission
2. **API Timeout**: 10-second timeout prevents hanging
3. **Error Handling**: Graceful error messages
4. **No Data Storage**: Frontend doesn't store patient data

---

## 📊 Performance

- **Fast Loading**: Minimal dependencies
- **Efficient API Calls**: Single request per prediction
- **Responsive UI**: Instant feedback
- **Optimized CSS**: Inline styles for speed

---

## 🎓 Examiner-Friendly Features

1. **Large Fonts**: Easy to read
2. **Clear Labels**: Self-explanatory
3. **Color Coding**: Visual risk indicators
4. **Simple Layout**: No clutter
5. **Professional Design**: Medical-grade appearance

---

**Frontend is production-ready!** 🎉

