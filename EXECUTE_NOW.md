# Quick Execution Guide

## 🚀 Ready to Execute!

Your complete system is ready to run. Follow these steps:

---

## Step 1: Install Dependencies (if not already done)

```bash
pip install -r requirements.txt
```

---

## Step 2: Start the System

### Option A: Unified Startup (Easiest - Recommended)
```bash
python run_full_system.py
```

This will:
- Start the backend API automatically
- Wait for backend to be ready
- Start the frontend automatically
- Open in your browser

### Option B: Manual Startup (Two Terminals)

**Terminal 1 - Backend:**
```bash
python run_api.py
```

Wait until you see: `[OK] API startup complete`

**Terminal 2 - Frontend:**
```bash
python run_frontend.py
```

---

## Step 3: Access the Application

Once started, the frontend will automatically open at:
- **Frontend UI**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## Step 4: Test the System

1. **Fill in patient data** in the form
2. **Click "Predict Heart Disease"**
3. **View the color-coded result**

### Example Test Data:
- Age: 63
- Gender: Male
- Blood Pressure: 145
- Cholesterol: 233
- (Fill in all other fields)

---

## Step 5: Verify Integration (Optional)

Run integration tests:
```bash
python test_integration.py
```

---

## 🎯 What to Expect

1. **Backend starts** → You'll see model loading messages
2. **Frontend starts** → Browser opens automatically
3. **API health check** → Green checkmark if connected
4. **Form appears** → Enter patient data
5. **Prediction** → Color-coded result card appears

---

## ✅ Success Indicators

- ✅ Backend shows: `[OK] API startup complete`
- ✅ Frontend shows: `✅ Backend API is connected and ready!`
- ✅ Prediction returns with probability and risk level
- ✅ Result card displays with correct color (Green/Yellow/Red)

---

## 🐛 If Something Goes Wrong

### Backend won't start
- Check if port 8000 is already in use
- Ensure all dependencies are installed
- Check for model files in `models/` directory

### Frontend can't connect
- Ensure backend is running first
- Check API URL in `config.py`
- Verify backend is accessible: `curl http://localhost:8000/health`

### Model not loading
- Train a model first: `python phase2_main.py`
- Or use existing models in `models/` directory

---

## 📝 Quick Commands Reference

```bash
# Start everything
python run_full_system.py

# Start backend only
python run_api.py

# Start frontend only
python run_frontend.py

# Run tests
python test_integration.py

# Check API health
curl http://localhost:8000/health
```

---

**You're all set! Execute when ready!** 🎉

