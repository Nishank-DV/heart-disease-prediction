# Troubleshooting Guide

## Backend Won't Start - Quick Fixes

### Step 1: Run Diagnostic Tool
```bash
python diagnose_backend.py
```

This will check:
- Python version
- Required packages
- Backend files
- Import errors
- Port availability
- Model files

### Step 2: Common Issues & Solutions

#### Issue 1: Missing Packages
**Error**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**:
```bash
pip install -r requirements.txt
```

#### Issue 2: Port Already in Use
**Error**: `[ERROR] Backend failed to start or is not responding`

**Solution**:
```bash
# Check what's using port 8000 (Windows)
netstat -ano | findstr :8000

# Kill the process or change port in config.py
```

#### Issue 3: Import Errors
**Error**: `ImportError: cannot import name 'app' from 'backend.app'`

**Solution**:
1. Check file structure - ensure `backend/app.py` exists
2. Check Python path - run from project root directory
3. Verify all backend files are present

#### Issue 4: Database Errors
**Error**: Database connection issues

**Solution**:
- Check file permissions
- Ensure SQLite is available
- Check database path in `config.py`

#### Issue 5: Model Loading Errors
**Error**: Model file not found

**Solution**:
- This is OK - system will use random initialization
- For accurate predictions, train model first:
  ```bash
  python phase2_main.py
  ```

### Step 3: Manual Backend Start (Better Error Visibility)

Instead of `run_full_system.py`, try starting backend manually:

```bash
python run_api.py
```

This will show detailed error messages directly in the terminal.

### Step 4: Check Backend Logs

When backend starts, you should see:
```
[OK] Database tables created
[OK] Model and scaler initialized
[OK] API startup complete
```

If you see errors before these messages, that's the issue.

### Step 5: Test Backend Directly

```bash
# Test health endpoint
curl http://localhost:8000/health

# Or in Python
python -c "import requests; print(requests.get('http://localhost:8000/health').json())"
```

---

## Quick Fixes Checklist

- [ ] Run `python diagnose_backend.py`
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Check port 8000 is free
- [ ] Start backend manually: `python run_api.py`
- [ ] Check for import errors in terminal output
- [ ] Verify all backend files exist
- [ ] Check Python version (3.8+)

---

## Still Not Working?

1. **Share the diagnostic output**: Run `python diagnose_backend.py` and share the output
2. **Check terminal errors**: Look for red error messages
3. **Verify file structure**: Ensure all files are in correct locations
4. **Check Python environment**: Ensure you're using the correct Python interpreter

---

**Most common fix**: Run `pip install -r requirements.txt` to install all dependencies!

