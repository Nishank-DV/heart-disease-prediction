"""
Backend Diagnostic Script
Helps identify why the backend isn't starting
"""

import sys
import os

print("=" * 60)
print("BACKEND DIAGNOSTIC TOOL")
print("=" * 60)

# Check 1: Python version
print("\n[1] Checking Python version...")
print(f"   Python: {sys.version}")

# Check 2: Required packages
print("\n[2] Checking required packages...")
required_packages = [
    'fastapi',
    'uvicorn',
    'sqlalchemy',
    'pydantic',
    'torch',
    'numpy',
    'pandas',
    'sklearn',
    'requests'
]

missing_packages = []
for package in required_packages:
    try:
        if package == 'sklearn':
            __import__('sklearn')
        else:
            __import__(package)
        print(f"   ✓ {package}")
    except ImportError:
        print(f"   ✗ {package} - MISSING")
        missing_packages.append(package)

if missing_packages:
    print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
    print("   Install with: pip install -r requirements.txt")
else:
    print("\n✓ All required packages are installed")

# Check 3: Backend files
print("\n[3] Checking backend files...")
backend_files = [
    'backend/__init__.py',
    'backend/app.py',
    'backend/database.py',
    'backend/models.py',
    'backend/schemas.py',
    'backend/crud.py'
]

missing_files = []
for file in backend_files:
    if os.path.exists(file):
        print(f"   ✓ {file}")
    else:
        print(f"   ✗ {file} - MISSING")
        missing_files.append(file)

if missing_files:
    print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
else:
    print("\n✓ All backend files exist")

# Check 4: Try importing backend
print("\n[4] Testing backend imports...")
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from backend.app import app
    print("   ✓ Backend app imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import backend app: {e}")
    print(f"   Error type: {type(e).__name__}")
    import traceback
    print("\n   Full traceback:")
    traceback.print_exc()

# Check 5: Check database file
print("\n[5] Checking database...")
db_file = "heart_disease_predictions.db"
if os.path.exists(db_file):
    print(f"   ✓ Database file exists: {db_file}")
    print(f"   Size: {os.path.getsize(db_file)} bytes")
else:
    print(f"   ℹ Database file will be created: {db_file}")

# Check 6: Check model files
print("\n[6] Checking model files...")
model_dir = "models"
if os.path.exists(model_dir):
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if model_files:
        print(f"   ✓ Found {len(model_files)} model file(s):")
        for f in model_files:
            print(f"     - {f}")
    else:
        print("   ⚠ No model files found (will use random initialization)")
else:
    print("   ⚠ Models directory not found (will use random initialization)")

# Check 7: Port availability
print("\n[7] Checking port 8000...")
try:
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', 8000))
    sock.close()
    if result == 0:
        print("   ⚠ Port 8000 is already in use!")
        print("   Solution: Stop other services or change port in config.py")
    else:
        print("   ✓ Port 8000 is available")
except Exception as e:
    print(f"   ⚠ Could not check port: {e}")

# Check 8: Try starting backend
print("\n[8] Attempting to start backend (5 second test)...")
try:
    import uvicorn
    from threading import Thread
    import time
    import requests
    
    def test_start():
        try:
            uvicorn.run(
                "backend.app:app",
                host="0.0.0.0",
                port=8000,
                log_level="info"
            )
        except Exception as e:
            print(f"   ✗ Backend failed to start: {e}")
    
    # Start in thread
    thread = Thread(target=test_start, daemon=True)
    thread.start()
    
    # Wait and check
    time.sleep(5)
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("   ✓ Backend started successfully!")
            print("   ✓ Health check passed")
        else:
            print(f"   ⚠ Backend responded with status: {response.status_code}")
    except:
        print("   ✗ Backend did not respond to health check")
        print("   Check the error messages above for details")
        
except Exception as e:
    print(f"   ✗ Could not test backend startup: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
print("\nIf backend still fails, check the error messages above.")
print("Common issues:")
print("  1. Missing packages - run: pip install -r requirements.txt")
print("  2. Port 8000 in use - stop other services")
print("  3. Import errors - check Python path and file structure")
print("=" * 60)

