"""
Unified System Startup Script
Starts both backend API and frontend in a coordinated manner
"""

import subprocess
import sys
import os
import time
import requests
from threading import Thread

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from config import API_BASE_URL, API_PORT, FRONTEND_PORT, get_health_endpoint
except ImportError:
    # Fallback if config.py not available
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "8501"))
    def get_health_endpoint():
        return f"{API_BASE_URL}/health"

def check_backend_health(max_retries=10, delay=2):
    """
    Wait for backend to be ready
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
        
    Returns:
        True if backend is healthy, False otherwise
    """
    print("Waiting for backend to start...")
    for i in range(max_retries):
        try:
            response = requests.get(get_health_endpoint(), timeout=2)
            if response.status_code == 200:
                print("[OK] Backend is ready!")
                return True
        except:
            pass
        print(f"  Attempt {i+1}/{max_retries}...")
        time.sleep(delay)
    return False

def start_backend():
    """Start the FastAPI backend server"""
    print("=" * 60)
    print("STARTING BACKEND API")
    print("=" * 60)
    print(f"API will be available at: {API_BASE_URL}")
    print(f"API Documentation: {API_BASE_URL}/docs")
    print("=" * 60)
    
    try:
        import uvicorn
        
        # Use string module path like run_api.py does
        # This avoids import path issues
        uvicorn.run(
            "backend.app:app",
            host="0.0.0.0",
            port=API_PORT,
            reload=False,  # Disable reload in threaded mode
            log_level="info"
        )
    except ImportError as e:
        print(f"\n[ERROR] Import error: {e}")
        print("[ERROR] Make sure all dependencies are installed:")
        print("        pip install -r requirements.txt")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Failed to start backend: {e}")
        print("[ERROR] Full error details:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def start_frontend():
    """Start the Streamlit frontend"""
    print("=" * 60)
    print("STARTING FRONTEND")
    print("=" * 60)
    print(f"Frontend will be available at: http://localhost:{FRONTEND_PORT}")
    print("=" * 60)
    
    frontend_app = os.path.join(os.path.dirname(__file__), "frontend", "app.py")
    
    try:
        subprocess.run([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            frontend_app,
            "--server.port", str(FRONTEND_PORT),
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n[INFO] Frontend server stopped by user")
    except Exception as e:
        print(f"[ERROR] Failed to start frontend: {e}")
        sys.exit(1)

def main():
    """Main function to start the complete system"""
    print("\n" + "=" * 60)
    print("HEART DISEASE PREDICTION - FULL SYSTEM STARTUP")
    print("=" * 60)
    print("\nStarting integrated system...")
    print("  - Backend API (FastAPI)")
    print("  - Frontend UI (Streamlit)")
    print("\nPress Ctrl+C to stop all services")
    print("=" * 60 + "\n")
    
    # Start backend in a separate thread
    backend_thread = Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    # Wait for backend to be ready
    if not check_backend_health():
        print("[ERROR] Backend failed to start or is not responding")
        print("Please check the backend logs for errors")
        sys.exit(1)
    
    # Small delay to ensure backend is fully ready
    time.sleep(1)
    
    # Start frontend (this will block)
    print("\n" + "=" * 60)
    print("Starting frontend...")
    print("=" * 60 + "\n")
    start_frontend()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Shutting down system...")
        print("[INFO] All services stopped")
        sys.exit(0)

