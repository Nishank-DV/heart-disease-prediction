"""
Run Streamlit Frontend Application
Starts the web-based UI for heart disease prediction
"""

import subprocess
import sys
import os

def main():
    """Start Streamlit frontend"""
    print("=" * 60)
    print("HEART DISEASE PREDICTION - FRONTEND")
    print("=" * 60)
    print("\nStarting Streamlit application...")
    print("\nThe application will open in your default web browser.")
    print("If it doesn't open automatically, visit:")
    print("  http://localhost:8501")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    # Get the frontend app path
    frontend_app = os.path.join(os.path.dirname(__file__), "frontend", "app.py")
    
    # Run Streamlit
    subprocess.run([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        frontend_app,
        "--server.port=8501",
        "--server.headless=false"
    ])

if __name__ == "__main__":
    main()

