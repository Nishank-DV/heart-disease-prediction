"""
Run the FastAPI Backend Server
Starts the REST API for heart disease prediction
"""

import uvicorn
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("=" * 60)
    print("HEART DISEASE PREDICTION API")
    print("=" * 60)
    print("\nStarting FastAPI server...")
    print("API will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Alternative Docs: http://localhost:8000/redoc")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )

