"""
Run the FastAPI Backend Server
Starts the REST API for heart disease prediction
"""

import uvicorn
import os
import sys
import config

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("=" * 60)
    print("HEART DISEASE PREDICTION API")
    print("=" * 60)
    print("\nStarting FastAPI server...")
    print(f"API will be available at: {config.get_public_api_url()}")
    print(f"API Documentation: {config.get_public_api_url()}/docs")
    print(f"Alternative Docs: {config.get_public_api_url()}/redoc")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    uvicorn.run(
        "backend.app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level="info"
    )

