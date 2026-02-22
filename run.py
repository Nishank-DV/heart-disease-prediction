"""
Single-command project runner.
Starts the FastAPI backend which now serves the web UI.
"""

import socket
import uvicorn
import config


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


if __name__ == "__main__":
    print("=" * 60)
    print("HEART DISEASE PREDICTION")
    print("=" * 60)
    print(f"Web UI: {config.get_public_api_url()}")
    print(f"API Docs: {config.get_public_api_url()}/docs")
    print("=" * 60)

    check_host = config.API_PUBLIC_HOST
    if _port_in_use(check_host, config.API_PORT):
        print(
            "[ERROR] Port 8000 is already in use. "
            "Stop the other server or change API_PORT and try again."
        )
    else:
        uvicorn.run(
            "backend.app:app",
            host=config.API_HOST,
            port=config.API_PORT,
            reload=False,
            log_level="info"
        )
