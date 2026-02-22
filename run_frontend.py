"""
Deprecated: FastAPI now serves the web UI.
Use python run.py for the single-command experience.
"""

import subprocess
import sys
import os


def main() -> None:
    print("=" * 60)
    print("HEART DISEASE PREDICTION")
    print("=" * 60)
    print("The Flask frontend is deprecated.")
    print("Starting the unified FastAPI UI instead...")
    print("=" * 60)

    runner = os.path.join(os.path.dirname(__file__), "run.py")
    subprocess.run([sys.executable, runner])


if __name__ == "__main__":
    main()

