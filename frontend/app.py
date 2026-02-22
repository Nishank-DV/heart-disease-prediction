import os
import sys
from typing import Dict, Any

import requests

try:
    from flask import Flask, render_template, request
except ModuleNotFoundError:
    print("Flask UI is deprecated. Use 'python run.py' for the FastAPI UI.")
    raise SystemExit(0)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config

app = Flask(__name__)


def make_prediction(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        response = requests.post(
            config.get_predict_endpoint(),
            json=payload,
            timeout=config.API_TIMEOUT
        )
        if response.status_code == 200:
            return response.json()
        return {"error": response.text}
    except Exception as exc:
        return {"error": str(exc)}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ui/predict", methods=["GET", "POST"])
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        payload = {
            "age": float(request.form["age"]),
            "sex": float(request.form["sex"]),
            "cp": float(request.form["cp"]),
            "trestbps": float(request.form["trestbps"]),
            "chol": float(request.form["chol"]),
            "fbs": float(request.form["fbs"]),
            "restecg": float(request.form["restecg"]),
            "thalach": float(request.form["thalach"]),
            "exang": float(request.form["exang"]),
            "oldpeak": float(request.form["oldpeak"]),
            "slope": float(request.form["slope"]),
            "ca": float(request.form["ca"]),
            "thal": float(request.form["thal"])
        }

        result = make_prediction(payload)
        if "error" in result:
            return render_template("predict.html", error=result["error"])

        return render_template(
            "result.html",
            prediction_text=result.get("prediction_text", ""),
            risk_level=result.get("risk_level", "Low"),
            probability=round(result.get("probability", 0.0) * 100, 2),
            record_id=result.get("record_id", "-")
        )

    return render_template("predict.html")


if __name__ == "__main__":
    app.run(host=config.FRONTEND_HOST, port=config.FRONTEND_PORT, debug=False)
 