from flask import Flask, render_template, request
import torch
import os
import sys

# Allow imports from project root
sys.path.append(os.path.abspath(".."))

from client.model import HeartDiseaseMLP

app = Flask(__name__)

# Load model
MODEL_PATH = os.path.join("..", "models", "client_1_model.pth")

model = HeartDiseaseMLP(input_size=13)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        features = [
            float(request.form["age"]),
            float(request.form["sex"]),
            float(request.form["cp"]),
            float(request.form["trestbps"]),
            float(request.form["chol"]),
            float(request.form["fbs"]),
            float(request.form["restecg"]),
            float(request.form["thalach"]),
            float(request.form["exang"]),
            float(request.form["oldpeak"]),
            float(request.form["slope"]),
            float(request.form["ca"]),
            float(request.form["thal"])
        ]

        with torch.no_grad():
            input_tensor = torch.tensor([features], dtype=torch.float32)
            output = model(input_tensor)
            probability = output.item()

        result = (
            "High Risk of Heart Disease"
            if probability >= 0.5
            else "Low Risk of Heart Disease"
        )

        return render_template(
            "result.html",
            result=result,
            probability=round(probability * 100, 2)
        )

    return render_template("predict.html")


if __name__ == "__main__":
    app.run(debug=True)
