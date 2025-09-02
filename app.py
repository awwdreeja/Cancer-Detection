from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # <-- Enable CORS

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return {"message": "✅ Breast Cancer Prediction API is running!"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    input_data = np.array(data["input"]).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0].tolist()
    return jsonify({
        "prediction": int(prediction),
        "probability": probability
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
