from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return {"message": "✅ Breast Cancer Prediction API is running!"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Expecting JSON input
        data = request.get_json(force=True)
        
        # Example: {"input": [values]}
        input_data = np.array(data["input"]).reshape(1, -1)
        
        # Prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0].tolist()

        return jsonify({
            "prediction": int(prediction),
            "probability": probability
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
