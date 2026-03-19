from flask import Flask, request, jsonify
import joblib
import os
import numpy as np

# -------------------------------
# Fix path for model loading
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "artifacts", "model.pkl")

# Load model
model = joblib.load(MODEL_PATH)

# Initialize app
app = Flask(__name__)

# -------------------------------
# Home route
# -------------------------------
@app.route("/")
def home():
    return "Penguin ML API is running 🚀"

# -------------------------------
# Prediction route
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        # Extract features
        bill_length = data.get("bill_length")
        bill_depth = data.get("bill_depth")
        flipper_length = data.get("flipper_length")
        body_mass = data.get("body_mass")

        # Validate input
        if None in [bill_length, bill_depth, flipper_length, body_mass]:
            return jsonify({"error": "Missing input values"}), 400

        # Convert to numpy array
        features = np.array([[bill_length, bill_depth, flipper_length, body_mass]])

        # Prediction
        prediction = model.predict(features)[0]

        return jsonify({
            "prediction": str(prediction)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

# -------------------------------
# Run server
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)