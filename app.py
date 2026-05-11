from flask import Flask, render_template, request
import pickle
from pathlib import Path
import numpy as np

# -----------------------------
# App Configuration
# -----------------------------
app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

# -----------------------------
# Load Model & Scaler
# -----------------------------
model_path = BASE_DIR / "MODELS" / "ridge.pkl"
scaler_path = BASE_DIR / "MODELS" / "scaler.pkl"

with open(model_path, "rb") as model_file:
    ridge_model = pickle.load(model_file)

with open(scaler_path, "rb") as scaler_file:
    standard_scaler = pickle.load(scaler_file)


# -----------------------------
# Home Route
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    # Render the empty form
    return render_template("home.html")


# -----------------------------
# Prediction Route
# -----------------------------
@app.route("/predictdata", methods=["POST"])
def predict_datapoint():
    try:
        # Get data from form
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        # Create array
        new_data = np.array(
            [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
        )

        # Scale data
        new_data_scaled = standard_scaler.transform(new_data)

        # Predict
        result = ridge_model.predict(new_data_scaled)

        # Render the template directly with the result
        return render_template(
            "home.html",
            results=round(result[0], 2)
        )

    except Exception as e:
        return render_template(
            "home.html",
            results=f"Error: {str(e)}"
        )


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)