from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os
import joblib

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "data/churn_model_clean.pkl")
model = joblib.load(model_path)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        features = np.array(data["features"])

        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        prediction = model.predict(features)

        return jsonify({
            "prediction": prediction.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict-ui", methods=["POST"])
def predict_ui():
    try:
        age = float(request.form.get("Age"))
        account_manager = int(request.form.get("Account_Manager"))
        years = float(request.form.get("Years"))
        num_sites = float(request.form.get("Num_Sites"))

        features = np.array([[age, account_manager, years, num_sites]])

        prediction = model.predict(features)[0]

        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features).max()

        return render_template(
            "index.html",
            prediction=prediction,
            proba=round(proba, 3) if proba else None,
            values=request.form
        )

    except Exception as e:
        return render_template("index.html", error=str(e), values=request.form)

if __name__ == "__main__":
    app.run(debug=True)