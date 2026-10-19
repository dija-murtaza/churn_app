"""
Customer Churn Prediction - Flask backend
Uses the Assignment 1 model (Logistic Regression) with the same preprocessing:
  1. AvgCostPerMonth = TotalCharges / (tenure + 1)
  2. Binary columns -> 0/1
  3. One-hot encoding (drop_first=True) for multi-category columns
  4. Column order aligned to training columns (columns.pkl)
  5. StandardScaler (scaler.pkl)
"""
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

# Absolute path so it also works on PythonAnywhere (not only locally)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# ---------- Load artifacts saved in Assignment 1 ----------
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))

# ---------- Field definitions (used for validation) ----------
BINARY_MAP = {"No": 0, "Yes": 1, "Female": 0, "Male": 1}
BINARY_COLS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]

MULTI_COLS = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
              "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
              "Contract", "PaymentMethod"]

INTERNET_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                     "TechSupport", "StreamingTV", "StreamingMovies"]

ALLOWED = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["0", "1"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"],
}

LABELS = {
    "gender": "Gender", "SeniorCitizen": "Senior citizen", "Partner": "Partner",
    "Dependents": "Dependents", "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines", "InternetService": "Internet service",
    "OnlineSecurity": "Online security", "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection", "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies",
    "Contract": "Contract", "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method", "tenure": "Tenure",
    "MonthlyCharges": "Monthly charges", "TotalCharges": "Total charges",
}


def validate(form):
    """Validate the submitted form. Returns (clean_data, errors)."""
    errors, data = [], {}

    # Dropdown fields
    for field, options in ALLOWED.items():
        value = (form.get(field) or "").strip()
        if not value:
            errors.append(f"Please select a value for '{LABELS[field]}'.")
        elif value not in options:
            errors.append(f"'{value}' is not a valid choice for '{LABELS[field]}'.")
        else:
            data[field] = value

    # Numeric fields
    numeric_rules = {
        "tenure": (int, 0, 72, "a whole number between 0 and 72 months"),
        "MonthlyCharges": (float, 0, 1000, "a number between 0 and 1000"),
        "TotalCharges": (float, 0, 100000, "a number between 0 and 100000"),
    }
    for field, (cast, lo, hi, hint) in numeric_rules.items():
        raw = (form.get(field) or "").strip()
        if not raw:
            errors.append(f"Please enter '{LABELS[field]}'.")
            continue
        try:
            num = cast(raw)
        except ValueError:
            errors.append(f"'{LABELS[field]}' must be {hint}.")
            continue
        if not (lo <= num <= hi):
            errors.append(f"'{LABELS[field]}' must be {hint}.")
            continue
        data[field] = num

    if errors:
        return None, errors

    # Make dependent fields consistent (same values the training data uses)
    if data["PhoneService"] == "No":
        data["MultipleLines"] = "No phone service"
    elif data["MultipleLines"] == "No phone service":
        errors.append("'Multiple lines' cannot be 'No phone service' when the customer has phone service.")

    if data["InternetService"] == "No":
        for f in INTERNET_SERVICES:
            data[f] = "No internet service"
    else:
        for f in INTERNET_SERVICES:
            if data[f] == "No internet service":
                errors.append(f"'{LABELS[f]}' cannot be 'No internet service' when the customer has internet.")

    return (None, errors) if errors else (data, [])


def preprocess(data):
    """Apply EXACTLY the same preprocessing as in training."""
    row = dict(data)
    row["SeniorCitizen"] = int(row["SeniorCitizen"])

    # Feature engineering
    row["AvgCostPerMonth"] = row["TotalCharges"] / (row["tenure"] + 1)

    # Binary encoding
    for col in BINARY_COLS:
        row[col] = BINARY_MAP[row[col]]

    # One-hot encoding with drop_first=True.
    # (done manually: pd.get_dummies on a SINGLE row would drop its only category
    #  and give wrong results, so we set the matching dummy column directly)
    encoded = {col: 0 for col in feature_columns}
    for key, value in row.items():
        if key in MULTI_COLS:
            dummy = f"{key}_{value}"
            if dummy in encoded:          # baseline category has no column -> stays all 0
                encoded[dummy] = 1
        elif key in encoded:
            encoded[key] = value

    # Same column order as training, then scale with the fitted scaler
    frame = pd.DataFrame([encoded], columns=feature_columns)
    return scaler.transform(frame)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", form={}, errors=[], result=None)


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form
    data, errors = validate(form)
    if errors:
        return render_template("index.html", form=form, errors=errors, result=None)

    try:
        scaled = preprocess(data)
        pred = int(model.predict(scaled)[0])
        prob = float(model.predict_proba(scaled)[0][1]) * 100
    except Exception:
        app.logger.exception("Prediction failed")
        return render_template(
            "index.html", form=form, result=None,
            errors=["Something went wrong while making the prediction. Please check your inputs and try again."])

    if prob >= 70:
        risk = "High"
    elif prob >= 40:
        risk = "Medium"
    else:
        risk = "Low"

    result = {
        "label": "Churn" if pred == 1 else "No Churn",
        "is_churn": pred == 1,
        "probability": round(prob, 1),
        "risk": risk,
    }
    return render_template("index.html", form=form, errors=[], result=result)


if __name__ == "__main__":
    app.run(debug=True)