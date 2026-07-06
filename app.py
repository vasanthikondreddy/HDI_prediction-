
import io
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
)

# --------------------------------------------------------------------------
# App configuration
# --------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = "hdi-prediction-secret-key-change-in-production"



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
FEATURE_COLUMNS = [
    "Life Expectancy",
    "Mean Years of Schooling",
    "Expected Years of Schooling",
    "GNI Per Capita",
]

# Reasonable real-world ranges, used for basic input validation
FEATURE_RANGES = {
    "Life Expectancy": (20, 90),
    "Mean Years of Schooling": (0, 20),
    "Expected Years of Schooling": (0, 23),
    "GNI Per Capita": (100, 200000),
}

CATEGORY_INFO = {
    "Very High": {
        "meaning": (
            "Countries in this category have excellent healthcare, "
            "very high education levels, and strong economic output. "
            "They generally rank at the top of global development indices."
        ),
        "color": "#2e7d32",
        "suggestions": [],
    },
    "High": {
        "meaning": (
            "Countries in this category show strong development with good "
            "healthcare and education systems, but still have some room to "
            "grow economically or educationally to reach the top tier."
        ),
        "color": "#1565c0",
        "suggestions": [
            "Increase investment in higher education and vocational training.",
            "Improve healthcare infrastructure to raise life expectancy further.",
            "Encourage economic diversification to boost GNI per capita.",
        ],
    },
    "Medium": {
        "meaning": (
            "Countries in this category are developing steadily but face "
            "notable gaps in healthcare access, education quality, or "
            "income levels compared to higher-ranked nations."
        ),
        "color": "#e65100",
        "suggestions": [
            "Expand access to quality primary and secondary education.",
            "Invest in public health infrastructure to raise life expectancy.",
            "Promote job creation and skill development to raise incomes.",
            "Reduce inequality in access to schooling across regions.",
        ],
    },
    "Low": {
        "meaning": (
            "Countries in this category face significant development "
            "challenges across health, education, and income, and would "
            "benefit strongly from targeted development programs."
        ),
        "color": "#c62828",
        "suggestions": [
            "Prioritize basic healthcare access and child/maternal health.",
            "Invest heavily in primary education enrollment and retention.",
            "Support economic infrastructure (roads, electricity, banking).",
            "Seek international development aid and capacity-building programs.",
            "Encourage foreign investment paired with strong local governance.",
        ],
    },
}


# --------------------------------------------------------------------------
# Load model artifacts once at startup
# --------------------------------------------------------------------------
def load_artifacts():
    """Load the trained model, scaler, and label encoder from disk.
    Raises a clear error if train_model.py hasn't been run yet."""
    missing = [p for p in [MODEL_PATH, SCALER_PATH, ENCODER_PATH] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifact(s): {missing}. "
            "Run `python train_model.py` first to generate them."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, scaler, encoder


model, scaler, label_encoder = load_artifacts()

# In-memory prediction history (per-process). Reset when the server restarts.
prediction_history = []

# Holds the most recent batch-prediction CSV result in memory so it can be
# downloaded via /download_batch. For a production multi-user app this
# should be keyed per-user/session and stored in a database or file store
# instead of a single global variable.
last_batch_csv = None


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
def validate_inputs(data: dict):
    """Validate that all required fields are present and numeric,
    and fall within a sane real-world range. Returns (values, errors)."""
    values = {}
    errors = []

    for col in FEATURE_COLUMNS:
        raw = data.get(col)
        if raw is None or str(raw).strip() == "":
            errors.append(f"'{col}' is required.")
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            errors.append(f"'{col}' must be a number.")
            continue

        low, high = FEATURE_RANGES[col]
        if not (low <= val <= high):
            errors.append(f"'{col}' should be between {low} and {high}.")
            continue

        values[col] = val

    return values, errors


def predict_single(values: dict):
    """Run the full prediction pipeline for one set of feature values."""
    X = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    X_scaled = scaler.transform(X)

    pred_encoded = model.predict(X_scaled)[0]
    category = label_encoder.inverse_transform([pred_encoded])[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        confidence = round(float(np.max(proba)) * 100, 2)

    return category, confidence


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    """Render the main page with the input form."""
    return render_template("index.html", features=FEATURE_COLUMNS)


@app.route("/predict", methods=["POST"])
def predict():
    """Handle a single prediction submitted from the form (AJAX/JSON)."""
    data = request.get_json(silent=True) or request.form.to_dict()

    values, errors = validate_inputs(data)
    if errors:
        return jsonify({"success": False, "errors": errors}), 400

    try:
        category, confidence = predict_single(values)
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"success": False, "errors": [f"Prediction failed: {exc}"]}), 500

    info = CATEGORY_INFO.get(category, {})
    result = {
        "success": True,
        "category": category,
        "confidence": confidence,
        "meaning": info.get("meaning", ""),
        "color": info.get("color", "#333"),
        "suggestions": info.get("suggestions", []),
        "inputs": values,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    prediction_history.append(result)
    if len(prediction_history) > 50:
        prediction_history.pop(0)

    return jsonify(result)


@app.route("/history")
def history():
    """Return the in-memory prediction history as JSON."""
    return jsonify({"history": list(reversed(prediction_history))})


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """Accept a CSV upload and return predictions for every row."""
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"success": False, "errors": ["No file uploaded."]}), 400

    try:
        df = pd.read_csv(file)
    except Exception:
        return jsonify({"success": False, "errors": ["Could not read the CSV file."]}), 400

    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        return (
            jsonify(
                {"success": False, "errors": [f"CSV is missing required column(s): {missing_cols}"]}
            ),
            400,
        )

    df_clean = df.dropna(subset=FEATURE_COLUMNS).copy()
    X_scaled = scaler.transform(df_clean[FEATURE_COLUMNS])
    preds_encoded = model.predict(X_scaled)
    df_clean["Predicted HDI Category"] = label_encoder.inverse_transform(preds_encoded)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)
        df_clean["Confidence (%)"] = np.round(np.max(proba, axis=1) * 100, 2)

    # Store the result CSV in memory so it can be downloaded
    global last_batch_csv
    buf = io.StringIO()
    df_clean.to_csv(buf, index=False)
    last_batch_csv = buf.getvalue()

    return jsonify(
        {
            "success": True,
            "rows": len(df_clean),
            "preview": df_clean.head(20).to_dict(orient="records"),
            "distribution": df_clean["Predicted HDI Category"].value_counts().to_dict(),
        }
    )


@app.route("/download_batch")
def download_batch():
    """Download the most recent batch prediction result as a CSV file."""
    if not last_batch_csv:
        return "No batch prediction result available.", 404
    return Response(
        last_batch_csv,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=hdi_batch_predictions.csv"},
    )


@app.errorhandler(404)
def not_found(_e):
    return jsonify({"success": False, "errors": ["Page not found."]}), 404


@app.errorhandler(500)
def server_error(_e):
    return jsonify({"success": False, "errors": ["Internal server error."]}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
