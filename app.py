from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import json
import os
from datetime import datetime

app = Flask(__name__)

# ── Load artefacts ────────────────────────────────────────────────────────────
MODEL_DIR = "models"

def load_artifacts():
    model        = joblib.load(os.path.join(MODEL_DIR, "best_model.joblib"))
    scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    feat_names   = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))
    with open(os.path.join(MODEL_DIR, "model_metadata.json")) as fh:
        metadata = json.load(fh)
    return model, scaler, feat_names, metadata

try:
    model, scaler, feature_names, metadata = load_artifacts()
    print(f"[OK] Loaded model: {metadata['model_name']}")
    print(f"[OK] Features   : {feature_names}")
except Exception as exc:
    print(f"[WARN] Could not load model artefacts: {exc}")
    print("       Run the notebook first, then restart this server.")
    model = scaler = feature_names = metadata = None


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    if metadata is None:
        return jsonify({"status": "no_model",
                        "message": "Run the notebook to train & save the model first."}), 503
    return jsonify({
        "status":  "healthy",
        "model":   metadata["model_name"],
        "roc_auc": metadata["roc_auc"],
        "f1":      metadata["f1_score"],
        "precision": metadata["precision"],
        "recall":  metadata["recall"],
        "features": feature_names,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run the notebook first."}), 503

    try:
        data    = request.get_json(force=True)
        tx_type = data.get("type", "TRANSFER")

        # One-hot encode 'type' (CASH_IN is the dropped reference)
        raw = {
            "amount":        float(data.get("amount",        0)),
            "oldbalanceOrg": float(data.get("oldbalanceOrg", 0)),
            "newbalanceOrig":float(data.get("newbalanceOrig",0)),
            "oldbalanceDest":float(data.get("oldbalanceDest",0)),
            "newbalanceDest":float(data.get("newbalanceDest",0)),
            "type_CASH_OUT": int(tx_type == "CASH_OUT"),
            "type_DEBIT":    int(tx_type == "DEBIT"),
            "type_PAYMENT":  int(tx_type == "PAYMENT"),
            "type_TRANSFER": int(tx_type == "TRANSFER"),
        }

        # Build DataFrame in the exact column order the model was trained on
        row        = pd.DataFrame([[raw[c] for c in feature_names]], columns=feature_names)
        row_scaled = scaler.transform(row)

        prob      = float(model.predict_proba(row_scaled)[0][1])
        is_fraud  = prob >= 0.5
        risk      = "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.3 else "LOW"

        return jsonify({
            "is_fraud":          is_fraud,
            "fraud_probability": round(prob, 6),
            "risk_level":        risk,
            "model":             metadata["model_name"],
            "timestamp":         datetime.now().isoformat(),
        })

    except KeyError as ke:
        return jsonify({"error": f"Missing field: {ke}"}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
