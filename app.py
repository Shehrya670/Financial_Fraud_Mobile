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
        data = request.get_json(force=True)
        tx_type = data.get("type", "TRANSFER")

        # --- Helper to parse floats safely ---
        def safe_float(val):
            try:
                if val is None or str(val).strip() == "": return 0.0
                return float(val)
            except: return 0.0

        # ── Preprocess ────────────────────────────────────────────────────────
        amount = safe_float(data.get("amount"))
        oldbalanceOrg = safe_float(data.get("oldbalanceOrg"))
        newbalanceOrig = safe_float(data.get("newbalanceOrig"))
        oldbalanceDest = safe_float(data.get("oldbalanceDest"))
        newbalanceDest = safe_float(data.get("newbalanceDest"))

        # --- 1. Basic Validation ---
        if amount <= 0:
            return jsonify({
                "is_fraud": False,
                "fraud_probability": 0.0,
                "risk_level": "LOW",
                "verdict": "Invalid Transaction",
                "recommendation": "Transaction amount must be greater than zero.",
                "timestamp": datetime.now().isoformat()
            })

        # --- 2. Feature Engineering ---
        # Get explicit Recipient Type from UI
        dest_type_raw = data.get("destType", "CUSTOMER")
        is_merchant = 1 if dest_type_raw == "MERCHANT" else 0

        # Heuristic fallback: if it's a PAYMENT and balances are 0, it's likely a merchant
        if not is_merchant and tx_type == "PAYMENT" and newbalanceDest == 0 and oldbalanceDest == 0:
            is_merchant = 1

        errorOrg = newbalanceOrig + amount - oldbalanceOrg
        errorDest = 0 if is_merchant else (oldbalanceDest + amount - newbalanceDest)

        raw = {
            "amount":           amount,
            "oldbalanceOrg":    oldbalanceOrg,
            "newbalanceOrig":   newbalanceOrig,
            "oldbalanceDest":   oldbalanceDest,
            "newbalanceDest":   newbalanceDest,
            "isMerchant":       is_merchant,
            "errorBalanceOrg":  errorOrg,
            "errorBalanceDest": errorDest,
            "type_CASH_OUT":    int(tx_type == "CASH_OUT"),
            "type_DEBIT":       int(tx_type == "DEBIT"),
            "type_PAYMENT":     int(tx_type == "PAYMENT"),
            "type_TRANSFER":    int(tx_type == "TRANSFER"),
        }

        # Build DataFrame
        row = pd.DataFrame([[raw[c] for c in feature_names]], columns=feature_names)
        row_scaled = scaler.transform(row)

        # Get Model Prediction
        prob = float(model.predict_proba(row_scaled)[0][1])
        reasons = []
        
        # --- 3. Expert Rules (Logical Guards) ---
        # These guards protect against logical impossibilities that the ML model
        # might miss due to dataset biases (e.g. PaySim only labeling Transfer fraud).
        
        if not is_merchant:
            # Rule: Vanishing Money (Origin balance drops, Dest balance stays same)
            if amount > 10 and (abs(errorOrg) < 0.1) and (abs(newbalanceDest - oldbalanceDest) < 0.1):
                prob = max(prob, 0.99)
                reasons.append("Logical Impossibility: Funds left origin but never reached destination.")
            
            # Rule: Reverse Balance (Receiving money but balance drops)
            if amount > 0 and (newbalanceDest < oldbalanceDest):
                prob = max(prob, 0.95)
                reasons.append("Anomalous Destination: Recipient balance decreased after transaction.")

            # Rule: Ghost Funds (New balance appears without origin drop)
            if amount > 10 and (newbalanceDest > oldbalanceDest + amount + 0.1) and (abs(errorOrg) > amount):
                prob = max(prob, 0.85)
                reasons.append("Suspicious Credit: Destination received more than was sent.")

        # Rule: Massive Overdraft (Transferring way more than owned)
        if (amount > oldbalanceOrg * 5) and (oldbalanceOrg > 0):
            prob = max(prob, 0.90)
            reasons.append(f"High Overdraft: Amount is {amount/oldbalanceOrg:.1f}x the available balance.")

        is_fraud = prob >= 0.5
        risk = "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.3 else "LOW"

        return jsonify({
            "is_fraud": is_fraud,
            "fraud_probability": round(prob, 6),
            "risk_level": risk,
            "reasons": reasons,
            "model": metadata["model_name"] if metadata else "LightGBM",
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as exc:
        print(f"[ERROR] Prediction failed: {exc}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc), "status": "failed"}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
