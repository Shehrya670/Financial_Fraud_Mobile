import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ml_pipeline import DESTINATION_TYPES, MODEL_INPUT_COLUMNS, TRANSACTION_TYPES


app = Flask(__name__)
MODEL_DIR = Path("models")


class TransactionPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str = Field(..., description="PAYMENT, TRANSFER, CASH_OUT, CASH_IN, or DEBIT")
    amount: float = Field(..., gt=0)
    oldbalanceOrg: float = Field(default=0.0, ge=0)
    oldbalanceDest: float = Field(default=0.0, ge=0)
    destType: str = "CUSTOMER"

    @field_validator("type", "destType", mode="before")
    @classmethod
    def normalize_upper(cls, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip().upper()

    @field_validator("amount", "oldbalanceOrg", "oldbalanceDest", mode="before")
    @classmethod
    def parse_optional_floats(cls, value):
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return 0.0
            try:
                return float(value)
            except ValueError:
                raise ValueError("must be a valid number")
        return value

    @field_validator("type")
    @classmethod
    def valid_type(cls, value):
        if value not in TRANSACTION_TYPES:
            raise ValueError(f"must be one of {sorted(TRANSACTION_TYPES)}")
        return value

    @field_validator("destType")
    @classmethod
    def valid_dest_type(cls, value):
        if value not in DESTINATION_TYPES:
            raise ValueError(f"must be one of {sorted(DESTINATION_TYPES)}")
        return value

    def to_model_frame(self) -> pd.DataFrame:
        is_merchant = int(self.destType == "MERCHANT")
        return pd.DataFrame(
            [
                {
                    "type": self.type,
                    "amount": self.amount,
                    "oldbalanceOrg": self.oldbalanceOrg,
                    "oldbalanceDest": self.oldbalanceDest,
                    "isMerchant": is_merchant,
                }
            ]
        )


def load_artifacts():
    pipeline_path = MODEL_DIR / "pipeline.joblib"
    if not pipeline_path.exists():
        pipeline_path = MODEL_DIR / "best_model.joblib"

    pipeline = joblib.load(pipeline_path)
    feature_names = joblib.load(MODEL_DIR / "feature_names.joblib")
    with open(MODEL_DIR / "model_metadata.json", encoding="utf-8") as fh:
        metadata = json.load(fh)

    return pipeline, feature_names, metadata


try:
    model_pipeline, feature_names, metadata = load_artifacts()
    print(f"[OK] Loaded model pipeline: {metadata['model_name']}")
    print(f"[OK] Features: {feature_names}")
except Exception as exc:
    print(f"[WARN] Could not load model artifacts: {exc}")
    print("       Run train_and_save.py, then restart this server.")
    model_pipeline = feature_names = metadata = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    if metadata is None:
        return jsonify(
            {
                "status": "no_model",
                "message": "Run train_and_save.py to train and save the model pipeline first.",
            }
        ), 503

    return jsonify(
        {
            "status": "healthy",
            "model": metadata["model_name"],
            "artifact": metadata.get("artifact", "best_model.joblib"),
            "decision_threshold": metadata.get("decision_threshold", 0.5),
            "roc_auc": metadata.get("roc_auc"),
            "pr_auc": metadata.get("pr_auc"),
            "f1": metadata.get("f1_score"),
            "precision": metadata.get("precision"),
            "recall": metadata.get("recall"),
            "false_positive_rate": metadata.get("false_positive_rate"),
            "false_negative_rate": metadata.get("false_negative_rate"),
            "confusion_matrix": metadata.get("confusion_matrix"),
            "features": feature_names,
        }
    )


def validation_error_response(exc: ValidationError):
    return jsonify(
        {
            "error": "Invalid transaction payload",
            "details": [
                {
                    "field": ".".join(str(part) for part in err["loc"]),
                    "message": err["msg"],
                }
                for err in exc.errors()
            ],
        }
    ), 400


@app.route("/predict", methods=["POST"])
def predict():
    if model_pipeline is None:
        return jsonify({"error": "Model not loaded. Run train_and_save.py first."}), 503

    try:
        payload = request.get_json(force=True)
        transaction = TransactionPayload.model_validate(payload)
    except ValidationError as exc:
        return validation_error_response(exc)
    except Exception as exc:
        return jsonify({"error": "Invalid JSON payload", "details": str(exc)}), 400

    row = transaction.to_model_frame()
    prob = float(model_pipeline.predict_proba(row)[0][1])
    threshold = float(metadata.get("decision_threshold", 0.5)) if metadata else 0.5

    reasons = []
    
    # Deterministic Guardrail: Massive Anomalous Transfers from Empty/Dormant Accounts
    if transaction.type in ("TRANSFER", "CASH_OUT") and transaction.oldbalanceOrg == 0 and transaction.amount > 100000:
        reasons.append(
            f"Dormant Account Abuse: Large {transaction.type} of (${transaction.amount:,.2f}) initiated from a newly-created or zero-balance account."
        )

    # Decision Logic (Override ML prediction only for extreme guardrails)
    if reasons:
        is_fraud = True
        prob = max(prob, 0.985)  # Force extremely high fraud probability

        risk = "HIGH"
    else:
        is_fraud = prob >= threshold
        risk = "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.3 else "LOW"

    # 4. Explainable AI (XAI) Basis attribution generator
    explainability = []
    
    # 4a. Type impact
    if transaction.type == "PAYMENT":
        explainability.append({
            "feature": "Transaction Type: PAYMENT",
            "impact": "DECREASED RISK",
            "description": "Payment transactions represent commercial trades with historically negligible baseline fraud risk.",
            "direction": "down",
            "magnitude": 42
        })
    elif transaction.type in ("TRANSFER", "CASH_OUT"):
        explainability.append({
            "feature": f"Transaction Type: {transaction.type}",
            "impact": "INCREASED RISK" if prob > 0.3 else "NEUTRAL",
            "description": f"{transaction.type} is an immediate-settlement method representing a higher threat baseline.",
            "direction": "up" if prob > 0.3 else "neutral",
            "magnitude": 35 if prob > 0.3 else 15
        })
    else:
        explainability.append({
            "feature": f"Transaction Type: {transaction.type}",
            "impact": "NEUTRAL",
            "description": "This transaction type has a statistically standard baseline profile.",
            "direction": "neutral",
            "magnitude": 10
        })

    # 4b. Amount impact
    if transaction.amount < 1000:
        explainability.append({
            "feature": f"Transaction Amount: ${transaction.amount:,.2f}",
            "impact": "DECREASED RISK",
            "description": "Low transaction value is fully consistent with everyday personal retail habits.",
            "direction": "down",
            "magnitude": 28
        })
    elif transaction.amount > 200000:
        explainability.append({
            "feature": f"Transaction Amount: ${transaction.amount:,.2f}",
            "impact": "INCREASED RISK",
            "description": "Extremely large transfer value is a highly anomalous statistical outlier.",
            "direction": "up",
            "magnitude": 45
        })
    else:
        explainability.append({
            "feature": f"Transaction Amount: ${transaction.amount:,.2f}",
            "impact": "NEUTRAL",
            "description": "The amount is within standard mid-tier validation boundaries.",
            "direction": "neutral",
            "magnitude": 12
        })

    # 4c. Balance Coverage impact
    if transaction.oldbalanceOrg > 0:
        coverage = transaction.amount / transaction.oldbalanceOrg
        if coverage <= 0.20:
            explainability.append({
                "feature": f"Balance Drawdown Ratio: {coverage:.1%}",
                "impact": "DECREASED RISK",
                "description": f"Draws only {coverage:.1%} of origin account capital, representing high liquidity and safe reserve cover.",
                "direction": "down",
                "magnitude": 25
            })
        elif coverage > 1.0:
            explainability.append({
                "feature": "Insufficient Reserve Funds",
                "impact": "INCREASED RISK",
                "description": f"Draw exceeds available reserve balance by ${(transaction.amount - transaction.oldbalanceOrg):,.2f}.",
                "direction": "up",
                "magnitude": 85
            })
        else:
            explainability.append({
                "feature": f"Balance Drawdown Ratio: {coverage:.1%}",
                "impact": "NEUTRAL",
                "description": "Draw is fully backed by sufficient historical reserves in the account.",
                "direction": "neutral",
                "magnitude": 10
            })
    else:
        if transaction.amount > 10000:
            explainability.append({
                "feature": "Empty Account Origin",
                "impact": "INCREASED RISK",
                "description": "High-value settlement initiated from a zero-balance or dormant origin account.",
                "direction": "up",
                "magnitude": 65
            })
        else:
            explainability.append({
                "feature": "Zero Balance Origin",
                "impact": "NEUTRAL",
                "description": "Transaction initiated from a zero-balance account with low-volume activity.",
                "direction": "neutral",
                "magnitude": 15
            })

    # 4d. Recipient profile impact
    if transaction.destType == "MERCHANT":
        explainability.append({
            "feature": "Recipient: Verified Merchant",
            "impact": "DECREASED RISK",
            "description": "Transfers to officially registered business/merchant merchant terminals have ultra-low risk profiles.",
            "direction": "down",
            "magnitude": 35
        })
    else:
        explainability.append({
            "feature": "Recipient: Individual Customer",
            "impact": "NEUTRAL",
            "description": "Destination is an unverified individual customer profile, representing standard baseline threat.",
            "direction": "neutral",
            "magnitude": 8
        })

    return jsonify(
        {
            "is_fraud": bool(is_fraud),
            "fraud_probability": round(prob, 6),
            "decision_threshold": threshold,
            "risk_level": risk,
            "reasons": reasons,
            "explainability": explainability,
            "model": "Hybrid (LGBM + Guardrails)" if reasons else (metadata["model_name"] if metadata else "LightGBM"),
            "features_used": MODEL_INPUT_COLUMNS,
            "timestamp": datetime.now().isoformat(),
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
