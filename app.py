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
    oldbalanceOrg: float = Field(..., ge=0)
    oldbalanceDest: float = Field(..., ge=0)
    destType: str = "CUSTOMER"

    @field_validator("type", "destType", mode="before")
    @classmethod
    def normalize_upper(cls, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip().upper()

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
    if transaction.oldbalanceOrg > 0 and transaction.amount > transaction.oldbalanceOrg * 5:
        reasons.append(
            f"High overdraft: amount is {transaction.amount / transaction.oldbalanceOrg:.1f}x the available balance."
        )

    is_fraud = prob >= threshold
    risk = "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.3 else "LOW"

    return jsonify(
        {
            "is_fraud": bool(is_fraud),
            "fraud_probability": round(prob, 6),
            "decision_threshold": threshold,
            "risk_level": risk,
            "reasons": reasons,
            "model": metadata["model_name"] if metadata else "LightGBM",
            "features_used": MODEL_INPUT_COLUMNS,
            "timestamp": datetime.now().isoformat(),
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
