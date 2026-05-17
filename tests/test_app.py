import importlib
import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as fraud_app


def client():
    return fraud_app.app.test_client()


def test_health_returns_model_metadata():
    response = client().get("/health")

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["model"]
    assert "decision_threshold" in data
    assert "features" in data


def test_predict_accepts_valid_transaction():
    response = client().post(
        "/predict",
        json={
            "type": "TRANSFER",
            "destType": "CUSTOMER",
            "amount": 500000,
            "oldbalanceOrg": 500000,
            "oldbalanceDest": 0,
        },
    )

    assert response.status_code == 200
    data = response.get_json()
    assert "fraud_probability" in data
    assert "is_fraud" in data
    assert "risk_level" in data
    assert data["model"]


def test_predict_rejects_invalid_payload():
    response = client().post(
        "/predict",
        json={
            "type": "WIRE",
            "amount": -10,
            "oldbalanceOrg": 0,
            "oldbalanceDest": 0,
        },
    )

    assert response.status_code == 400
    data = response.get_json()
    assert data["error"] == "Invalid transaction payload"
    assert data["details"]


def test_feature_order_matches_metadata():
    metadata = json.loads(Path("models/model_metadata.json").read_text(encoding="utf-8"))
    saved_features = joblib.load("models/feature_names.joblib")

    assert saved_features == metadata["feature_names"]


def test_model_artifacts_load_in_fresh_import():
    importlib.reload(fraud_app)

    assert fraud_app.model_pipeline is not None
    assert fraud_app.metadata is not None
    assert fraud_app.feature_names == fraud_app.metadata["feature_names"]
