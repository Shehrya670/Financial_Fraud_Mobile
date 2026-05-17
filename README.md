# FraudShield - Mobile Transaction Fraud Detection

FraudShield is a Flask application that serves a LightGBM fraud-detection model for mobile money transactions. The current serving model is trained on the PaySim dataset with a single scikit-learn pipeline so training and inference share the same preprocessing path.

## What Changed

- Uses one saved `Pipeline` artifact: `models/pipeline.joblib`.
- Uses a time-ordered train/validation/test split by `step`.
- Tunes the decision threshold on the validation split instead of using a fixed `0.5`.
- Reports ROC-AUC, PR-AUC, F1, precision, recall, FPR, FNR, and confusion matrix.
- Excludes post-transaction leakage fields from serving features.
- Validates API input with Pydantic and returns proper `400` errors for bad payloads.
- Adds basic pytest coverage for model health, prediction, validation, and artifacts.

## Project Structure

```text
Financial_Fraud_Mobile/
|-- app.py                         # Flask API and dashboard route
|-- ml_pipeline.py                 # Shared feature contract and transformer
|-- train_and_save.py              # Training, evaluation, threshold tuning, artifact saving
|-- requirements.txt               # Runtime and test dependencies
|-- runtime.txt                    # Python runtime for deployment
|-- Procfile                       # Gunicorn start command
|-- render.yaml                    # Render.com service config
|-- models/
|   |-- pipeline.joblib            # Preferred production artifact
|   |-- best_model.joblib          # Compatibility copy of the same pipeline
|   |-- best_model.pkl             # Pickle compatibility copy
|   |-- feature_names.joblib       # Serving feature contract
|   |-- model_metadata.json        # Metrics, split strategy, threshold, features
|-- templates/
|   |-- index.html                 # Web dashboard
|-- tests/
|   |-- test_app.py                # API and artifact tests
|-- Fraud_Detection_Analysis.ipynb # Exploratory notebook
```

## Current Model

The saved model is a LightGBM classifier wrapped in a scikit-learn `Pipeline`.

Current test metrics from the regenerated artifacts:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.9986 |
| PR-AUC | 0.8865 |
| F1 | 0.7997 |
| Precision | 0.9528 |
| Recall | 0.6890 |
| False positive rate | 0.000142 |
| False negative rate | 0.3110 |
| Decision threshold | 0.99 |

Test confusion matrix:

|  | Predicted legitimate | Predicted fraud |
|---|---:|---:|
| Actual legitimate | 190061 | 27 |
| Actual fraud | 246 | 545 |

## Serving Features

The production model intentionally avoids post-transaction leakage fields such as `newbalanceOrig`, `newbalanceDest`, `errorBalanceOrg`, and `errorBalanceDest`.

Current serving features:

| Feature | Description |
|---|---|
| `type` | Transaction type: `PAYMENT`, `TRANSFER`, `CASH_OUT`, `CASH_IN`, or `DEBIT` |
| `amount` | Transaction amount |
| `oldbalanceOrg` | Origin account balance before the transaction |
| `oldbalanceDest` | Destination account balance before the transaction |
| `isMerchant` | Derived from `destType` in the API or `nameDest` during training |

## Training Pipeline

```text
Raw PaySim CSV
-> optional repeatable 20% sample
-> sort by step
-> time-ordered 70/15/15 train/validation/test split
-> no-leakage feature builder
-> ColumnTransformer:
   - StandardScaler for numeric features
   - OneHotEncoder for transaction type
-> LightGBM with balanced class weights
-> tune threshold on validation F1
-> evaluate on held-out future-like test split
-> save pipeline and metadata
```

Retrain with:

```bash
python train_and_save.py
```

By default training uses a repeatable 20% sample. To train on a different fraction:

```bash
set TRAIN_SAMPLE_FRAC=1.0
python train_and_save.py
```

## API

### `GET /health`

Returns model status, metrics, selected threshold, confusion matrix, and feature contract.

### `POST /predict`

Request:

```json
{
  "type": "TRANSFER",
  "destType": "CUSTOMER",
  "amount": 500000,
  "oldbalanceOrg": 500000,
  "oldbalanceDest": 0
}
```

Response:

```json
{
  "is_fraud": true,
  "fraud_probability": 0.992345,
  "decision_threshold": 0.99,
  "risk_level": "HIGH",
  "reasons": [],
  "model": "LightGBM",
  "features_used": ["type", "amount", "oldbalanceOrg", "oldbalanceDest", "isMerchant"],
  "timestamp": "2026-05-17T21:00:00"
}
```

Invalid payloads return `400` with field-level details.

## Local Setup

```bash
pip install -r requirements.txt
python app.py
```

Open:

```text
http://localhost:5000
```

## Tests

```bash
pytest
```

The tests cover:

- `/health` returns model metadata.
- `/predict` accepts a valid transaction.
- invalid input is rejected.
- saved feature order matches metadata.
- model artifacts load on a fresh app import.

## Deployment

Render/Heroku-style start command:

```bash
gunicorn app:app
```

The model artifact is committed under `models/`, so deployment does not need the full PaySim CSV.

## Dataset

PaySim simulated mobile money transactions:

https://www.kaggle.com/datasets/ealaxi/paysim1

The raw CSV is intentionally ignored by git because it is large.
