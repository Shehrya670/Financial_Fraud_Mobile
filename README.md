# 🛡️ FraudShield — AI Fraud Detection for Mobile Transactions

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green?logo=flask)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-ROC--AUC%200.9989-orange)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)
[![Deploy](https://img.shields.io/badge/Deploy-Render.com-blue?logo=render)](https://render.com)

> An end-to-end **explainable machine learning** pipeline that detects fraudulent mobile financial transactions in real time. Trained on the **PaySim dataset** (6.3 M transactions), deployed as a Flask REST API with an interactive web dashboard.

---

## 📸 Demo

| Dashboard | Fraud Detection Result |
|---|---|
| Input transaction details via the form | Animated probability gauge + risk badge |

---

## 🔍 Research Questions

- Which ML algorithms are most effective for detecting fraudulent mobile transactions?
- How does class imbalance affect fraud detection performance?
- Can explainable AI (SHAP) improve transparency and trust in fraud detection systems?
- Which transaction features contribute most to fraud prediction?

---

## 🏗️ Project Structure

```
Financial_Fraud_Mobile/
│
├── app.py                        # Flask REST API
├── train_and_save.py             # Standalone training script
├── Procfile                      # Render/Heroku process file
├── render.yaml                   # Render.com config
├── requirements.txt              # Python dependencies
├── runtime.txt                   # Python 3.11.0
│
├── models/                       # Saved model artefacts
│   ├── best_model.pkl            # Best model (pickle)
│   ├── best_model.joblib         # Best model (joblib)
│   ├── scaler.pkl                # StandardScaler
│   ├── scaler.joblib
│   ├── feature_names.joblib      # Ordered feature list
│   └── model_metadata.json       # Name + metrics
│
├── templates/
│   └── index.html                # Interactive web dashboard
│
└── Fraud_Detection_Analysis.ipynb  # Full EDA + training notebook
```

---

## 🤖 Models Compared

| Model | ROC-AUC | F1 Score | Precision | Recall |
|---|---|---|---|---|
| **XGBoost** 🏆 | **0.9989** | 0.3975 | — | — |
| Random Forest | 0.9982 | 0.1457 | — | — |
| LightGBM | 0.9947 | 0.1799 | — | — |
| Gradient Boosting | 0.9917 | 0.1842 | — | — |
| Decision Tree | 0.9901 | 0.2273 | — | — |
| Logistic Regression | 0.9871 | 0.0388 | — | — |

> Best model selected automatically by **ROC-AUC** score. SHAP values explain feature importance for the winning model.

---

## 📊 Dataset

**PaySim — Simulated Mobile Money Transactions**  
Source: [Kaggle — ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1)

| Property | Value |
|---|---|
| Total transactions | 6,362,620 |
| Features | 11 |
| Fraud rate | ~0.13 % |
| Training sample | 10 % (random, stratified) |

**Features used after preprocessing:**

| Feature | Description |
|---|---|
| `amount` | Transaction amount |
| `oldbalanceOrg` | Origin account balance before |
| `newbalanceOrig` | Origin account balance after |
| `oldbalanceDest` | Destination account balance before |
| `newbalanceDest` | Destination account balance after |
| `type_CASH_OUT` | One-hot: CASH_OUT transaction |
| `type_DEBIT` | One-hot: DEBIT transaction |
| `type_PAYMENT` | One-hot: PAYMENT transaction |
| `type_TRANSFER` | One-hot: TRANSFER transaction |

---

## ⚙️ ML Pipeline

```
Raw CSV  →  Drop irrelevant cols  →  One-hot encode 'type'
         →  Train/Test split (80/20, stratified)
         →  StandardScaler
         →  SMOTE (oversampling minority fraud class)
         →  Train 6 models in parallel
         →  Compare ROC-AUC
         →  Save best model + scaler
         →  SHAP explainability on best model
```

---

## 🚀 Quick Start (Local)

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/Financial_Fraud_Mobile.git
cd Financial_Fraud_Mobile
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Re-train the model
Download the dataset from Kaggle, place `PS_20174392719_1491204439457_log.csv`
in the project root, then run:
```bash
python train_and_save.py
```

### 4. Start the Flask server
```bash
python app.py
```

### 5. Open the dashboard
```
http://localhost:5000
```

---

## 🌐 Web Deployment (Render.com — Free)

### One-time setup
1. Push this repo to GitHub (public)
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo and fill in:

| Field | Value |
|---|---|
| Build Command | `pip install -r requirements.txt` |
| Start Command | `gunicorn app:app` |
| Plan | Free |

4. Click **Create Web Service** → wait ~3-5 min → live URL! 🎉

### Auto-deploy on push
Every `git push` to `main` triggers an automatic redeploy on Render.

---

## 🔌 API Reference

### `GET /health`
Returns model status and performance metrics.

```json
{
  "status": "healthy",
  "model": "XGBoost",
  "roc_auc": 0.9989,
  "f1": 0.3975,
  "precision": 0.87,
  "recall": 0.76,
  "features": ["amount", "oldbalanceOrg", ...]
}
```

---

### `POST /predict`
Classify a single transaction.

**Request body:**
```json
{
  "type": "TRANSFER",
  "amount": 500000,
  "oldbalanceOrg": 500000,
  "newbalanceOrig": 0,
  "oldbalanceDest": 0,
  "newbalanceDest": 500000
}
```

**Response:**
```json
{
  "is_fraud": true,
  "fraud_probability": 0.9234,
  "risk_level": "HIGH",
  "model": "XGBoost",
  "timestamp": "2026-05-16T13:45:22"
}
```

| Field | Type | Description |
|---|---|---|
| `is_fraud` | bool | `true` if `fraud_probability ≥ 0.5` |
| `fraud_probability` | float | Raw probability [0–1] |
| `risk_level` | string | `LOW` (<0.3) · `MEDIUM` (0.3–0.7) · `HIGH` (≥0.7) |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `flask` | REST API + template rendering |
| `gunicorn` | Production WSGI server |
| `xgboost` | Best-performing classifier |
| `lightgbm` | Candidate model |
| `scikit-learn` | Preprocessing, evaluation, other models |
| `pandas` / `numpy` | Data wrangling |
| `joblib` | Model serialisation |

---

## 📁 Notebook Overview (`Fraud_Detection_Analysis.ipynb`)

| Section | Content |
|---|---|
| §1 | Setup & Dataset Download |
| §2 | Exploratory Data Analysis (EDA) |
| §3 | Data Cleaning & Preprocessing |
| §4 | Class Imbalance (SMOTE) |
| §5 | Train & Compare 6 Models |
| §5.1 | Bar charts + heatmap comparison |
| §5.2 | Best model: report, confusion matrix, ROC curve |
| §5.3 | **Save model artefacts** |
| §6 | Explainable AI (SHAP feature importance) |

---

## 🧠 Explainable AI

SHAP (SHapley Additive exPlanations) is applied to the best model to:
- Identify which features most influence fraud predictions
- Produce both **summary bar** and **beeswarm** plots
- Increase trust and transparency in the model

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute.

---

## 👤 Author

Built as part of a research project on **Explainable Machine Learning for Financial Fraud Detection**.

---

*Made with ❤️ and a lot of XGBoost*
