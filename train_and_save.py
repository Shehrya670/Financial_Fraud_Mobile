"""
train_and_save.py
-----------------
Trains all fraud-detection models, picks the best, and saves:
  models/best_model.pkl          ← standard pickle
  models/best_model.joblib       ← joblib (faster for sklearn)
  models/scaler.pkl
  models/scaler.joblib
  models/feature_names.joblib
  models/model_metadata.json

Run:  python train_and_save.py
"""

import os, json, pickle, joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import warnings; warnings.filterwarnings("ignore")

# Optional
try:
    from xgboost import XGBClassifier; HAS_XGB = True
except ImportError:
    HAS_XGB = False; print("XGBoost not found – skipping")

try:
    from lightgbm import LGBMClassifier; HAS_LGB = True
except ImportError:
    HAS_LGB = False; print("LightGBM not found – skipping")

# ── 1. Load dataset ──────────────────────────────────────────────────────────
CSV = "PS_20174392719_1491204439457_log.csv"
print(f"Loading {CSV} (10 % sample)…")
df = pd.read_csv(CSV).sample(frac=0.1, random_state=42)
print(f"  Shape: {df.shape}")

# ── 2. Preprocess ────────────────────────────────────────────────────────────
drop_cols = ["nameOrig", "nameDest", "isFlaggedFraud", "step"]
df_clean  = df.drop(columns=drop_cols)
df_clean  = pd.get_dummies(df_clean, columns=["type"], drop_first=True)

X = df_clean.drop("isFraud", axis=1)
y = df_clean["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler         = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)

feature_names  = X_train_scaled.columns.tolist()
print(f"  Features: {feature_names}")

# ── 3. SMOTE ─────────────────────────────────────────────────────────────────
print("Applying SMOTE…")
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train_scaled, y_train)

# ── 4. Train all models ───────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, C=0.1),
    "Decision Tree":       DecisionTreeClassifier(random_state=42, max_depth=10),
    "Random Forest":       RandomForestClassifier(n_estimators=50, random_state=42,
                                                   n_jobs=-1, max_depth=10),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=50, random_state=42,
                                                       max_depth=5),
}
if HAS_XGB:
    models["XGBoost"]  = XGBClassifier(n_estimators=50, random_state=42, n_jobs=-1,
                                        max_depth=5, eval_metric="logloss", verbosity=0)
if HAS_LGB:
    models["LightGBM"] = LGBMClassifier(n_estimators=50, random_state=42, n_jobs=-1,
                                         max_depth=5, verbose=-1)

results = {}
for name, m in models.items():
    print(f"  Training {name}…", end=" ", flush=True)
    m.fit(X_train_sm, y_train_sm)
    yp    = m.predict(X_test_scaled)
    yprob = m.predict_proba(X_test_scaled)[:, 1]
    results[name] = {
        "model":         m,
        "roc_auc":       roc_auc_score(y_test, yprob),
        "f1":            f1_score(y_test, yp),
        "precision":     precision_score(y_test, yp),
        "recall":        recall_score(y_test, yp),
    }
    r = results[name]
    print(f"ROC-AUC={r['roc_auc']:.4f}  F1={r['f1']:.4f}")

# ── 5. Select best ────────────────────────────────────────────────────────────
best_name  = max(results, key=lambda n: results[n]["roc_auc"])
best_model = results[best_name]["model"]
best_r     = results[best_name]
print(f"\nBest model: {best_name}  (ROC-AUC={best_r['roc_auc']:.4f})")

# ── 6. Save artefacts ─────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

# joblib (preferred for sklearn – smaller, faster)
joblib.dump(best_model,   "models/best_model.joblib")
joblib.dump(scaler,       "models/scaler.joblib")
joblib.dump(feature_names,"models/feature_names.joblib")

# pickle (alternative)
with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# metadata JSON
meta = {
    "model_name":    best_name,
    "roc_auc":       best_r["roc_auc"],
    "f1_score":      best_r["f1"],
    "precision":     best_r["precision"],
    "recall":        best_r["recall"],
    "feature_names": feature_names,
}
with open("models/model_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

# Print summary
print("\n-- Saved artefacts --")
for fn in os.listdir("models"):
    size = os.path.getsize(f"models/{fn}")
    print(f"  models/{fn:<30}  {size/1024:.1f} KB")
print("\nDone! Start the app with:  python app.py")
