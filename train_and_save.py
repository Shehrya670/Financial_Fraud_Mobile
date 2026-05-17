"""
Train the fraud-detection model and save a single serving pipeline.

The production pipeline intentionally excludes post-transaction leakage fields
such as new balances and balance-error calculations. It uses only fields that
can be known before or at authorization time.
"""

import json
import os
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_pipeline import CATEGORICAL_FEATURES, MODEL_INPUT_COLUMNS, NUMERIC_FEATURES, NoLeakageFeatureBuilder


CSV_PATH = Path("PS_20174392719_1491204439457_log.csv")
MODEL_DIR = Path("models")
SAMPLE_FRAC = float(os.environ.get("TRAIN_SAMPLE_FRAC", "0.2"))
RANDOM_STATE = 42


def load_dataset() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

    print(f"Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    if 0 < SAMPLE_FRAC < 1:
        print(f"Sampling {SAMPLE_FRAC:.0%} of rows for repeatable local training...")
        df = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)

    return df.sort_values("step").reset_index(drop=True)


def time_split(df: pd.DataFrame):
    train_end = int(len(df) * 0.70)
    val_end = int(len(df) * 0.85)
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    return train_df, val_df, test_df


def build_pipeline(y_train: pd.Series) -> Pipeline:
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / max(pos, 1)
    print(f"Training fraud ratio: 1:{scale_pos_weight:.1f}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("type", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    preprocessor.set_output(transform="pandas")

    model = LGBMClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    return Pipeline(
        steps=[
            ("features", NoLeakageFeatureBuilder()),
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, dict]:
    candidates = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_metrics = {"f1": -1.0}

    for threshold in candidates:
        preds = probabilities >= threshold
        metrics = classification_metrics(y_true, probabilities, preds)
        if metrics["f1_score"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = {"f1": metrics["f1_score"], **metrics}

    best_metrics.pop("f1", None)
    return best_threshold, best_metrics


def classification_metrics(y_true, probabilities, preds) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "f1_score": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def main():
    df = load_dataset()
    train_df, val_df, test_df = time_split(df)
    print(f"Rows: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")

    X_train = train_df.drop(columns=["isFraud"])
    y_train = train_df["isFraud"]
    X_val = val_df.drop(columns=["isFraud"])
    y_val = val_df["isFraud"]
    X_test = test_df.drop(columns=["isFraud"])
    y_test = test_df["isFraud"]

    pipeline = build_pipeline(y_train)
    pipeline.fit(X_train, y_train)

    val_prob = pipeline.predict_proba(X_val)[:, 1]
    threshold, val_metrics = choose_threshold(y_val, val_prob)
    print(f"Selected threshold={threshold:.2f} on validation F1={val_metrics['f1_score']:.4f}")

    test_prob = pipeline.predict_proba(X_test)[:, 1]
    test_pred = test_prob >= threshold
    test_metrics = classification_metrics(y_test, test_prob, test_pred)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "pipeline.joblib")
    joblib.dump(pipeline, MODEL_DIR / "best_model.joblib")
    joblib.dump(MODEL_INPUT_COLUMNS, MODEL_DIR / "feature_names.joblib")
    with open(MODEL_DIR / "best_model.pkl", "wb") as fh:
        pickle.dump(pipeline, fh)

    for legacy_artifact in ("scaler.joblib", "scaler.pkl"):
        legacy_path = MODEL_DIR / legacy_artifact
        if legacy_path.exists():
            legacy_path.unlink()

    metadata = {
        "model_name": "LightGBM",
        "artifact": "pipeline.joblib",
        "training_sample_fraction": SAMPLE_FRAC,
        "split_strategy": "time_ordered_70_15_15_by_step",
        "leakage_policy": "serving_safe_no_newbalance_or_balance_error_features",
        "decision_threshold": threshold,
        "feature_names": MODEL_INPUT_COLUMNS,
        "validation": val_metrics,
        "test": test_metrics,
        **test_metrics,
    }

    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print("\nSaved artifacts:")
    for path in sorted(MODEL_DIR.iterdir()):
        print(f"  {path} ({path.stat().st_size / 1024:.1f} KB)")
    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
