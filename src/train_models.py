"""
Equipment Health Engine - Model Training
ML-004: Enterprise Predictive Maintenance Platform

Trains:
  1. RUL Regression model  -> predicts Remaining Useful Life (hours)
  2. Failure Classification -> predicts failure_within_24h (binary)

Split strategy: machine-level split (not random row split) to prevent
leakage between train/test since rows from the same machine are
temporally correlated.
"""
import pandas as pd
import numpy as np
import joblib, json
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from xgboost import XGBRegressor, XGBClassifier

FEATURES_PATH = "data/features.parquet"
DROP_COLS = ["timestamp", "machine_id", "failure_type", "rul_hours",
             "failure_within_24h", "estimated_repair_cost"]

def load_data():
    df = pd.read_parquet(FEATURES_PATH)
    return df

def machine_split(df, test_frac=0.25, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df["machine_id"]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

def get_feature_cols(df):
    return [c for c in df.columns if c not in DROP_COLS]

def train_rul_model(train, test, feat_cols):
    X_train, y_train = train[feat_cols], train["rul_hours"]
    X_test, y_test = test[feat_cols], test["rul_hours"]

    model = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": float(r2_score(y_test, preds)),
    }
    return model, metrics, (y_test, preds)

def train_failure_model(train, test, feat_cols):
    X_train, y_train = train[feat_cols], train["failure_within_24h"]
    X_test, y_test = test[feat_cols], test["failure_within_24h"]

    # class imbalance handling
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "Accuracy": float(accuracy_score(y_test, preds)),
        "Precision": float(precision_score(y_test, preds)),
        "Recall": float(recall_score(y_test, preds)),
        "F1": float(f1_score(y_test, preds)),
        "ROC_AUC": float(roc_auc_score(y_test, proba)),
        "ConfusionMatrix": confusion_matrix(y_test, preds).tolist(),
    }
    return model, metrics, (y_test, proba, preds)

def compute_health_score(rul_pred, failure_proba, rul_cap=500):
    """Composite Health Score (0-100): blends normalized RUL and inverse
    failure probability, per the case study's Equipment Health Engine spec."""
    rul_norm = np.clip(rul_pred / rul_cap, 0, 1)
    health = 100 * (0.5 * rul_norm + 0.5 * (1 - failure_proba))
    return np.clip(health, 0, 100)

def main():
    df = load_data()
    train, test = machine_split(df)
    feat_cols = get_feature_cols(df)

    print(f"Train machines: {train['machine_id'].nunique()}, rows: {len(train)}")
    print(f"Test machines: {test['machine_id'].nunique()}, rows: {len(test)}")
    print(f"Feature count: {len(feat_cols)}")

    rul_model, rul_metrics, rul_preds = train_rul_model(train, test, feat_cols)
    print("\nRUL Regression metrics:", rul_metrics)

    fail_model, fail_metrics, fail_preds = train_failure_model(train, test, feat_cols)
    print("\nFailure Classification metrics:", fail_metrics)

    # Health score on test set
    health = compute_health_score(rul_preds[1], fail_preds[1])
    test = test.assign(pred_rul=rul_preds[1], pred_fail_proba=fail_preds[1], health_score=health)

    # Persist
    joblib.dump(rul_model, "models/rul_model.joblib")
    joblib.dump(fail_model, "models/failure_model.joblib")
    with open("models/feature_cols.json", "w") as f:
        json.dump(feat_cols, f)
    with open("models/metrics.json", "w") as f:
        json.dump({"rul": rul_metrics, "failure": fail_metrics}, f, indent=2)
    test.to_parquet("data/test_predictions.parquet", index=False)

    print("\nModels + metrics saved to models/")

if __name__ == "__main__":
    main()
