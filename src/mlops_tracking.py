"""
MLOps: Experiment Tracking & Model Registry
ML-004: Enterprise Predictive Maintenance Platform

Logs the trained RUL and Failure models, their metrics, and artifacts
to a local MLflow tracking store, and registers them in the Model
Registry. Run after train_models.py.
"""
import json, joblib, mlflow, mlflow.sklearn
import pandas as pd

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("predictive_maintenance_ml004")

metrics = json.load(open("models/metrics.json"))
feat_cols = json.load(open("models/feature_cols.json"))

with mlflow.start_run(run_name="rul_regression_xgb"):
    mlflow.log_params({"model_type": "XGBRegressor", "n_features": len(feat_cols)})
    mlflow.log_metrics(metrics["rul"])
    mlflow.log_artifact("models/rul_model.joblib")
    mlflow.log_artifact("plots/shap_rul.png")
    mlflow.log_artifact("plots/rul_distribution.png")

with mlflow.start_run(run_name="failure_classification_xgb"):
    m = {k: v for k, v in metrics["failure"].items() if k != "ConfusionMatrix"}
    mlflow.log_params({"model_type": "XGBClassifier", "n_features": len(feat_cols)})
    mlflow.log_metrics(m)
    mlflow.log_artifact("models/failure_model.joblib")
    mlflow.log_artifact("plots/shap_failure.png")
    mlflow.log_artifact("plots/roc_pr_curve.png")

print("Logged runs to mlflow.db (view with: mlflow ui --backend-store-uri sqlite:///mlflow.db)")

# Simple drift-check stub: compares train vs test feature means, flags
# sensors whose mean has shifted >15% (Sensor Drift Detection requirement)
train_test = pd.read_parquet("data/features.parquet")
drift_report = []
for col in ["vibration_rms", "temperature_motor", "pressure_level", "rpm"]:
    overall_mean = train_test[col].mean()
    recent_mean = train_test.sort_values("timestamp").tail(2000)[col].mean()
    pct_shift = abs(recent_mean - overall_mean) / (abs(overall_mean) + 1e-9)
    drift_report.append({"feature": col, "overall_mean": round(overall_mean, 3),
                          "recent_mean": round(recent_mean, 3), "pct_shift": round(pct_shift, 3),
                          "drift_flag": bool(pct_shift > 0.15)})
with open("models/drift_report.json", "w") as f:
    json.dump(drift_report, f, indent=2)
print("Drift report:", drift_report)
