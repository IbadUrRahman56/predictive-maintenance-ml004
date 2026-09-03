# ML-004: Enterprise Predictive Maintenance & Industrial Equipment Failure Intelligence Platform

Internship case study project — Ezitech Engineering Framework (EEF).
Team: 2 ML Engineers | Duration: 4 Weeks

## What this project does

Given IoT sensor readings from industrial machines (vibration, temperature,
pressure, RPM, current, voltage, humidity, load), this platform predicts:

- **Health Score** (0–100) per machine
- **Failure probability** within the next 24 hours
- **Remaining Useful Life (RUL)** in hours
- A **maintenance recommendation** (Emergency Shutdown / Immediate Repair /
  Scheduled Maintenance / Continue Monitoring / Normal Operation)

## Project structure

```
proj/
├── data/
│   ├── predictive_maintenance.csv    # raw input data
│   ├── features.parquet              # engineered feature matrix (generated)
│   └── test_predictions.parquet      # held-out predictions (generated)
├── src/
│   ├── feature_engineering.py        # builds the 38-feature matrix
│   ├── train_models.py               # trains RUL regressor + failure classifier
│   ├── explain_and_visualize.py      # SHAP explainability + dashboard plots
│   ├── mlops_tracking.py             # MLflow logging + drift detection
│   └── api.py                        # FastAPI prediction service
├── models/                           # trained models + metrics (generated)
├── plots/                            # dashboard visuals (generated)
├── reports/                          # Model Evaluation Report (.docx)
└── mlflow.db                         # MLflow tracking store (generated)
```

## How to run the pipeline end-to-end

```bash
pip install -r requirements.txt

# 1. Build features from raw sensor data
python src/feature_engineering.py

# 2. Train the RUL regression + failure classification models
python src/train_models.py

# 3. Generate SHAP explainability plots + operations dashboard visuals
python src/explain_and_visualize.py

# 4. Log runs to MLflow + run drift check
python src/mlops_tracking.py

# 5. Serve predictions via API
cd src && uvicorn api:app --reload --port 8000
# Interactive docs: http://localhost:8000/docs
```

## Requirements

```
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
pyarrow
fastapi
uvicorn
mlflow
```

## Model summary

| Model | Metric | Value |
|---|---|---|
| RUL Regression (XGBoost) | MAE | ~17.8 hours |
| RUL Regression (XGBoost) | R² | 0.26 |
| Failure Classification (XGBoost) | ROC-AUC | 0.987 |
| Failure Classification (XGBoost) | Recall | 94.6% |
| Failure Classification (XGBoost) | Precision | 58.8% |

Full methodology, EDA, and evaluation are in
`reports/ML-004_Model_Evaluation_Report.docx`.

## Key design decisions

- **Machine-level train/test split** (not random row split) — prevents
  leakage between readings from the same machine.
- **Recall-weighted failure model** — a missed failure is costlier than a
  false alarm in this domain, so class weighting favors catching failures.
- **Composite health score** blends normalized RUL and inverse failure
  probability — gives operators one number instead of two separate ones.

## Deployment notes

- Swap the local `data/` CSV ingestion for a Kafka/streaming source to hit
  the case study's "Real-Time Kafka Sensor Streaming" bonus challenge.
- Add PostgreSQL for persisting live readings/predictions and Redis for
  caching hot predictions, per the case study's backend spec.
- Promote `mlflow.db` to a shared tracking server for multi-user experiment
  tracking across the team.

## Next steps / open items for the group

- Tune the failure-alert threshold (currently 0.5) once false-alarm cost is known.
- Try an LSTM on raw sensor windows to improve RUL R².
- Build the interactive Operations Dashboard (React/Streamlit) on top of
  `plots/` and the `/predict` API — currently delivered as static visuals.
