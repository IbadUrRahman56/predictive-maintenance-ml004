"""
Predictive Maintenance API
ML-004: Enterprise Predictive Maintenance Platform

Run: uvicorn api:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd, joblib, json, numpy as np

app = FastAPI(
    title="Predictive Maintenance API",
    description="Predicts equipment health, failure probability, and RUL from sensor readings.",
    version="1.0.0",
)

rul_model = joblib.load("models/rul_model.joblib")
fail_model = joblib.load("models/failure_model.joblib")
FEATURE_COLS = json.load(open("models/feature_cols.json"))

class SensorReading(BaseModel):
    machine_id: int
    machine_type: str          # CNC, Pump, Compressor, Robotic Arm
    operating_mode: str        # idle, normal, peak
    vibration_rms: float
    temperature_motor: float
    current: float
    pressure_level: float
    rpm: float
    hours_since_maintenance: float
    ambient_temp: float
    humidity: float
    voltage: float
    machine_load: float

class PredictionResponse(BaseModel):
    machine_id: int
    health_score: float
    failure_probability: float
    predicted_rul_hours: float
    recommendation: str

def recommend(health_score: float, fail_proba: float) -> str:
    if fail_proba >= 0.8 or health_score < 20:
        return "Emergency Shutdown"
    if fail_proba >= 0.5 or health_score < 40:
        return "Immediate Repair"
    if fail_proba >= 0.25 or health_score < 60:
        return "Scheduled Maintenance"
    if health_score < 80:
        return "Continue Monitoring"
    return "Normal Operation"

def build_feature_row(r: SensorReading) -> pd.DataFrame:
    # Build a single-row frame matching training feature schema.
    # Rolling/trend features fall back to the instantaneous reading
    # (no history available for a single live reading via the API).
    row = {c: 0.0 for c in FEATURE_COLS}
    base = {
        "vibration_rms": r.vibration_rms, "temperature_motor": r.temperature_motor,
        "current": r.current, "pressure_level": r.pressure_level, "rpm": r.rpm,
        "hours_since_maintenance": r.hours_since_maintenance, "ambient_temp": r.ambient_temp,
        "humidity": r.humidity, "voltage": r.voltage, "machine_load": r.machine_load,
        "vibration_energy": r.vibration_rms ** 2,
        "temperature_gradient": r.temperature_motor - r.ambient_temp,
        "utilization_rate": r.machine_load / 100.0,
        "operating_hours": r.hours_since_maintenance,
        "cum_failures": 0.0,
    }
    for col in ["vibration_rms", "temperature_motor", "current", "pressure_level", "rpm"]:
        base[f"{col}_roll_mean"] = base.get(col, 0.0)
        base[f"{col}_roll_std"] = 0.0
    for col in ["vibration_rms", "temperature_motor"]:
        base[f"{col}_trend"] = 0.0
    for k, v in base.items():
        if k in row:
            row[k] = v
    mtype_col = f"mtype_{r.machine_type}"
    mode_col = f"mode_{r.operating_mode}"
    if mtype_col in row:
        row[mtype_col] = 1
    if mode_col in row:
        row[mode_col] = 1
    return pd.DataFrame([row])[FEATURE_COLS]

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    try:
        X = build_feature_row(reading)
        rul_pred = float(rul_model.predict(X)[0])
        fail_proba = float(fail_model.predict_proba(X)[0, 1])
        health = float(np.clip(100 * (0.5 * np.clip(rul_pred / 500, 0, 1) + 0.5 * (1 - fail_proba)), 0, 100))
        return PredictionResponse(
            machine_id=reading.machine_id,
            health_score=round(health, 2),
            failure_probability=round(fail_proba, 4),
            predicted_rul_hours=round(rul_pred, 1),
            recommendation=recommend(health, fail_proba),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
