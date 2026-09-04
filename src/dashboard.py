"""
Operations Dashboard
ML-004: Enterprise Predictive Maintenance Platform

Run: streamlit run src/dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib, json, os

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide", page_icon="🏭")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data
def load_data():
    test = pd.read_parquet(os.path.join(ROOT, "data", "test_predictions.parquet"))
    raw = pd.read_csv(os.path.join(ROOT, "data", "predictive_maintenance.csv"), parse_dates=["timestamp"])
    merged = test.merge(raw[["machine_id", "machine_type"]].drop_duplicates(), on="machine_id", how="left")
    return merged, raw

@st.cache_resource
def load_models():
    rul_model = joblib.load(os.path.join(ROOT, "models", "rul_model.joblib"))
    fail_model = joblib.load(os.path.join(ROOT, "models", "failure_model.joblib"))
    feat_cols = json.load(open(os.path.join(ROOT, "models", "feature_cols.json")))
    metrics = json.load(open(os.path.join(ROOT, "models", "metrics.json")))
    return rul_model, fail_model, feat_cols, metrics

def recommend(health_score, fail_proba):
    if fail_proba >= 0.8 or health_score < 20:
        return "🔴 Emergency Shutdown"
    if fail_proba >= 0.5 or health_score < 40:
        return "🟠 Immediate Repair"
    if fail_proba >= 0.25 or health_score < 60:
        return "🟡 Scheduled Maintenance"
    if health_score < 80:
        return "🔵 Continue Monitoring"
    return "🟢 Normal Operation"

df, raw = load_data()
rul_model, fail_model, feat_cols, metrics = load_models()

st.title("🏭 Predictive Maintenance — Operations Dashboard")
st.caption("ML-004 · Enterprise Predictive Maintenance Platform · Test-set machines (held out from training)")

# ---- Top-level KPIs ----
machine_summary = df.groupby(["machine_id", "machine_type"]).agg(
    avg_health=("health_score", "mean"),
    min_health=("health_score", "min"),
    avg_fail_proba=("pred_fail_proba", "mean"),
    max_fail_proba=("pred_fail_proba", "max"),
    avg_rul=("pred_rul", "mean"),
).reset_index().sort_values("avg_health")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Machines monitored", df["machine_id"].nunique())
col2.metric("Avg fleet health score", f"{df['health_score'].mean():.0f} / 100")
col3.metric("Machines at high risk", int((machine_summary["max_fail_proba"] > 0.5).sum()))
col4.metric("Failure model ROC-AUC", f"{metrics['failure']['ROC_AUC']:.3f}")

st.divider()

# ---- Machine risk table ----
left, right = st.columns([1.3, 1])
with left:
    st.subheader("Fleet risk overview")
    display_df = machine_summary.copy()
    display_df["Recommendation"] = display_df.apply(
        lambda r: recommend(r["avg_health"], r["max_fail_proba"]), axis=1
    )
    display_df = display_df.rename(columns={
        "machine_id": "Machine", "machine_type": "Type", "avg_health": "Avg Health",
        "min_health": "Min Health", "avg_fail_proba": "Avg Fail Prob",
        "max_fail_proba": "Max Fail Prob", "avg_rul": "Avg RUL (hrs)"
    })
    st.dataframe(
        display_df.style.format({
            "Avg Health": "{:.1f}", "Min Health": "{:.1f}",
            "Avg Fail Prob": "{:.2%}", "Max Fail Prob": "{:.2%}", "Avg RUL (hrs)": "{:.1f}"
        }).background_gradient(subset=["Avg Health"], cmap="RdYlGn", vmin=0, vmax=100),
        use_container_width=True, hide_index=True
    )

with right:
    st.subheader("Health by machine type")
    fig = px.bar(
        machine_summary.groupby("machine_type")["avg_health"].mean().reset_index(),
        x="avg_health", y="machine_type", orientation="h",
        labels={"avg_health": "Avg Health Score", "machine_type": ""},
        color="avg_health", color_continuous_scale="RdYlGn", range_color=[0, 100],
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False, height=320)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---- Per-machine drill-down ----
st.subheader("Machine drill-down")
machine_id = st.selectbox("Select a machine", sorted(df["machine_id"].unique()))
mdf = df[df["machine_id"] == machine_id].merge(
    raw[["machine_id", "timestamp"]], on="machine_id", how="left"
).sort_values("timestamp") if "timestamp" not in df.columns else df[df["machine_id"] == machine_id]

c1, c2, c3 = st.columns(3)
c1.metric("Current health score", f"{mdf['health_score'].iloc[-1]:.1f}")
c2.metric("Failure probability", f"{mdf['pred_fail_proba'].iloc[-1]:.1%}")
c3.metric("Predicted RUL", f"{mdf['pred_rul'].iloc[-1]:.0f} hrs")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=mdf["health_score"].values, mode="lines", name="Health Score", line=dict(color="#2b6cb0")))
fig2.update_layout(title=f"Machine {machine_id} — Health Score Over Readings", yaxis_range=[0, 100], height=300)
st.plotly_chart(fig2, use_container_width=True)

# ---- Live "what-if" predictor ----
st.divider()
st.subheader("Live prediction — try a sensor reading")
st.caption("Simulates what the /predict API endpoint returns for a manually entered reading.")

with st.form("predict_form"):
    fc1, fc2, fc3, fc4 = st.columns(4)
    vibration = fc1.number_input("Vibration RMS", 0.0, 10.0, 2.0)
    temp = fc1.number_input("Motor Temp (°C)", 0.0, 150.0, 70.0)
    current = fc2.number_input("Current (A)", 0.0, 50.0, 10.0)
    pressure = fc2.number_input("Pressure", 0.0, 20.0, 4.0)
    rpm = fc3.number_input("RPM", 0.0, 5000.0, 1200.0)
    hours = fc3.number_input("Hours since maintenance", 0.0, 1000.0, 200.0)
    load = fc4.number_input("Machine load (%)", 0.0, 100.0, 70.0)
    ambient = fc4.number_input("Ambient temp (°C)", 0.0, 50.0, 25.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    row = {c: 0.0 for c in feat_cols}
    base = {
        "vibration_rms": vibration, "temperature_motor": temp, "current": current,
        "pressure_level": pressure, "rpm": rpm, "hours_since_maintenance": hours,
        "ambient_temp": ambient, "machine_load": load,
        "vibration_energy": vibration ** 2, "temperature_gradient": temp - ambient,
        "utilization_rate": load / 100.0, "operating_hours": hours,
    }
    for k, v in base.items():
        if k in row:
            row[k] = v
    for col in ["vibration_rms", "temperature_motor", "current", "pressure_level", "rpm"]:
        if f"{col}_roll_mean" in row:
            row[f"{col}_roll_mean"] = base.get(col, 0.0)
    X = pd.DataFrame([row])[feat_cols]
    rul_pred = float(rul_model.predict(X)[0])
    fail_proba = float(fail_model.predict_proba(X)[0, 1])
    health = float(np.clip(100 * (0.5 * np.clip(rul_pred / 500, 0, 1) + 0.5 * (1 - fail_proba)), 0, 100))

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Health Score", f"{health:.1f}")
    r2.metric("Failure Probability", f"{fail_proba:.1%}")
    r3.metric("Predicted RUL", f"{rul_pred:.0f} hrs")
    r4.metric("Recommendation", recommend(health, fail_proba))

st.divider()
st.caption("Model metrics — RUL: MAE {:.1f}h, R² {:.2f} · Failure: ROC-AUC {:.3f}, Recall {:.1%}".format(
    metrics["rul"]["MAE"], metrics["rul"]["R2"], metrics["failure"]["ROC_AUC"], metrics["failure"]["Recall"]
))
