"""
Operations Dashboard — Predictive Maintenance Platform
ML-004: Enterprise Predictive Maintenance Platform

Run: streamlit run src/dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib, json, os

st.set_page_config(
    page_title="Predictive Maintenance | Operations Dashboard",
    layout="wide",
    page_icon="🏭",
    initial_sidebar_state="expanded",
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# THEME
# ---------------------------------------------------------------------------
NAVY = "#1A2332"
NAVY_LIGHT = "#243247"
SLATE = "#3D4F5C"
ORANGE = "#FF6B35"
GREEN = "#2E9E5B"
AMBER = "#E8A33D"
RED = "#D64545"
MUTED = "#8A94A0"
CARD_BG = "#212C3D"
BORDER = "#2E3B4E"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

    .stApp {{ background-color: {NAVY}; }}

    section[data-testid="stSidebar"] {{
        background-color: {NAVY_LIGHT};
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] * {{ color: #D8DCE0 !important; }}

    #MainMenu, footer, header {{ visibility: hidden; }}

    h1, h2, h3, h4, p, span, label, div {{ color: #E8EAED; }}

    .hero {{
        background: linear-gradient(135deg, {NAVY_LIGHT} 0%, {NAVY} 100%);
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
    }}
    .hero-kicker {{
        color: {ORANGE}; font-size: 12px; font-weight: 700; letter-spacing: 2px;
        text-transform: uppercase; margin-bottom: 6px;
    }}
    .hero-title {{ color: #FFFFFF; font-size: 30px; font-weight: 800; margin: 0; }}
    .hero-sub {{ color: {MUTED}; font-size: 14px; margin-top: 6px; }}
    .live-badge {{
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(46,158,91,0.15); color: {GREEN};
        padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;
        border: 1px solid rgba(46,158,91,0.35);
    }}
    .live-dot {{
        width: 7px; height: 7px; border-radius: 50%; background: {GREEN};
        display: inline-block; box-shadow: 0 0 6px {GREEN};
    }}

    .kpi-card {{
        background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 14px;
        padding: 20px 22px; height: 118px;
    }}
    .kpi-label {{ color: {MUTED}; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }}
    .kpi-value {{ color: #FFFFFF; font-size: 32px; font-weight: 800; margin-top: 6px; }}
    .kpi-delta {{ font-size: 12px; margin-top: 4px; font-weight: 600; }}

    .section-title {{
        color: #FFFFFF; font-size: 17px; font-weight: 700; margin: 6px 0 14px 0;
        display: flex; align-items: center; gap: 8px;
    }}
    .section-title .bar {{
        width: 4px; height: 18px; background: {ORANGE}; border-radius: 2px; display: inline-block;
    }}

    .panel {{
        background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 14px;
        padding: 18px 20px; margin-bottom: 16px;
    }}

    div[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}

    .stSelectbox label, .stNumberInput label {{ color: #D8DCE0 !important; font-weight: 500 !important; }}
    div[data-baseweb="select"] > div {{ background-color: {CARD_BG}; border-color: {BORDER}; }}
    .stNumberInput input {{ background-color: {CARD_BG}; color: #E8EAED; border-color: {BORDER}; }}

    button[data-testid*="FormSubmit"] {{
        background: {ORANGE} !important; color: white !important; border: none !important; border-radius: 8px !important;
        font-weight: 700 !important; padding: 10px 24px !important;
    }}
    button[data-testid*="FormSubmit"]:hover {{ background: #E85A2A !important; color: white !important; }}
    button[data-testid*="FormSubmit"] p {{ color: white !important; font-weight: 700 !important; }}

    hr {{ border-color: {BORDER}; }}
</style>
""", unsafe_allow_html=True)


def rec_tier(fail_proba, health_score):
    if fail_proba >= 0.8 or health_score < 20:
        return "Emergency Shutdown", RED
    if fail_proba >= 0.5 or health_score < 40:
        return "Immediate Repair", ORANGE
    if fail_proba >= 0.25 or health_score < 60:
        return "Scheduled Maintenance", AMBER
    if health_score < 80:
        return "Continue Monitoring", SLATE
    return "Normal Operation", GREEN


def health_color(v):
    if v < 30:
        return RED
    if v < 60:
        return AMBER
    return GREEN


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


df, raw = load_data()
rul_model, fail_model, feat_cols, metrics = load_models()

machine_summary = df.groupby(["machine_id", "machine_type"]).agg(
    avg_health=("health_score", "mean"),
    min_health=("health_score", "min"),
    avg_fail_proba=("pred_fail_proba", "mean"),
    max_fail_proba=("pred_fail_proba", "max"),
    avg_rul=("pred_rul", "mean"),
).reset_index().sort_values("avg_health")

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏭 Predictive Maintenance")
    st.caption("ML-004 · Operations Console")
    st.markdown("---")
    st.markdown("**Filter fleet**")
    type_filter = st.multiselect(
        "Machine type", sorted(df["machine_type"].unique()),
        default=list(sorted(df["machine_type"].unique()))
    )
    risk_filter = st.select_slider("Min risk to show", options=["All", "Elevated", "High only"], value="All")
    st.markdown("---")
    st.markdown("**Model performance**")
    st.caption(f"Failure model ROC-AUC: **{metrics['failure']['ROC_AUC']:.3f}**")
    st.caption(f"Failure recall: **{metrics['failure']['Recall']:.1%}**")
    st.caption(f"RUL MAE: **{metrics['rul']['MAE']:.1f} hrs**")
    st.markdown("---")
    st.caption("Built for the Ezitech ML-004 internship case study.")

filtered = machine_summary[machine_summary["machine_type"].isin(type_filter)]
if risk_filter == "Elevated":
    filtered = filtered[filtered["max_fail_proba"] > 0.25]
elif risk_filter == "High only":
    filtered = filtered[filtered["max_fail_proba"] > 0.5]

# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
    <div class="hero-kicker">Enterprise Predictive Maintenance Platform</div>
    <div style="display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap; gap:12px;">
        <div>
            <div class="hero-title">Fleet Operations Dashboard</div>
            <div class="hero-sub">Real-time equipment health, failure risk, and maintenance recommendations across the monitored fleet</div>
        </div>
        <div class="live-badge"><span class="live-dot"></span> Live model · {df['machine_id'].nunique()} machines</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# KPI ROW
# ---------------------------------------------------------------------------
avg_health = df["health_score"].mean()
high_risk = int((machine_summary["max_fail_proba"] > 0.5).sum())
avg_rul = df["pred_rul"].mean()
total_machines = df["machine_id"].nunique()

k1, k2, k3, k4 = st.columns(4)
kpis = [
    (k1, "Machines Monitored", f"{total_machines}", MUTED, "across 4 machine types"),
    (k2, "Avg Fleet Health", f"{avg_health:.0f}/100", health_color(avg_health), "weighted across readings"),
    (k3, "High-Risk Machines", f"{high_risk}", RED if high_risk else GREEN, "failure prob > 50%"),
    (k4, "Avg Predicted RUL", f"{avg_rul:.0f} hrs", MUTED, "remaining useful life"),
]
for col, label, value, color, sub in kpis:
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-delta" style="color:{color};">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# FLEET GAUGE + HEALTH BY TYPE
# ---------------------------------------------------------------------------
c1, c2 = st.columns([1, 1.6])

with c1:
    st.markdown('<div class="section-title"><span class="bar"></span>Fleet Health</div>', unsafe_allow_html=True)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_health,
        number={"suffix": " / 100", "font": {"color": "#FFFFFF", "size": 34}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"color": MUTED, "size": 10}},
            "bar": {"color": health_color(avg_health)},
            "bgcolor": CARD_BG,
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(214,69,69,0.25)"},
                {"range": [30, 60], "color": "rgba(232,163,61,0.25)"},
                {"range": [60, 100], "color": "rgba(46,158,91,0.25)"},
            ],
        },
    ))
    fig_gauge.update_layout(
        height=260, margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#E8EAED"},
    )
    st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

with c2:
    st.markdown('<div class="section-title"><span class="bar"></span>Health by Machine Type</div>', unsafe_allow_html=True)
    type_health = machine_summary.groupby("machine_type")["avg_health"].mean().reset_index().sort_values("avg_health")
    fig_bar = px.bar(
        type_health, x="avg_health", y="machine_type", orientation="h",
        text=type_health["avg_health"].round(0).astype(int).astype(str),
    )
    fig_bar.update_traces(
        marker_color=[health_color(v) for v in type_health["avg_health"]],
        textposition="outside", textfont_color="#E8EAED", marker_line_width=0,
    )
    fig_bar.update_layout(
        height=260, margin=dict(l=10, r=30, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 105], showgrid=False, color=MUTED, title=None),
        yaxis=dict(showgrid=False, color="#E8EAED", title=None),
        font={"color": "#E8EAED"},
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# FLEET RISK TABLE
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title"><span class="bar"></span>Fleet Risk Overview</div>', unsafe_allow_html=True)

display_df = filtered.copy()
display_df["Recommendation"] = display_df.apply(lambda r: rec_tier(r["max_fail_proba"], r["avg_health"])[0], axis=1)
display_df = display_df.rename(columns={
    "machine_id": "Machine", "machine_type": "Type", "avg_health": "Avg Health",
    "min_health": "Min Health", "avg_fail_proba": "Avg Fail Prob",
    "max_fail_proba": "Max Fail Prob", "avg_rul": "Avg RUL (hrs)",
})

st.dataframe(
    display_df.style
        .format({"Avg Health": "{:.1f}", "Min Health": "{:.1f}", "Avg Fail Prob": "{:.1%}",
                  "Max Fail Prob": "{:.1%}", "Avg RUL (hrs)": "{:.1f}"})
        .background_gradient(subset=["Avg Health"], cmap="RdYlGn", vmin=0, vmax=100)
        .background_gradient(subset=["Max Fail Prob"], cmap="Reds", vmin=0, vmax=1),
    use_container_width=True, hide_index=True, height=280,
)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MACHINE DRILL-DOWN
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title"><span class="bar"></span>Machine Drill-Down</div>', unsafe_allow_html=True)

dd1, dd2 = st.columns([1, 3])
with dd1:
    machine_id = st.selectbox("Select machine", sorted(df["machine_id"].unique()))

mdf = df[df["machine_id"] == machine_id].reset_index(drop=True)
cur_health = mdf["health_score"].iloc[-1]
cur_fail = mdf["pred_fail_proba"].iloc[-1]
cur_rul = mdf["pred_rul"].iloc[-1]
tier_label, tier_color = rec_tier(cur_fail, cur_health)

m1, m2, m3, m4 = st.columns(4)
for col, label, value, color in [
    (m1, "Health Score", f"{cur_health:.1f}", health_color(cur_health)),
    (m2, "Failure Probability", f"{cur_fail:.1%}", RED if cur_fail > 0.5 else (AMBER if cur_fail > 0.25 else GREEN)),
    (m3, "Predicted RUL", f"{cur_rul:.0f} hrs", MUTED),
    (m4, "Recommendation", tier_label, tier_color),
]:
    col.markdown(f"""
    <div class="kpi-card" style="height:96px;">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="font-size:22px; color:{color};">{value}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    y=mdf["health_score"].values, mode="lines", name="Health Score",
    line=dict(color=ORANGE, width=2.5), fill="tozeroy", fillcolor="rgba(255,107,53,0.08)",
))
fig_trend.update_layout(
    title=dict(text=f"Machine {machine_id} — Health Score Trend", font=dict(color="#E8EAED", size=14)),
    yaxis=dict(range=[0, 100], gridcolor=BORDER, color=MUTED),
    xaxis=dict(gridcolor=BORDER, color=MUTED, title="Reading #"),
    height=280, margin=dict(l=10, r=10, t=40, b=10),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#E8EAED"},
)
st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# LIVE WHAT-IF PREDICTOR
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title"><span class="bar"></span>Live Prediction — Try a Sensor Reading</div>', unsafe_allow_html=True)
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
    submitted = st.form_submit_button("Run Prediction")

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
    label, color = rec_tier(fail_proba, health)

    st.markdown("<br>", unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)
    for col, lab, val, c in [
        (r1, "Health Score", f"{health:.1f}", health_color(health)),
        (r2, "Failure Probability", f"{fail_proba:.1%}", RED if fail_proba > 0.5 else (AMBER if fail_proba > 0.25 else GREEN)),
        (r3, "Predicted RUL", f"{rul_pred:.0f} hrs", MUTED),
        (r4, "Recommendation", label, color),
    ]:
        col.markdown(f"""
        <div class="kpi-card" style="height:96px;">
            <div class="kpi-label">{lab}</div>
            <div class="kpi-value" style="font-size:22px; color:{c};">{val}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption(
    f"Model metrics — RUL: MAE {metrics['rul']['MAE']:.1f}h, R² {metrics['rul']['R2']:.2f} · "
    f"Failure: ROC-AUC {metrics['failure']['ROC_AUC']:.3f}, Recall {metrics['failure']['Recall']:.1%} · "
    f"ML-004 Predictive Maintenance Platform"
)
