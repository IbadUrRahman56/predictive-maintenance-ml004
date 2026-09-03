const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, ImageRun, AlignmentType, BorderStyle, PageBreak
} = require("docx");

const metrics = JSON.parse(fs.readFileSync("models/metrics.json"));

function img(path, width, height, maxW = 550) {
  const scale = Math.min(1, maxW / width);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 100, after: 200 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path), transformation: { width: width * scale, height: height * scale } })],
  });
}

function h1(text) { return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } }); }
function h2(text) { return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 100 } }); }
function p(text, opts = {}) { return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 120 } }); }
function bullet(text) { return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 60 } }); }

function simpleTable(headers, rows, widths) {
  const totalWidth = 9000;
  const colWidths = widths || headers.map(() => Math.floor(totalWidth / headers.length));
  const headerRow = new TableRow({
    children: headers.map((hText, i) => new TableCell({
      width: { size: colWidths[i], type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, color: "auto", fill: "2b6cb0" },
      children: [new Paragraph({ children: [new TextRun({ text: hText, bold: true, color: "FFFFFF" })] })],
    })),
  });
  const bodyRows = rows.map(r => new TableRow({
    children: r.map((cell, i) => new TableCell({
      width: { size: colWidths[i], type: WidthType.DXA },
      children: [new Paragraph({ text: String(cell) })],
    })),
  }));
  return new Table({ width: { size: totalWidth, type: WidthType.DXA }, columnWidths: colWidths, rows: [headerRow, ...bodyRows] });
}

const doc = new Document({
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 } } },
    children: [
      new Paragraph({ text: "Enterprise Predictive Maintenance &", heading: HeadingLevel.TITLE, alignment: AlignmentType.CENTER }),
      new Paragraph({ text: "Industrial Equipment Failure Intelligence Platform", heading: HeadingLevel.TITLE, alignment: AlignmentType.CENTER }),
      new Paragraph({ text: "Case Study ML-004 — Model Evaluation & Technical Report", alignment: AlignmentType.CENTER, spacing: { before: 100, after: 300 } }),
      p("Ezitech Engineering Framework (EEF) Internship Project", { italics: true }),
      p("Team Size: 2 ML Engineers  |  Duration: 4 Weeks", { italics: true }),
      new Paragraph({ text: "", spacing: { after: 400 } }),

      h1("1. Executive Summary"),
      p("This report documents the design and implementation of a predictive maintenance platform for a global manufacturing company operating 45 factories, 18,000 industrial machines, and 1,200 maintenance engineers. The platform ingests IoT sensor streams (vibration, temperature, pressure, RPM, current, voltage, humidity, load), engineers time-aware features, and produces three outputs per machine: a Health Score (0-100), a 24-hour failure probability, and an estimated Remaining Useful Life (RUL). Two gradient-boosted models were trained and evaluated: a failure classifier achieving 98.7% ROC-AUC with 94.6% recall on held-out machines, and an RUL regressor with a mean absolute error of approximately 17.8 hours. Explainability (SHAP), a maintenance recommendation engine, drift monitoring, and a REST prediction API were implemented to satisfy the case study's MLOps and Explainable AI requirements."),

      h1("2. Business Problem"),
      p("The company currently relies on fixed maintenance schedules, leading to unexpected failures, high maintenance costs, unplanned downtime, poor spare parts planning, and no real-time visibility into equipment health. The objective is an AI-powered system that continuously monitors equipment and recommends maintenance action before failure occurs."),

      h1("3. Dataset Overview"),
      p("The working dataset contains 24,042 sensor readings from 20 machines across 4 machine types (CNC, Pump, Compressor, Robotic Arm), collected over a two-week period at irregular intervals (~1,200 readings per machine)."),
      simpleTable(
        ["Attribute", "Value"],
        [
          ["Total readings", "24,042"],
          ["Machines", "20 (across 4 machine types)"],
          ["Sensor channels", "Vibration RMS, Motor Temperature, Current, Pressure, RPM, Ambient Temp, Humidity, Voltage, Machine Load"],
          ["Target 1 — RUL", "rul_hours (continuous, regression target)"],
          ["Target 2 — Failure", "failure_within_24h (binary, 14.8% positive rate)"],
          ["Failure types observed", "hydraulic, bearing, electrical, motor_overheat, none"],
          ["Missing data", "Vibration (4.2%), Temperature (3.5%), Pressure (3.8%), RPM (2.2%) — sensor dropout, imputed"],
        ],
        [3000, 6000]
      ),

      h1("4. Feature Engineering"),
      p("Following the case study's Feature Engineering module, the raw sensor stream was transformed into a 38-feature matrix per reading:"),
      bullet("Rolling statistics (mean, standard deviation) over a 6-reading window per machine for vibration, temperature, current, pressure and RPM"),
      bullet("Trend features: local linear slope of vibration and temperature to capture degradation direction"),
      bullet("Vibration energy (RMS²) — standard vibration-analysis severity measure"),
      bullet("Temperature gradient (motor temperature minus ambient temperature)"),
      bullet("Utilization rate (machine load normalized to 0-1) and operating hours since last service"),
      bullet("Cumulative prior failure count per machine (failure frequency)"),
      bullet("Maintenance interval bucket (recent / moderate / overdue / critical) derived from hours since maintenance"),
      bullet("One-hot encoded machine type and operating mode"),
      p("Missing sensor values (2-4% per channel, consistent with intermittent IoT dropout) were imputed per-machine using forward/backward fill, with a global median fallback."),

      h1("5. Equipment Health Engine — Model Results"),
      h2("5.1 Train/Test Split"),
      p("Machines — not individual rows — were split 75/25 into train and test sets (15 vs. 5 machines) using a group-aware split. This prevents data leakage that would occur if readings from the same machine appeared in both sets, and gives an honest estimate of performance on machines the model has never seen."),

      h2("5.2 Remaining Useful Life (RUL) Regression — XGBoost"),
      simpleTable(["Metric", "Value"], [
        ["MAE", `${metrics.rul.MAE.toFixed(2)} hours`],
        ["RMSE", `${metrics.rul.RMSE.toFixed(2)} hours`],
        ["R²", metrics.rul.R2.toFixed(3)],
      ], [4000, 4000]),
      p("On average, RUL predictions are within ~18 hours of true remaining life. The R² of 0.26 reflects the difficulty of the regression task on this dataset — sensor readings alone explain a modest share of RUL variance, since failure timing also depends on latent wear and maintenance-history factors not fully captured in the sensor stream. This is a realistic outcome for a first-pass model and a natural target for iteration (see Section 9)."),
      img("plots/rul_distribution.png", 770, 510),
      img("plots/shap_rul.png", 1020, 809, 480),

      h2("5.3 Failure Classification (failure within 24h) — XGBoost"),
      simpleTable(["Metric", "Value"], [
        ["Accuracy", (metrics.failure.Accuracy * 100).toFixed(1) + "%"],
        ["Precision", (metrics.failure.Precision * 100).toFixed(1) + "%"],
        ["Recall", (metrics.failure.Recall * 100).toFixed(1) + "%"],
        ["F1 Score", metrics.failure.F1.toFixed(3)],
        ["ROC-AUC", metrics.failure.ROC_AUC.toFixed(3)],
      ], [4000, 4000]),
      p(`Confusion matrix (test set, 5 held-out machines): True Negatives ${metrics.failure.ConfusionMatrix[0][0]}, False Positives ${metrics.failure.ConfusionMatrix[0][1]}, False Negatives ${metrics.failure.ConfusionMatrix[1][0]}, True Positives ${metrics.failure.ConfusionMatrix[1][1]}.`),
      p("The model was tuned to favor recall over precision (via class-weighting) because in predictive maintenance the cost of a missed failure (safety incident, unplanned downtime) far outweighs the cost of an unnecessary inspection. 94.6% of true failures are caught, at the cost of some false alarms (precision 58.8%) — a defensible trade-off for a first deployment, refinable once the cost of false alarms is known precisely."),
      img("plots/roc_pr_curve.png", 1160, 510),
      img("plots/shap_failure.png", 1020, 809, 480),

      h2("5.4 Composite Health Score"),
      p("A 0-100 Health Score is derived per reading by blending normalized predicted RUL and inverse failure probability (50/50 weighting, tunable). This gives operators a single, intuitive number per machine for the Operations Dashboard."),
      img("plots/health_by_type.png", 770, 510),

      h1("6. Failure Classification by Type"),
      p("Among readings flagged as failing within 24 hours, four failure modes are represented in the historical logs:"),
      img("plots/failure_type_breakdown.png", 770, 510),

      h1("7. Explainable AI"),
      p("SHAP (SHapley Additive exPlanations) values were computed for both models to satisfy the case study's Explainable AI module. For the failure classifier, cumulative prior failures, rolling motor temperature, and hours since maintenance are the dominant risk drivers — consistent with mechanical intuition (heat and time-since-service are classic precursors to failure). The RUL regressor is driven primarily by hours since maintenance and vibration-related features. These SHAP summaries can be surfaced per-machine in the dashboard as a 'Top Risk Factors' explanation panel."),

      h1("8. Maintenance Recommendation Engine & Failure Risk Heatmap"),
      p("Each prediction is mapped to one of five actions using health score and failure probability thresholds: Emergency Shutdown, Immediate Repair, Scheduled Maintenance, Continue Monitoring, or Normal Operation. This logic is implemented in the prediction API (Section 10)."),
      img("plots/failure_heatmap.png", 517, 770, 350),

      h1("9. MLOps Implementation"),
      bullet("Experiment tracking: both models logged to MLflow (params, metrics, artifacts) under experiment 'predictive_maintenance_ml004', backed by a local SQLite store — viewable via `mlflow ui --backend-store-uri sqlite:///mlflow.db`"),
      bullet("Model registry: trained models persisted as versioned joblib artifacts alongside their feature schema (feature_cols.json) for reproducible inference"),
      bullet("Sensor drift detection: a lightweight drift check compares recent vs. overall feature means per sensor and flags shifts beyond a 15% threshold — all four core sensors were within tolerance on this dataset (see drift_report.json)"),
      bullet("Retraining: train_models.py is idempotent and can be scheduled (e.g., weekly) to retrain on newly ingested data"),
      p("For production scale (2M+ daily readings across 45 factories), the recommended path is to promote this pipeline into a scheduled Airflow/Prefect job writing to a feature store, with MLflow's Model Registry stages (Staging/Production) gating deployment."),

      h1("10. Predictive Maintenance API"),
      p("A FastAPI service (src/api.py) exposes a POST /predict endpoint accepting a single sensor reading and returning health score, failure probability, predicted RUL, and a maintenance recommendation. Interactive API docs are auto-generated at /docs. Example request/response:"),
      p("POST /predict  →  {\"health_score\": 6.8, \"failure_probability\": 0.9668, \"predicted_rul_hours\": 51.4, \"recommendation\": \"Emergency Shutdown\"}", { font: "Consolas", size: 20 }),

      h1("11. Architecture Overview"),
      bullet("Data Ingestion: CSV/streaming sensor data → pandas pipeline (Kafka streaming noted as a bonus extension)"),
      bullet("Feature Engineering: src/feature_engineering.py → Parquet feature store"),
      bullet("Equipment Health Engine: src/train_models.py (XGBoost RUL + Failure models)"),
      bullet("Explainability: src/explain_and_visualize.py (SHAP, dashboard plots)"),
      bullet("MLOps: src/mlops_tracking.py (MLflow tracking, drift checks)"),
      bullet("Serving: src/api.py (FastAPI REST endpoint)"),
      bullet("Backend stack per case study spec: Python, FastAPI, PostgreSQL, Redis (Postgres/Redis integration is a straightforward next step for persisting live readings and caching predictions)"),

      h1("12. Deliverables Mapping"),
      simpleTable(["Case Study Deliverable", "Status"], [
        ["Complete Source Code", "Delivered — src/ (feature engineering, training, explainability, MLOps, API)"],
        ["ML Pipeline", "Delivered — end-to-end, reproducible via 4 scripts"],
        ["Feature Engineering Pipeline", "Delivered — 38 engineered features"],
        ["Predictive Maintenance API", "Delivered — FastAPI /predict endpoint"],
        ["Operations Dashboard", "Delivered as static visuals; interactive dashboard is a natural next iteration"],
        ["Model Evaluation Report", "This document"],
        ["MLOps Pipeline", "Delivered — MLflow tracking + drift detection"],
        ["Architecture Diagram", "Section 11 (textual); diagram file included in deliverables"],
        ["Deployment Guide", "See README.md"],
        ["README", "Included"],
        ["Technical Presentation", "Recommend building slides from this report's sections 1, 5, 7-10"],
      ], [4500, 4500]),

      h1("13. Business Impact"),
      p("By predicting failures before they occur and quantifying remaining useful life, this platform enables the transition from fixed-interval to condition-based maintenance. Expected impact: reduced unplanned downtime, extended equipment lifespan, better spare-parts planning (repair cost is already logged per failure type and can feed inventory forecasting), and improved worker safety through earlier intervention on high-risk machines."),

      h1("14. Limitations & Next Steps"),
      bullet("RUL regression R² (0.26) leaves room for improvement — consider LSTM sequence models on raw sensor windows, or survival analysis for censored RUL, both listed as optional case-study extensions"),
      bullet("Precision on failure classification (58.8%) means roughly 2 in 5 alerts are false alarms; a cost-based threshold tuning pass (rather than the default 0.5 cutoff) would let engineers trade recall for precision explicitly"),
      bullet("Dataset covers 20 machines over 2 weeks — validating generalization across the full 18,000-machine, multi-factory fleet would need federated or multi-tenant retraining"),
      bullet("Interactive dashboard (React/Streamlit) and live Kafka streaming remain as the bonus-challenge extensions"),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("reports/ML-004_Model_Evaluation_Report.docx", buf);
  console.log("Report written.");
});
