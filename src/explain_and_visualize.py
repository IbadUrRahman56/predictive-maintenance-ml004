"""
Explainable AI + Operations Dashboard
ML-004: Enterprise Predictive Maintenance Platform
"""
import pandas as pd, numpy as np, joblib, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

def load():
    rul_model = joblib.load("models/rul_model.joblib")
    fail_model = joblib.load("models/failure_model.joblib")
    feat_cols = json.load(open("models/feature_cols.json"))
    test = pd.read_parquet("data/test_predictions.parquet")
    return rul_model, fail_model, feat_cols, test

def shap_summary(model, X, title, fname, max_display=12):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    plt.figure()
    shap.summary_plot(sv, X, max_display=max_display, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

def failure_heatmap(test, fname):
    df = test.copy()
    pivot = df.pivot_table(index="machine_id", values="pred_fail_proba", aggfunc="mean")
    plt.figure(figsize=(4, 6))
    plt.imshow(pivot.values, cmap="Reds", aspect="auto", vmin=0, vmax=pivot.values.max())
    plt.yticks(range(len(pivot)), [f"M{i}" for i in pivot.index])
    plt.xticks([])
    plt.colorbar(label="Avg predicted failure probability")
    plt.title("Failure Risk Heatmap by Machine")
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

def rul_distribution(test, fname):
    plt.figure(figsize=(6, 4))
    plt.hist(test["pred_rul"], bins=40, color="#2b6cb0", alpha=0.8)
    plt.xlabel("Predicted Remaining Useful Life (hours)")
    plt.ylabel("Count of readings")
    plt.title("RUL Distribution Across Test Machines")
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

def health_score_by_type(test, raw_df, fname):
    merged = test.merge(raw_df[["machine_id", "machine_type"]].drop_duplicates(), on="machine_id", how="left")
    means = merged.groupby("machine_type")["health_score"].mean().sort_values()
    plt.figure(figsize=(6, 4))
    plt.barh(means.index, means.values, color="#38a169")
    plt.xlabel("Average Health Score (0-100)")
    plt.title("Factory Health Score by Machine Type")
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

def failure_type_breakdown(raw_df, fname):
    counts = raw_df[raw_df["failure_within_24h"] == 1]["failure_type"].value_counts()
    plt.figure(figsize=(6, 4))
    plt.bar(counts.index, counts.values, color="#c05621")
    plt.ylabel("Count")
    plt.title("Failure Type Breakdown (failure_within_24h=1)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

def roc_pr_curve(test, fname):
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    y = test["failure_within_24h"]
    p = test["pred_fail_proba"]
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(fpr, tpr, color="#2b6cb0")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set_title(f"ROC Curve (AUC={auc(fpr, tpr):.3f})")
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[1].plot(rec, prec, color="#c05621")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

def main():
    rul_model, fail_model, feat_cols, test = load()
    raw_df = pd.read_csv("data/predictive_maintenance.csv")
    Xtest = test[feat_cols]

    sample = Xtest.sample(min(1500, len(Xtest)), random_state=42)
    shap_summary(rul_model, sample, "SHAP Feature Importance - RUL Regression", "plots/shap_rul.png")
    shap_summary(fail_model, sample, "SHAP Feature Importance - Failure Classification", "plots/shap_failure.png")

    failure_heatmap(test, "plots/failure_heatmap.png")
    rul_distribution(test, "plots/rul_distribution.png")
    health_score_by_type(test, raw_df, "plots/health_by_type.png")
    failure_type_breakdown(raw_df, "plots/failure_type_breakdown.png")
    roc_pr_curve(test, "plots/roc_pr_curve.png")

    print("All plots saved to plots/")

if __name__ == "__main__":
    main()
