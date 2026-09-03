"""
Sanity tests for the ML-004 predictive maintenance pipeline.
Run with: pytest tests/
"""
import subprocess
import sys
import os
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


@pytest.fixture(scope="module")
def raw_df():
    return pd.read_csv(os.path.join(ROOT, "data", "predictive_maintenance.csv"))


def test_raw_data_shape(raw_df):
    assert len(raw_df) > 0
    expected_cols = {"machine_id", "machine_type", "rul_hours", "failure_within_24h"}
    assert expected_cols.issubset(set(raw_df.columns))


def test_no_negative_rul(raw_df):
    assert (raw_df["rul_hours"] >= 0).all()


def test_failure_label_is_binary(raw_df):
    assert set(raw_df["failure_within_24h"].unique()).issubset({0, 1})


def test_feature_engineering_runs(tmp_path):
    from feature_engineering import load_raw, impute_missing, add_rolling_features, \
        add_trend_features, add_engineered_features, add_failure_frequency

    os.chdir(ROOT)
    df = load_raw()
    df = impute_missing(df)
    assert df[["vibration_rms", "temperature_motor"]].isna().sum().sum() == 0

    df = add_rolling_features(df)
    assert "vibration_rms_roll_mean" in df.columns

    df = add_trend_features(df)
    assert "vibration_rms_trend" in df.columns

    df = add_engineered_features(df)
    for col in ["vibration_energy", "temperature_gradient", "utilization_rate"]:
        assert col in df.columns

    df = add_failure_frequency(df)
    assert (df["cum_failures"] >= 0).all()


def test_api_recommendation_thresholds():
    sys.path.insert(0, os.path.join(ROOT, "src"))
    os.chdir(ROOT)
    from api import recommend

    assert recommend(health_score=10, fail_proba=0.9) == "Emergency Shutdown"
    assert recommend(health_score=90, fail_proba=0.01) == "Normal Operation"
