"""
Feature Engineering Pipeline
ML-004: Enterprise Predictive Maintenance Platform

Generates rolling statistics, trend, energy, and utilization features
per machine from raw IoT sensor streams, as required by the case study's
Feature Engineering module.
"""
import pandas as pd
import numpy as np

RAW_PATH = "data/predictive_maintenance.csv"
OUT_PATH = "data/features.parquet"

SENSOR_COLS = [
    "vibration_rms", "temperature_motor", "current", "pressure_level",
    "rpm", "ambient_temp", "humidity", "voltage", "machine_load"
]

def load_raw():
    df = pd.read_csv(RAW_PATH, parse_dates=["timestamp"])
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    return df

def impute_missing(df):
    # Sensor dropouts are common in IoT streams; impute per-machine using
    # forward/backward fill then median as a last resort (no future leakage
    # concerns for a demo dataset with random dropout, not systematic gaps).
    for col in SENSOR_COLS:
        if df[col].isna().any():
            df[col] = df.groupby("machine_id")[col].transform(
                lambda s: s.ffill().bfill()
            )
            df[col] = df[col].fillna(df[col].median())
    return df

def add_rolling_features(df, window=6):
    df = df.copy()
    g = df.groupby("machine_id")
    for col in ["vibration_rms", "temperature_motor", "current", "pressure_level", "rpm"]:
        df[f"{col}_roll_mean"] = g[col].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        df[f"{col}_roll_std"] = g[col].transform(
            lambda s: s.rolling(window, min_periods=1).std().fillna(0)
        )
    return df

def add_trend_features(df, window=6):
    df = df.copy()
    def slope(s):
        if len(s) < 2:
            return 0.0
        x = np.arange(len(s))
        return np.polyfit(x, s.values, 1)[0]
    g = df.groupby("machine_id")
    for col in ["vibration_rms", "temperature_motor"]:
        df[f"{col}_trend"] = g[col].transform(
            lambda s: s.rolling(window, min_periods=2).apply(slope, raw=False).fillna(0)
        )
    return df

def add_engineered_features(df):
    df = df.copy()
    # Vibration energy proxy (RMS^2, standard vibration analysis measure)
    df["vibration_energy"] = df["vibration_rms"] ** 2
    # Temperature gradient vs. ambient
    df["temperature_gradient"] = df["temperature_motor"] - df["ambient_temp"]
    # Utilization rate proxy from machine_load
    df["utilization_rate"] = df["machine_load"] / 100.0
    # Operating hours since last maintenance already present as hours_since_maintenance
    df["operating_hours"] = df["hours_since_maintenance"]
    # Maintenance interval bucket
    df["maintenance_interval_bucket"] = pd.cut(
        df["hours_since_maintenance"],
        bins=[-1, 100, 300, 600, np.inf],
        labels=["recent", "moderate", "overdue", "critical"]
    )
    return df

def add_failure_frequency(df):
    # Historical failure frequency per machine up to (not including) current row
    df = df.copy()
    df["cum_failures"] = df.groupby("machine_id")["failure_within_24h"].transform(
        lambda s: s.shift(1).fillna(0).cumsum()
    )
    return df

def encode_categoricals(df):
    df = df.copy()
    df = pd.get_dummies(df, columns=["machine_type", "operating_mode", "maintenance_interval_bucket"],
                         prefix=["mtype", "mode", "maint"])
    return df

def build_features():
    df = load_raw()
    df = impute_missing(df)
    df = add_rolling_features(df)
    df = add_trend_features(df)
    df = add_engineered_features(df)
    df = add_failure_frequency(df)
    df_encoded = encode_categoricals(df)
    df_encoded.to_parquet(OUT_PATH, index=False)
    print(f"Feature matrix: {df_encoded.shape}")
    print(f"Saved to {OUT_PATH}")
    return df_encoded

if __name__ == "__main__":
    build_features()
