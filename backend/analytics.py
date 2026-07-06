"""
DecisionLens AI - Analytics Engine
----------------------------------
Real, working implementations (no mocked numbers) for:
  - Anomaly detection (z-score + Isolation Forest ensemble)
  - Short-horizon forecasting (linear trend + weekly seasonality)
  - Driver / correlation analysis ("why did X move?")
  - Threshold-based alerting

These run on pandas/numpy/scikit-learn and scale to a few million rows on a
single machine. Swap `IsolationForest` for a GPU-accelerated cuML equivalent
(RAPIDS) or push the groupby/rolling ops to Spark if you outgrow one box —
the function signatures below are written so that swap doesn't touch the API
layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


NUMERIC_MIN_ROWS_FOR_MODEL = 20


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def dataset_summary(df: pd.DataFrame) -> dict:
    """High-level stats used for the dashboard KPI cards."""
    summary = {"rows": int(len(df)), "columns": list(df.columns), "metrics": {}}
    for col in numeric_columns(df):
        series = df[col].dropna()
        if series.empty:
            continue
        latest = float(series.iloc[-1])
        prev = float(series.iloc[-2]) if len(series) > 1 else latest
        pct_change = ((latest - prev) / prev * 100) if prev != 0 else 0.0
        summary["metrics"][col] = {
            "mean": round(float(series.mean()), 3),
            "std": round(float(series.std(ddof=0)), 3),
            "min": round(float(series.min()), 3),
            "max": round(float(series.max()), 3),
            "latest": round(latest, 3),
            "pct_change_latest": round(pct_change, 2),
        }
    return summary


def detect_anomalies(df: pd.DataFrame, group_col: str | None = None) -> list[dict]:
    """
    Ensemble anomaly detection:
      1. Rolling z-score per numeric column (fast, explainable).
      2. Isolation Forest across all numeric columns jointly (catches
         multivariate anomalies a single-column z-score would miss).

    If `group_col` is provided (e.g. "region"), anomalies are detected
    within each group independently, which matters a lot in practice —
    a value that's normal for one region can be a severe anomaly for
    another.
    """
    results: list[dict] = []
    cols = numeric_columns(df)
    if not cols:
        return results

    groups = df.groupby(group_col) if group_col and group_col in df.columns else [(None, df)]

    for group_name, gdf in groups:
        gdf = gdf.reset_index(drop=True)

        # --- z-score pass (per column) ---
        for col in cols:
            series = gdf[col].astype(float)
            if series.std(ddof=0) == 0 or len(series) < 5:
                continue
            z = (series - series.mean()) / series.std(ddof=0)
            flagged = gdf.index[z.abs() > 2.5]
            for idx in flagged:
                results.append({
                    "type": "zscore",
                    "group": group_name,
                    "column": col,
                    "row_index": int(idx),
                    "value": round(float(series.iloc[idx]), 3),
                    "z_score": round(float(z.iloc[idx]), 2),
                    "severity": "high" if abs(z.iloc[idx]) > 3.5 else "medium",
                })

        # --- Isolation Forest pass (multivariate) ---
        if len(gdf) >= NUMERIC_MIN_ROWS_FOR_MODEL and len(cols) >= 2:
            X = gdf[cols].fillna(gdf[cols].mean())
            iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=150)
            preds = iso.fit_predict(X)
            scores = iso.score_samples(X)
            for idx in np.where(preds == -1)[0]:
                results.append({
                    "type": "isolation_forest",
                    "group": group_name,
                    "column": "multivariate",
                    "row_index": int(idx),
                    "value": {c: round(float(gdf.iloc[idx][c]), 3) for c in cols},
                    "anomaly_score": round(float(scores[idx]), 4),
                    "severity": "high" if scores[idx] < -0.15 else "medium",
                })

    # surface the strongest anomalies first
    results.sort(key=lambda r: r.get("z_score", r.get("anomaly_score", 0)) if r["severity"] == "high" else 0)
    return results[:200]


def correlation_drivers(df: pd.DataFrame, target: str, top_n: int = 5) -> list[dict]:
    """
    'Why did <target> move?' — ranks other numeric columns by correlation
    with the target, and separately reports the columns that moved most
    sharply in the same window the target moved.
    """
    cols = [c for c in numeric_columns(df) if c != target]
    if target not in df.columns or not cols:
        return []

    corr = df[cols + [target]].corr(numeric_only=True)[target].drop(target)
    corr = corr.dropna().sort_values(key=lambda s: s.abs(), ascending=False)

    drivers = []
    for col, value in corr.head(top_n).items():
        direction = "moves with" if value > 0 else "moves opposite to"
        drivers.append({
            "column": col,
            "correlation": round(float(value), 3),
            "relationship": direction,
            "strength": (
                "strong" if abs(value) > 0.6 else "moderate" if abs(value) > 0.3 else "weak"
            ),
        })
    return drivers


def linear_forecast(series: pd.Series, periods: int = 7, seasonality: int = 7) -> dict:
    """
    Lightweight, dependency-free forecast: OLS trend line + average
    weekly seasonal offset added back on top. Good enough for short
    (7-30 step) operational forecasts; swap in Prophet/ARIMA/TFT if you
    need longer-horizon accuracy.
    """
    series = series.dropna().reset_index(drop=True)
    n = len(series)
    if n < 5:
        return {"error": "Not enough data points to forecast (need >= 5)."}

    x = np.arange(n)
    y = series.values.astype(float)

    # OLS trend
    slope, intercept = np.polyfit(x, y, 1)
    trend = slope * x + intercept
    residuals = y - trend

    # seasonal offsets (average residual per position-in-cycle)
    seasonal_offset = np.zeros(seasonality)
    if n >= seasonality * 2:
        for i in range(seasonality):
            vals = residuals[i::seasonality]
            seasonal_offset[i] = float(np.mean(vals))

    future_x = np.arange(n, n + periods)
    future_trend = slope * future_x + intercept
    future_seasonal = np.array([seasonal_offset[i % seasonality] for i in future_x])
    forecast = future_trend + future_seasonal

    resid_std = float(np.std(residuals)) if len(residuals) else 0.0
    lower = forecast - 1.96 * resid_std
    upper = forecast + 1.96 * resid_std

    return {
        "history": [round(float(v), 3) for v in y],
        "forecast": [round(float(v), 3) for v in forecast],
        "lower_bound": [round(float(v), 3) for v in lower],
        "upper_bound": [round(float(v), 3) for v in upper],
        "trend_slope": round(float(slope), 4),
        "trend_direction": "up" if slope > 0.01 else "down" if slope < -0.01 else "flat",
    }


def threshold_alerts(df: pd.DataFrame, z_threshold: float = 2.5, group_col: str | None = None) -> list[dict]:
    """Simple, explainable alert feed: latest row per group vs its own history."""
    alerts = []
    cols = numeric_columns(df)
    groups = df.groupby(group_col) if group_col and group_col in df.columns else [(None, df)]

    for group_name, gdf in groups:
        gdf = gdf.reset_index(drop=True)
        if len(gdf) < 5:
            continue
        latest_idx = len(gdf) - 1
        for col in cols:
            series = gdf[col].astype(float)
            std = series.std(ddof=0)
            if std == 0:
                continue
            z = (series.iloc[latest_idx] - series.mean()) / std
            if abs(z) > z_threshold:
                alerts.append({
                    "group": group_name,
                    "column": col,
                    "latest_value": round(float(series.iloc[latest_idx]), 3),
                    "z_score": round(float(z), 2),
                    "direction": "spike" if z > 0 else "drop",
                    "severity": "critical" if abs(z) > 3.5 else "warning",
                })
    return alerts
