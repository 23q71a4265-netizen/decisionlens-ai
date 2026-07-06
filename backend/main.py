"""
DecisionLens AI - Backend API
------------------------------
Run locally:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000 in a browser (the frontend is served
statically from this same app, so there's nothing else to run).

Deploy: see ../README.md for Render / Railway / Docker instructions.
"""

from __future__ import annotations

import io
import os
from typing import Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from analytics import (
    dataset_summary,
    detect_anomalies,
    correlation_drivers,
    linear_forecast,
    threshold_alerts,
    numeric_columns,
)
from nlq import answer_question

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DATA_PATH = os.path.join(APP_DIR, "sample_data.csv")
FRONTEND_DIR = os.path.join(os.path.dirname(APP_DIR), "frontend")

app = FastAPI(title="DecisionLens AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your deployed frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory dataset store.
# This is intentionally simple for the MVP: one active dataset per server
# process, kept in RAM as a pandas DataFrame. For multi-user production use,
# replace this with a real store (Postgres for metadata + a columnar store
# like DuckDB/Parquet-on-S3 for the actual rows) keyed by user/session id -
# every function below already takes a DataFrame as its first argument, so
# the swap is confined to `get_active_df()` / `set_active_df()`.
# ---------------------------------------------------------------------------
class DataStore:
    def __init__(self) -> None:
        self.df: Optional[pd.DataFrame] = None
        self.group_col: Optional[str] = None
        self.dataset_name: Optional[str] = None


store = DataStore()


def get_df() -> pd.DataFrame:
    if store.df is None:
        raise HTTPException(status_code=404, detail="No dataset loaded yet. Upload a CSV or load the demo dataset.")
    return store.df


def guess_group_col(df: pd.DataFrame) -> Optional[str]:
    for candidate in ["region", "category", "segment", "store", "device_id", "location"]:
        if candidate in df.columns:
            return candidate
    # fallback: first low-cardinality non-numeric column
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= 20:
            return col
    return None


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str


class AlertThresholdRequest(BaseModel):
    z_threshold: float = 2.5


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "dataset_loaded": store.df is not None}


@app.post("/api/load-sample")
def load_sample():
    """Loads the bundled synthetic retail dataset so the product works with zero setup."""
    if not os.path.exists(SAMPLE_DATA_PATH):
        raise HTTPException(status_code=500, detail="Sample dataset missing on server.")
    df = pd.read_csv(SAMPLE_DATA_PATH)
    store.df = df
    store.dataset_name = "sample_retail_data.csv"
    store.group_col = guess_group_col(df)
    return {
        "dataset_name": store.dataset_name,
        "rows": len(df),
        "columns": list(df.columns),
        "group_col": store.group_col,
        "preview": df.head(10).to_dict(orient="records"),
    }


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".csv", ".tsv")):
        raise HTTPException(status_code=400, detail="Please upload a .csv or .tsv file.")
    raw = await file.read()
    sep = "\t" if file.filename.lower().endswith(".tsv") else ","
    try:
        df = pd.read_csv(io.BytesIO(raw), sep=sep)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file has no rows.")

    store.df = df
    store.dataset_name = file.filename
    store.group_col = guess_group_col(df)
    return {
        "dataset_name": store.dataset_name,
        "rows": len(df),
        "columns": list(df.columns),
        "group_col": store.group_col,
        "preview": df.head(10).to_dict(orient="records"),
    }


@app.get("/api/summary")
def summary():
    df = get_df()
    s = dataset_summary(df)
    s["dataset_name"] = store.dataset_name
    s["group_col"] = store.group_col
    return s


@app.get("/api/insights")
def insights():
    df = get_df()
    anomalies = detect_anomalies(df, group_col=store.group_col)
    return {"count": len(anomalies), "anomalies": anomalies, "group_col": store.group_col}


@app.get("/api/drivers")
def drivers(target: str = Query(..., description="Numeric column to explain")):
    df = get_df()
    if target not in numeric_columns(df):
        raise HTTPException(status_code=400, detail=f"'{target}' is not a numeric column in the active dataset.")
    return {"target": target, "drivers": correlation_drivers(df, target)}


@app.get("/api/forecast")
def forecast(
    column: str = Query(..., description="Numeric column to forecast"),
    periods: int = Query(7, ge=1, le=90),
    group: Optional[str] = Query(None, description="Optional group value to filter to, e.g. region=West"),
):
    df = get_df()
    if column not in numeric_columns(df):
        raise HTTPException(status_code=400, detail=f"'{column}' is not a numeric column in the active dataset.")
    if group and store.group_col:
        df = df[df[store.group_col] == group]
        if df.empty:
            raise HTTPException(status_code=400, detail=f"No rows found for group '{group}'.")
    result = linear_forecast(df[column], periods=periods)
    result["column"] = column
    result["group"] = group
    return result


@app.get("/api/alerts")
def alerts(z_threshold: float = Query(2.5, ge=0.5, le=6.0)):
    df = get_df()
    result = threshold_alerts(df, z_threshold=z_threshold, group_col=store.group_col)
    return {"count": len(result), "alerts": result, "z_threshold": z_threshold}


@app.post("/api/query")
def query(req: QueryRequest):
    df = get_df()
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    return answer_question(df, req.question, group_col=store.group_col)


@app.get("/api/groups")
def groups():
    """Distinct values for the detected grouping column (e.g. list of regions)."""
    df = get_df()
    if not store.group_col:
        return {"group_col": None, "values": []}
    return {"group_col": store.group_col, "values": sorted(df[store.group_col].dropna().unique().tolist(), key=str)}


# ---------------------------------------------------------------------------
# Serve the frontend (so `uvicorn main:app` alone gives you the whole app)
# ---------------------------------------------------------------------------
if os.path.isdir(FRONTEND_DIR):
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")

    @app.get("/")
    def root():
        index_path = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        raise HTTPException(status_code=404, detail="Frontend not found.")
