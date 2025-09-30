import os
import numpy as np
import pandas as pd
from .schema import TEMPLATE_COLS, NUM_COLS

DATA_DIR = "data"

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)

def read_csv_strict(path_or_buffer) -> pd.DataFrame:
    """Read CSV and enforce the data template CSV schema in use."""
    df = pd.read_csv(path_or_buffer)
    df.columns = df.columns.str.strip()

    missing = [c for c in TEMPLATE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Keep only expected columns in correct order
    df = df[TEMPLATE_COLS].copy()

    # Types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def coalesce_zero(series: pd.Series):
    return series.replace([0, np.inf, -np.inf], np.nan)