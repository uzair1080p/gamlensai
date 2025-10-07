import os
import numpy as np
import pandas as pd
from .schema import TEMPLATE_COLS, NUM_COLS

DATA_DIR = "data"

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)

def read_csv_strict(path_or_buffer) -> pd.DataFrame:
    """Read CSV/TSV robustly and enforce the strict data template schema.

    - Auto-detect delimiter (comma or tab)
    - Handle UTF-8 / UTF-16 encodings
    - Trim/normalize headers; fix common typos
    - Enforce column order and numeric types
    """
    # Try encodings
    encodings_to_try = ["utf-8", "utf-16", "utf-16-le", "utf-16-be"]
    last_err = None
    for enc in encodings_to_try:
        try:
            # Auto-detect delimiter with python engine
            df = pd.read_csv(path_or_buffer, encoding=enc, sep=None, engine="python")
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        raise ValueError(f"Failed to read file with common encodings: {last_err}")

    # Normalize headers
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # Fix common header typo(s)
    if "vel_25_events" in df.columns and "level_25_events" not in df.columns:
        df.rename(columns={"vel_25_events": "level_25_events"}, inplace=True)

    # Re-title-case expected headers to match TEMPLATE_COLS exactly
    # Build a map from lowercase->template
    template_map = {c.lower(): c for c in TEMPLATE_COLS}
    rename_map = {c: template_map.get(c, c) for c in df.columns}
    df.rename(columns=rename_map, inplace=True)

    # Validate required columns
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