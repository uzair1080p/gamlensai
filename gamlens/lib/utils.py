import os
import numpy as np
import pandas as pd
from pandas import NA
from .schema import TEMPLATE_COLS, NUM_COLS
from io import BytesIO, StringIO

DATA_DIR = "data"

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)

TEXT_COLS = ["game", "channel", "platform", "country"]

def _is_nan_like(x):
    import math
    try:
        return x is None or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return False

def _clean_text_val(x):
    if _is_nan_like(x):
        return NA
    s = str(x).strip()
    if s == "" or s.lower() in {"0", "nan", "none", "null"}:
        return NA
    return s

def read_csv_strict(path_or_buffer) -> pd.DataFrame:
    """Read CSV robustly and enforce the strict data template schema."""
    # Grab raw bytes once if a file-like object was provided
    raw_bytes = None
    if hasattr(path_or_buffer, "read") and not isinstance(path_or_buffer, (str, bytes)):
        # Streamlit's UploadedFile returns bytes; ensure we reset after read
        try:
            pos = path_or_buffer.tell()
        except Exception:
            pos = None
        raw_bytes = path_or_buffer.read()
        try:
            if pos is not None:
                path_or_buffer.seek(pos)
        except Exception:
            pass
    elif isinstance(path_or_buffer, bytes):
        raw_bytes = path_or_buffer

    encodings_to_try = ["utf-8", "utf-16", "utf-16-le", "utf-16-be"]
    last_err = None
    df = None
    for enc in encodings_to_try:
        try:
            if raw_bytes is not None:
                text = raw_bytes.decode(enc, errors="replace")
                handle = StringIO(text)
                df = pd.read_csv(handle, sep=None, engine="python", dtype=str)
            else:
                # Path on disk
                df = pd.read_csv(path_or_buffer, sep=None, engine="python", encoding=enc, dtype=str)
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

    # Drop rows that are entirely empty (common in spreadsheet exports)
    df.dropna(how="all", inplace=True)

    # Fix common header typo(s)
    if "vel_25_events" in df.columns and "level_25_events" not in df.columns:
        df.rename(columns={"vel_25_events": "level_25_events"}, inplace=True)

    # Map back to canonical template casing
    template_map = {c.lower(): c for c in TEMPLATE_COLS}
    rename_map = {c: template_map.get(c, c) for c in df.columns}
    df.rename(columns=rename_map, inplace=True)

    # Validate required columns
    missing = [c for c in TEMPLATE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Keep only expected columns in correct order
    df = df[TEMPLATE_COLS].copy()

    # --- New text sanitation: make non-numeric columns pure nullable strings ---
    for dim in TEXT_COLS:
        if dim in df.columns:
            df[dim] = df[dim].map(_clean_text_val).astype("string")

    numeric_set = set(NUM_COLS) | {"date"}
    for col in df.columns:
        if col not in numeric_set:
            df[col] = df[col].map(_clean_text_val).astype("string")

    # Drop rows where all key fields are empty (typical trailing blanks)
    df.dropna(subset=["game", "channel", "platform", "country", "date"], how="all", inplace=True)

    # Parse date and numerics
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows that have no date and no numeric signal (fully blank after typing)
    numeric_subset = [c for c in NUM_COLS if c in df.columns]
    df = df[~(df["date"].isna() & df[numeric_subset].isna().all(axis=1))]

    return df

def coalesce_zero(series: pd.Series):
    return series.replace([0, np.inf, -np.inf], np.nan)