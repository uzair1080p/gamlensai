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
    """Read CSV robustly and enforce the strict data template schema.

    If the vectorized parser fails, a fallback line-by-line parser is used to
    detect and report the exact faulty line(s).
    """
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
            print(f"[CSV DEBUG] Read using encoding={enc}, shape={getattr(df, 'shape', None)}")
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        # Fallback: line-by-line diagnostics
        # Acquire text for detailed parsing
        if raw_bytes is None:
            with open(path_or_buffer, "rb") as f:
                raw_bytes = f.read()
        text = raw_bytes.decode("utf-8", errors="replace")
        return _read_csv_line_by_line_with_diagnostics(text)

    # Normalize headers
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    print(f"[CSV DEBUG] Normalized headers: {list(df.columns)}")

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
    try:
        samples = {c: df[c].head(3).tolist() for c in TEXT_COLS if c in df.columns}
        print(f"[CSV DEBUG] Raw text samples (first 3): {samples}")
    except Exception as e:
        print(f"[CSV DEBUG] Sampling text columns failed: {e}")

    # --- New text sanitation: make non-numeric columns pure nullable strings ---
    for dim in TEXT_COLS:
        if dim in df.columns:
            df[dim] = df[dim].map(_clean_text_val).astype("string")
    print(f"[CSV DEBUG] Text dtypes: {{c: str(df[c].dtype) for c in TEXT_COLS if c in df.columns}}")

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
    print(f"[CSV DEBUG] Numeric non-null counts: {{c: int(df[c].notna().sum()) for c in NUM_COLS if c in df.columns}}")

    # Drop rows that have no date and no numeric signal (fully blank after typing)
    numeric_subset = [c for c in NUM_COLS if c in df.columns]
    df = df[~(df["date"].isna() & df[numeric_subset].isna().all(axis=1))]
    print(f"[CSV DEBUG] Final shape: {df.shape}")

    return df

def coalesce_zero(series: pd.Series):
    return series.replace([0, np.inf, -np.inf], np.nan)

# -------------------- Fallback diagnostic parser --------------------
def _read_csv_line_by_line_with_diagnostics(text: str) -> pd.DataFrame:
    """Fallback CSV parser that scans line-by-line and reports faulty rows.

    Returns a cleaned DataFrame when possible; raises ValueError with a
    detailed report if structural errors are found.
    """
    import csv
    from datetime import datetime

    lines = text.splitlines()
    if not lines:
        raise ValueError("Empty file")

    # Detect delimiter
    header_raw = lines[0]
    delimiter = "\t" if "\t" in header_raw and header_raw.count("\t") >= header_raw.count(",") else ","

    reader = csv.reader(lines, delimiter=delimiter)
    header = next(reader)
    norm_header = [str(h).strip().lower().replace(" ", "_") for h in header]

    # Typo fix
    norm_header = ["level_25_events" if h == "vel_25_events" else h for h in norm_header]

    # Map to template names
    template_map = {c.lower(): c for c in TEMPLATE_COLS}
    mapped_header = [template_map.get(h, h) for h in norm_header]

    missing = [c for c in TEMPLATE_COLS if c not in mapped_header]
    if missing:
        raise ValueError(f"Header missing required columns: {missing}. Parsed header={mapped_header}")

    # Build index map for required columns
    name_to_idx = {name: mapped_header.index(name) for name in mapped_header}

    records = []
    errors = []
    for i, row in enumerate(reader, start=2):  # human 1-based; +1 for header
        # Pad/trim row length
        if len(row) != len(mapped_header):
            errors.append((i, "column_mismatch", f"expected {len(mapped_header)} cols, got {len(row)}", row[:10]))
            continue

        rec = {}
        for name in mapped_header:
            rec[name] = row[name_to_idx[name]]

        # Clean text cols
        for dim in TEXT_COLS:
            if dim in rec:
                rec[dim] = _clean_text_val(rec[dim])

        # Parse date
        try:
            rec["date"] = pd.to_datetime(rec.get("date"), errors="coerce")
        except Exception:
            rec["date"] = pd.NaT

        # Numerics
        for c in NUM_COLS:
            if c in rec:
                try:
                    rec[c] = pd.to_numeric(rec[c], errors="coerce")
                except Exception:
                    rec[c] = np.nan

        # If completely empty row, skip
        if pd.isna(rec.get("date")) and all(pd.isna(rec.get(c)) for c in NUM_COLS):
            # skip
            continue

        # Flag missing key dims
        if all(pd.isna(rec.get(k)) for k in ["game", "channel", "platform", "country"]):
            errors.append((i, "missing_dimensions", "all dims empty", {k: rec.get(k) for k in TEXT_COLS}))
            continue

        records.append(rec)

    if errors:
        # Report first few errors to help user fix quickly
        sample = "\n".join([f"line {ln}: {code} - {detail}" for ln, code, detail, *_ in errors[:5]])
        raise ValueError(f"CSV validation failed. First issues:\n{sample}\nTotal errors: {len(errors)}")

    if not records:
        raise ValueError("No valid data rows found after parsing.")

    df = pd.DataFrame.from_records(records)
    # Ensure all required columns exist
    for c in TEMPLATE_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[TEMPLATE_COLS]
    # Set dtypes comparable to main path
    for dim in TEXT_COLS:
        df[dim] = df[dim].astype("string")
    return df