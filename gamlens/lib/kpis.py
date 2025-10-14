import numpy as np
import pandas as pd
from .schema import ROAS_COLS

def add_core_kpis(df: pd.DataFrame):
    """Add CPI/ARPU/ROAS/ROI day and build the summary table."""
    out = df.copy()
    
    # Calculate total revenue (in-app purchases + ad revenue)
    out["total_revenue"] = out["revenue"].fillna(0) + out["ad_revenue"].fillna(0)
    
    # Normalize ROAS columns: if values are > 1, assume they're percentages and divide by 100
    # Additionally: treat zeros as missing if later-day columns are systematically zero (likely days not reached)
    for col in ROAS_COLS:
        if col in out.columns:
            # Convert to numeric, coercing errors to NaN
            out[col] = pd.to_numeric(out[col], errors='coerce')
            # If max value is > 1, assume it's in percentage format (50 instead of 0.50)
            max_val = out[col].max()
            if pd.notnull(max_val) and max_val > 1:
                out[col] = out[col] / 100

    # Heuristic: if most roas_d* are exactly 0 for later days, interpret as "not yet occurred"
    # Replace exact zeros with NaN for days beyond the first non-zero observation to avoid false plateaus
    try:
        day_cols_present = [c for c in ROAS_COLS if c in out.columns]
        if day_cols_present:
            # Work row-wise
            def _mask_future_zeros(row):
                seen_nonzero = False
                for c in day_cols_present:
                    v = row.get(c)
                    if not seen_nonzero and pd.notnull(v) and v > 0:
                        seen_nonzero = True
                    elif seen_nonzero and pd.notnull(v) and v == 0:
                        row[c] = pd.NA
                return row
            out[day_cols_present] = out.apply(lambda r: _mask_future_zeros(r)[day_cols_present], axis=1)
    except Exception:
        # Best-effort; if anything goes wrong, keep original values
        pass

    out["CPI ($)"]  = (out["cost"] / out["installs"].replace(0, np.nan)).round(2)
    out["ARPU ($)"] = (out["total_revenue"] / out["installs"].replace(0, np.nan)).round(2)
    out["ROAS"]     = (out["total_revenue"] / out["cost"].replace(0, np.nan)).round(2)

    def roi_day(row):
        for c in ROAS_COLS:
            v = row.get(c)
            if pd.notnull(v) and v >= 1:
                return c.replace("roas_d", "D")
        return None

    out["ROI 100% By (Day)"] = out.apply(roi_day, axis=1)
    out["Retention D7 (%)"]  = (out["retention_rate_d7"] * 100).round(2)
    
    # Convert ROAS columns to percentages for display
    out["ROAS D7 (%)"] = (out["roas_d7"] * 100).round(2)
    out["ROAS D14 (%)"] = (out["roas_d14"] * 100).round(2)

    # Friendly campaign label
    out["Campaign"] = out[["game","channel","platform","country","date"]].astype(str).agg(" | ".join, axis=1)

    table = out[[
        "Campaign","CPI ($)","installs","ROAS D7 (%)","ROAS D14 (%)","ROI 100% By (Day)",
        "Retention D7 (%)","ARPU ($)","ROAS"
    ]].rename(columns={
        "installs":"Installs"
    })

    return out, table