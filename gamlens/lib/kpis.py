import numpy as np
import pandas as pd
from .schema import ROAS_COLS

def add_core_kpis(df: pd.DataFrame):
    """Add CPI/ARPU/ROAS/ROI day and build the summary table."""
    out = df.copy()
    
    # Calculate total revenue (in-app purchases + ad revenue)
    out["total_revenue"] = out["revenue"].fillna(0) + out["ad_revenue"].fillna(0)

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