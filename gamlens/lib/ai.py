# streamlit_app/lib/ai.py
import os
import json
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from .schema import ROAS_COLS

# Load .env once
load_dotenv()

def _stringify_dates(df: pd.DataFrame, cols=("date",)) -> pd.DataFrame:
    """Ensure any datetime/Timestamp columns become ISO date strings."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.strftime("%Y-%m-%d")
    return out

def build_payload_for_ai(df_kpi: pd.DataFrame, dataset_name: str, top_k: int = 15) -> dict:
    """Create a comprehensive payload with full ROAS time-series data for AI analysis."""
    # Include all essential columns plus all ROAS columns for trend analysis
    keep = ["game","channel","platform","country","date","installs","cost","revenue","ad_revenue","total_revenue",
            "CPI ($)","ARPU ($)","ROAS","ROI 100% By (Day)","Retention D7 (%)"] + ROAS_COLS
    
    # Filter to only existing columns
    keep = [c for c in keep if c in df_kpi.columns]
    slim = df_kpi[keep].copy()

    # make dates JSON-safe
    slim = _stringify_dates(slim, cols=("date",))

    # Send ALL campaign-day records for detailed analysis (not just top_k)
    # This allows AI to see ROAS progression across all days
    all_data = slim.to_dict(orient="records")

    # Also provide aggregated summary for context
    agg = (slim.groupby(["channel","country"], dropna=False)
           .agg(installs=("installs","sum"),
                cost=("cost","sum"),
                total_revenue=("total_revenue","sum") if "total_revenue" in slim.columns else ("revenue","sum"),
                cpi=("CPI ($)","mean"),
                arpu=("ARPU ($)","mean"),
                roas=("ROAS","mean"),
                ret7=("Retention D7 (%)","mean"))
           .reset_index()
           .sort_values("cost", ascending=False))

    payload = {
        "schema_version": "v2",
        "dataset_name": dataset_name,
        "granularity": "campaign-day",
        "data_format": {
            "roas_columns": "All ROAS values (roas_d0, roas_d1, roas_d3, roas_d7, roas_d14, roas_d30, roas_d60, roas_d90) are in DECIMAL format where 1.0 = 100% ROI. Example: 0.4 = 40% ROAS, 0.5 = 50% ROAS, 1.0 = 100% ROAS",
            "retention_columns": "All retention values (retention_rate_d1, retention_rate_d2, etc.) are in DECIMAL format where 1.0 = 100% retention. Example: 0.19 = 19% retention",
            "currency_columns": "All monetary values (cost, revenue, ad_revenue, total_revenue, CPI, ARPU) are in dollars ($)",
            "percentage_display": "When displaying percentages, multiply decimal values by 100. Example: 0.4 ROAS = 40% ROAS"
        },
        "columns": keep,
        "rows_count": int(len(slim)),
        "all_campaigns": all_data,  # Full data with all ROAS columns for trend analysis
        "aggregates_channel_country": agg.to_dict(orient="records"),
    }
    return payload

def _get_api_key(passed_key: Optional[str]) -> str:
    """Use provided key; else read from env OPENAI_API_KEY; else raise."""
    key = (passed_key or os.getenv("OPENAI_API_KEY", "")).strip()
    if not key:
        raise RuntimeError(
            "OpenAI API key not found. Enter it in the UI or set OPENAI_API_KEY in .env"
        )
    return key

def ask_one_question(api_key: Optional[str], question: str, payload: dict,
                     model: str = "gpt-4o-mini") -> str:
    """Send [payload + single question] to GPT and return the text answer."""
    from openai import OpenAI  # lazy import
    key = _get_api_key(api_key)

    client = OpenAI(api_key=key)

    system = (
        "You are a UA & games analytics copilot specializing in mobile game campaign performance and ROAS forecasting. "
        "CRITICAL: All ROAS values in the data are in DECIMAL format where 1.0 = 100% ROI. "
        "Example: roas_d7 = 0.5 means 50% ROAS, roas_d14 = 1.0 means 100% ROAS. "
        "When analyzing ROAS trends and making projections: "
        "1) Use the exact decimal values from the data (0.4, 0.5, etc.) for calculations "
        "2) When displaying results, convert to percentages (0.4 = 40%, 0.5 = 50%) "
        "3) For 100% ROI projections, calculate when ROAS will reach 1.0 (not 100) "
        "4) Analyze ROAS progression: roas_d0 → roas_d1 → roas_d3 → roas_d7 → roas_d14 → roas_d30 "
        "5) Calculate consistent growth rates using decimal values "
        "6) Project when ROAS will reach 1.0 based on the trend "
        "7) Provide specific day estimates with mathematical calculations "
        "8) Always state your assumptions and show your work. "
        "Be mathematically consistent and precise. Use only the provided data."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "DATA PAYLOAD (JSON):"},
        {"role": "user", "content": json.dumps(payload, default=str)},  # guard
        {"role": "user", "content": f"QUESTION: {question}"}
    ]
    r = client.chat.completions.create(model=model, messages=messages, temperature=0.1)
    return r.choices[0].message.content