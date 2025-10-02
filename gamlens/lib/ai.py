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
        "The data payload contains detailed campaign metrics including ROAS progression across multiple days "
        "(roas_d0, roas_d1, roas_d3, roas_d7, roas_d14, roas_d30, roas_d60, roas_d90). "
        "When asked about ROAS projections or when 100% ROI will be achieved: "
        "1) Analyze the ROAS trend across available days to identify growth patterns "
        "2) Calculate growth rates and project future ROAS values "
        "3) Estimate when ROAS will reach 1.0 (100% ROI) based on the trend "
        "4) Provide specific day estimates (e.g., D45, D60) with supporting calculations "
        "5) State assumptions and confidence levels clearly. "
        "Use only the provided data. Be analytical, numeric, and actionable."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "DATA PAYLOAD (JSON):"},
        {"role": "user", "content": json.dumps(payload, default=str)},  # guard
        {"role": "user", "content": f"QUESTION: {question}"}
    ]
    r = client.chat.completions.create(model=model, messages=messages, temperature=0.2)
    return r.choices[0].message.content