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
        "CRITICAL FORMAT RULES: ROAS values are DECIMALS where 1.0 = 100% ROI (0.4 = 40%). "
        "Retention values are DECIMALS where 1.0 = 100% (0.19 = 19%). Currency is USD. "
        "When answering, follow this exact methodology and output format: "
        "\n\nMETHOD (always use):\n"
        "1) Identify the scope (channel/country/game/date range if implied).\n"
        "2) Read ROAS progression in order: roas_d0 → roas_d1 → roas_d3 → roas_d7 → roas_d14 → roas_d30 → roas_d60 → roas_d90.\n"
        "3) Use decimal values for math; convert to % only for display.\n"
        "4) Compute growth between successive waypoints (e.g., D1→D3, D3→D7).\n"
        "5) Fit a simple trend (piecewise linear or log-saturation) using available points.\n"
        "6) Project the day D* when ROAS will reach 1.0 (100% ROI). If data plateaus below 1.0, state that explicitly.\n"
        "7) Validate projection against recent growth; cap unrealistic extrapolations and explain uncertainty.\n"
        "\nOUTPUT (use this exact structure):\n"
        "- Summary: one paragraph with the answer (day to reach 100% ROI or Not Achieved).\n"
        "- Projection: D* estimate with 95% range, and the trend type used.\n"
        "- Key Figures (table): columns [Day, ROAS (%), Δ vs prior (pp)].\n"
        "- Assumptions & Risks: bullet points (max 4)."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "DATA PAYLOAD (JSON):"},
        {"role": "user", "content": json.dumps(payload, default=str)},  # guard
        {"role": "user", "content": f"QUESTION: {question}"}
    ]
    r = client.chat.completions.create(model=model, messages=messages, temperature=0.1)
    return r.choices[0].message.content