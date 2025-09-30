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
    """Create a compact, JSON-safe payload to send with each question."""
    keep = ["game","channel","platform","country","date","installs","cost","revenue","ad_revenue",
            "CPI ($)","ARPU ($)","ROAS","ROI 100% By (Day)","Retention D7 (%)"] + ROAS_COLS
    slim = df_kpi[keep].copy()

    # make dates JSON-safe
    slim = _stringify_dates(slim, cols=("date",))

    agg = (slim.groupby(["channel","country"], dropna=False)
           .agg(installs=("installs","sum"),
                cost=("cost","sum"),
                revenue=("revenue","sum"),
                cpi=("CPI ($)","mean"),
                arpu=("ARPU ($)","mean"),
                roas=("ROAS","mean"),
                ret7=("Retention D7 (%)","mean"))
           .reset_index()
           .sort_values("cost", ascending=False))

    payload = {
        "schema_version": "v1",
        "dataset_name": dataset_name,
        "granularity": "campaign-day",
        "columns": keep,
        "rows_count": int(len(slim)),
        "aggregates_channel_country": agg.to_dict(orient="records"),
        "top_spend": slim.sort_values("cost", ascending=False).head(top_k).to_dict(orient="records"),
        "top_installs": slim.sort_values("installs",   ascending=False).head(top_k).to_dict(orient="records"),
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
        "You are a UA & games analytics copilot. "
        "Use only the provided data payload to answer the user's single question. "
        "Be concise, numeric, and state assumptions/uncertainty."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "DATA PAYLOAD (JSON):"},
        {"role": "user", "content": json.dumps(payload, default=str)},  # guard
        {"role": "user", "content": f"QUESTION: {question}"}
    ]
    r = client.chat.completions.create(model=model, messages=messages, temperature=0.2)
    return r.choices[0].message.content