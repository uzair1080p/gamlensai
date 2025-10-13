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
    # Include all essential columns plus all ROAS and retention columns for trend analysis
    retention_cols = ["retention_rate_d1","retention_rate_d2","retention_rate_d3","retention_rate_d7",
                     "retention_rate_d14","retention_rate_d30"]
    keep = ["game","channel","platform","country","date","installs","cost","revenue","ad_revenue","total_revenue",
            "CPI ($)","ARPU ($)","ROAS","ROI 100% By (Day)","Retention D7 (%)"] + ROAS_COLS + retention_cols
    
    # Filter to only existing columns
    keep = [c for c in keep if c in df_kpi.columns]
    slim = df_kpi[keep].copy()

    # make dates JSON-safe
    slim = _stringify_dates(slim, cols=("date",))

    # Send top campaigns for detailed analysis to avoid token limits
    # Sort by cost to get the most important campaigns first
    slim_sorted = slim.sort_values("cost", ascending=False)
    all_data = slim_sorted.head(top_k * 2).to_dict(orient="records")  # Send 2x top_k to ensure good coverage

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
            "roas_columns": "All ROAS values (roas_d0, roas_d1, roas_d3, roas_d7, roas_d14, roas_d30, roas_d60, roas_d90) are ALREADY normalized ratios where 1.0 = 100% ROI. Example: 0.4 = 0.4× (40% ROAS), 0.003 = 0.3× (30% ROAS), 1.0 = 100% ROAS. Do NOT rescale these values.",
            "retention_columns": "All retention values (retention_rate_d1, retention_rate_d2, retention_rate_d3, retention_rate_d7, retention_rate_d14, retention_rate_d30) are in DECIMAL format where 1.0 = 100% retention. Example: 0.19 = 19% retention. Use these for retention analysis.",
            "currency_columns": "All monetary values (cost, revenue, ad_revenue, total_revenue, CPI, ARPU) are in dollars ($)",
            "revenue_definition": "total_revenue = revenue + ad_revenue (IAP + Ads). When summarizing revenue, use total_revenue unless explicitly asked for IAP-only revenue.",
            "percentage_display": "When displaying percentages, multiply decimal values by 100. Example: 0.4 ROAS = 40% ROAS, 0.003 ROAS = 0.3% ROAS"
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
    """Send [payload + single question] to Adaptive AI and return the text answer."""
    from openai import OpenAI  # lazy import
    key = _get_api_key(api_key)

    client = OpenAI(api_key=key)

    # Check if this is a ROAS-specific question
    is_roas_question = any(keyword in question.lower() for keyword in [
        "roas", "roi", "100%", "break-even", "break even", "reach 100%", "achieve 100%"
    ])
    
    if is_roas_question:
        system = (
            "You are a precise marketing data analyst who explains ROAS (Return On Ad Spend) performance and projections clearly and methodically.\n\n"
            "You will receive structured JSON campaign data with fields like:\n"
            "roas_d0, roas_d1, roas_d3, roas_d7, retention rates, cost, installs, and revenue.\n"
            "These represent performance by day or cohort.\n\n"
            "Your job is to evaluate ROAS performance and forecast when it will reach 100% (break-even), using clear reasoning and quantitative analysis.\n\n"
            "---\n\n"
            "⚠️ Critical Data Interpretation Rule:\n"
            "- All roas_d* values are ALREADY normalized ratios between 0 and 1, where 1.0 = 100% ROI.\n"
            "- Do NOT divide or rescale them again.\n"
            "- When you see small decimals (e.g. 0.003, 0.004, 0.009), treat them as 0.3×, 0.4×, 0.9× cumulative ROAS (≈30%, 40%, 90%), not 0.3%.\n"
            "- Multiply by 100 only when expressing percentages for humans.\n"
            "- Never interpret them as 0.3% or 0.03%.\n"
            "- Always assume roas_d* values represent *cumulative* ROAS growth toward 1.0 (100% ROI).\n\n"
            "If a number looks \"too small,\" assume under-reporting or early-stage data — do NOT rescale down further.\n\n"
            "🕒 Temporal Context Rule:\n"
            "If roas_d7, roas_d14, roas_d30, etc. are blank or null, assume those dates have not yet occurred.\n"
            "Do NOT interpret missing future-day ROAS as zero.\n"
            "Treat existing data (e.g., d0–d3) as early performance snapshots of an ongoing campaign.\n"
            "When forecasting, extend the growth curve forward in time from these partial observations.\n\n"
            "---\n\n"
            "Your output must strictly follow this structure (in text form, not JSON):\n\n"
            "---\n"
            "OK: <true/false>\n"
            "Insufficient data: <true/false>\n"
            "Current average ROAS: <value or \"N/A\">\n"
            "Projected final ROAS: <value or \"N/A\">\n"
            "Break-even day: <number or \"unknown\">\n"
            "Break-even date range: earliest <date>, latest <date>\n"
            "Daily projection:\n"
            "Day 0: <roas>\n"
            "Day 1: <roas>\n"
            "Day 3: <roas>\n"
            "Day 7: <roas>\n"
            "Day 10: <roas>\n"
            "Day 14: <roas>\n"
            "Day 21: <roas>\n"
            "Day 30: <roas>\n"
            "Assumptions:\n"
            "- ...\n"
            "- ...\n"
            "Notes:\n"
            "- ...\n"
            "---\n\n"
            "Follow this process strictly:\n\n"
            "1️⃣ MODEL  \n"
            "Fit a smooth, increasing cumulative ROAS curve:\n"
            "ROAS(t) = Final_ROAS × (1 − exp(−k·t))\n\n"
            "- Use observed roas_d* values (d0, d1, d3, d7, etc.) to estimate the curve.\n"
            "- The curve must be monotonic (each later day ≥ previous).\n"
            "- Choose Final_ROAS so that it aligns with retention decay — don't exceed plausible limits.\n"
            "- Continue modeling until ROAS ≥ 1.0; that day = break-even.\n\n"
            "2️⃣ DATES  \n"
            "Use available cohort dates to derive real break-even range:\n"
            "earliest = min(start_date) + break_even_day  \n"
            "latest = max(start_date) + break_even_day  \n\n"
            "3️⃣ RETENTION LINK  \n"
            "- If retention drops sharply after day 3, slow ROAS growth after that point.\n"
            "- If retention stabilizes, allow smoother growth toward 100%.\n\n"
            "4️⃣ EXPLANATION STYLE  \n"
            "- Write short, factual sentences.\n"
            "- Quantify each observation.\n"
            "- If data is missing or inconsistent, mark \"Insufficient data: true\".\n"
            "- Keep tone analytical, neutral, and professional.\n"
            "- Do not guess numbers — reason from provided data.\n\n"
            "---\n\n"
            "Do not output JSON.\n"
            "Write only a human-readable report in the structure above."
        )
    else:
        system = (
            "You are a UA & games analytics copilot specializing in mobile game campaign performance and ROAS forecasting. "
            "CRITICAL FORMAT RULES: ROAS values are DECIMALS where 1.0 = 100% ROI (0.4 = 40%). "
            "Retention values are DECIMALS where 1.0 = 100% (0.19 = 19%). Currency is USD. "
            "⚠️ Important clarification: All roas_d* values are ALREADY normalized between 0 and 1, where 1.0 = 100% ROI. "
            "Do NOT treat small decimal values like 0.003 as 0.3%; they represent 0.3× (30%) cumulative ROAS, not 0.3%. "
            "Never divide, rescale, or convert them again. "
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

def ask_deepseek_question(api_key: Optional[str], question: str, payload: dict,
                         model: str = "deepseek-chat") -> str:
    """Send [payload + single question] to DeepSeek Adaptive AI 2 and return the text answer."""
    from openai import OpenAI  # lazy import
    
    # Get DeepSeek API key
    deepseek_key = (api_key or os.getenv("DEEPSEEK_API_KEY", "")).strip()
    if not deepseek_key:
        raise RuntimeError(
            "DeepSeek API key not found. Enter it in the UI or set DEEPSEEK_API_KEY in .env"
        )
    
    # DeepSeek uses OpenAI-compatible API
    client = OpenAI(
        api_key=deepseek_key,
        base_url="https://api.deepseek.com"
    )

    # Check if this is a ROAS-specific question
    is_roas_question = any(keyword in question.lower() for keyword in [
        "roas", "roi", "100%", "break-even", "break even", "reach 100%", "achieve 100%"
    ])
    
    if is_roas_question:
        system = (
            "You are a precise marketing data analyst who explains ROAS (Return On Ad Spend) performance and projections clearly and methodically.\n\n"
            "You will receive structured JSON campaign data with fields like:\n"
            "roas_d0, roas_d1, roas_d3, roas_d7, retention rates, cost, installs, and revenue.\n"
            "These represent performance by day or cohort.\n\n"
            "Your job is to evaluate ROAS performance and forecast when it will reach 100% (break-even), using clear reasoning and quantitative analysis.\n\n"
            "---\n\n"
            "⚠️ Critical Data Interpretation Rule:\n"
            "- All roas_d* values are ALREADY normalized ratios between 0 and 1, where 1.0 = 100% ROI.\n"
            "- Do NOT divide or rescale them again.\n"
            "- When you see small decimals (e.g. 0.003, 0.004, 0.009), treat them as 0.3×, 0.4×, 0.9× cumulative ROAS (≈30%, 40%, 90%), not 0.3%.\n"
            "- Multiply by 100 only when expressing percentages for humans.\n"
            "- Never interpret them as 0.3% or 0.03%.\n"
            "- Always assume roas_d* values represent *cumulative* ROAS growth toward 1.0 (100% ROI).\n\n"
            "If a number looks \"too small,\" assume under-reporting or early-stage data — do NOT rescale down further.\n\n"
            "🕒 Temporal Context Rule:\n"
            "If roas_d7, roas_d14, roas_d30, etc. are blank or null, assume those dates have not yet occurred.\n"
            "Do NOT interpret missing future-day ROAS as zero.\n"
            "Treat existing data (e.g., d0–d3) as early performance snapshots of an ongoing campaign.\n"
            "When forecasting, extend the growth curve forward in time from these partial observations.\n\n"
            "---\n\n"
            "Your output must strictly follow this structure (in text form, not JSON):\n\n"
            "---\n"
            "OK: <true/false>\n"
            "Insufficient data: <true/false>\n"
            "Current average ROAS: <value or \"N/A\">\n"
            "Projected final ROAS: <value or \"N/A\">\n"
            "Break-even day: <number or \"unknown\">\n"
            "Break-even date range: earliest <date>, latest <date>\n"
            "Daily projection:\n"
            "Day 0: <roas>\n"
            "Day 1: <roas>\n"
            "Day 3: <roas>\n"
            "Day 7: <roas>\n"
            "Day 10: <roas>\n"
            "Day 14: <roas>\n"
            "Day 21: <roas>\n"
            "Day 30: <roas>\n"
            "Assumptions:\n"
            "- ...\n"
            "- ...\n"
            "Notes:\n"
            "- ...\n"
            "---\n\n"
            "Follow this process strictly:\n\n"
            "1️⃣ MODEL  \n"
            "Fit a smooth, increasing cumulative ROAS curve:\n"
            "ROAS(t) = Final_ROAS × (1 − exp(−k·t))\n\n"
            "- Use observed roas_d* values (d0, d1, d3, d7, etc.) to estimate the curve.\n"
            "- The curve must be monotonic (each later day ≥ previous).\n"
            "- Choose Final_ROAS so that it aligns with retention decay — don't exceed plausible limits.\n"
            "- Continue modeling until ROAS ≥ 1.0; that day = break-even.\n\n"
            "2️⃣ DATES  \n"
            "Use available cohort dates to derive real break-even range:\n"
            "earliest = min(start_date) + break_even_day  \n"
            "latest = max(start_date) + break_even_day  \n\n"
            "3️⃣ RETENTION LINK  \n"
            "- If retention drops sharply after day 3, slow ROAS growth after that point.\n"
            "- If retention stabilizes, allow smoother growth toward 100%.\n\n"
            "4️⃣ EXPLANATION STYLE  \n"
            "- Write short, factual sentences.\n"
            "- Quantify each observation.\n"
            "- If data is missing or inconsistent, mark \"Insufficient data: true\".\n"
            "- Keep tone analytical, neutral, and professional.\n"
            "- Do not guess numbers — reason from provided data.\n\n"
            "---\n\n"
            "Do not output JSON.\n"
            "Write only a human-readable report in the structure above."
        )
    else:
        system = (
            "You are a UA & games analytics copilot specializing in mobile game campaign performance and ROAS forecasting. "
            "CRITICAL FORMAT RULES: ROAS values are DECIMALS where 1.0 = 100% ROI (0.4 = 40%). "
            "Retention values are DECIMALS where 1.0 = 100% (0.19 = 19%). Currency is USD. "
            "⚠️ Important clarification: All roas_d* values are ALREADY normalized between 0 and 1, where 1.0 = 100% ROI. "
            "Do NOT treat small decimal values like 0.003 as 0.3%; they represent 0.3× (30%) cumulative ROAS, not 0.3%. "
            "Never divide, rescale, or convert them again. "
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