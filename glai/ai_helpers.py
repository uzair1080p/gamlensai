"""
Helper functions for AI analysis and recommendations
"""

import os
import pandas as pd
from typing import Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ROAS columns for analysis
ROAS_COLS = ["roas_d0", "roas_d1", "roas_d3", "roas_d7", "roas_d14", "roas_d30", "roas_d60", "roas_d90"]

def build_payload_for_ai(df_kpi: pd.DataFrame, dataset_name: str, top_k: int = 15) -> dict:
    """Create a comprehensive payload with full ROAS time-series data for AI analysis."""
    # Include all essential columns plus all ROAS columns for trend analysis
    keep = ["game", "channel", "platform", "country", "date", "installs", "cost", "revenue", "ad_revenue", "total_revenue",
            "CPI ($)", "ARPU ($)", "ROAS", "ROI 100% By (Day)", "Retention D7 (%)"] + ROAS_COLS
    
    # Filter to only existing columns
    keep = [c for c in keep if c in df_kpi.columns]
    slim = df_kpi[keep].copy()

    # Make dates JSON-safe
    if 'date' in slim.columns:
        slim['date'] = slim['date'].astype(str)

    # Send ALL campaign-day records for detailed analysis (not just top_k)
    # This allows AI to see ROAS progression across all days
    all_data = slim.to_dict(orient="records")

    # Also provide aggregated summary for context
    agg_cols = {
        'installs': ('installs', 'sum'),
        'cost': ('cost', 'sum'),
        'cpi': ('CPI ($)', 'mean'),
        'arpu': ('ARPU ($)', 'mean'),
        'roas': ('ROAS', 'mean'),
    }
    
    if 'total_revenue' in slim.columns:
        agg_cols['total_revenue'] = ('total_revenue', 'sum')
    
    if 'Retention D7 (%)' in slim.columns:
        agg_cols['ret7'] = ('Retention D7 (%)', 'mean')
    
    agg = (slim.groupby(["channel", "country"], dropna=False)
           .agg(**agg_cols)
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


def ask_one_question(api_key: str, question: str, payload: dict) -> str:
    """Ask a single question to GPT about campaign data."""
    # Use provided API key or fall back to environment
    effective_key = api_key.strip() if api_key else os.getenv("OPENAI_API_KEY", "")
    
    if not effective_key:
        return "❌ No OpenAI API key provided. Please set OPENAI_API_KEY in .env or provide it in the interface."
    
    try:
        client = OpenAI(api_key=effective_key)
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # Build system prompt with explicit instructions
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
        
        # Build user prompt with data
        import json
        user_prompt = f"""Dataset: {payload['dataset_name']}
        
Data Format Guidelines:
{json.dumps(payload['data_format'], indent=2)}

Campaign Data (All Records):
{json.dumps(payload['all_campaigns'][:50], indent=2)}  # Show first 50 for context

Aggregated Summary by Channel/Country:
{json.dumps(payload['aggregates_channel_country'], indent=2)}

Total Records: {payload['rows_count']}

Question: {question}

Please provide a detailed, data-driven answer with specific numbers and actionable insights."""

        # Make API call
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for consistent responses
            max_tokens=1500
        )
        
        answer = response.choices[0].message.content
        return answer if answer else "❌ No response from GPT"
        
    except Exception as e:
        return f"❌ Error getting AI response: {str(e)}"


def add_core_kpis(df: pd.DataFrame):
    """Add CPI/ARPU/ROAS/ROI day and build the summary table."""
    import numpy as np
    
    out = df.copy()
    
    # Calculate total revenue (in-app purchases + ad revenue)
    out["total_revenue"] = out["revenue"].fillna(0) + out["ad_revenue"].fillna(0)
    
    # Normalize ROAS columns: if values are > 1, assume they're percentages and divide by 100
    for col in ROAS_COLS:
        if col in out.columns:
            # Convert to numeric, coercing errors to NaN
            out[col] = pd.to_numeric(out[col], errors='coerce')
            # If max value is > 1, assume it's in percentage format (50 instead of 0.50)
            max_val = out[col].max()
            if pd.notnull(max_val) and max_val > 1:
                out[col] = out[col] / 100

    out["CPI ($)"] = (out["cost"] / out["installs"].replace(0, np.nan)).round(2)
    out["ARPU ($)"] = (out["total_revenue"] / out["installs"].replace(0, np.nan)).round(2)
    out["ROAS"] = (out["total_revenue"] / out["cost"].replace(0, np.nan)).round(2)

    def roi_day(row):
        for c in ROAS_COLS:
            v = row.get(c)
            if pd.notnull(v) and v >= 1:
                return c.replace("roas_d", "D")
        return None

    out["ROI 100% By (Day)"] = out.apply(roi_day, axis=1)
    
    # Handle retention columns
    if "retention_rate_d7" in out.columns:
        out["Retention D7 (%)"] = (out["retention_rate_d7"] * 100).round(2)
    
    # Convert ROAS columns to percentages for display
    if "roas_d7" in out.columns:
        out["ROAS D7 (%)"] = (out["roas_d7"] * 100).round(2)
    if "roas_d14" in out.columns:
        out["ROAS D14 (%)"] = (out["roas_d14"] * 100).round(2)

    # Friendly campaign label
    label_cols = ["game", "channel", "platform", "country", "date"]
    existing_label_cols = [c for c in label_cols if c in out.columns]
    if existing_label_cols:
        out["Campaign"] = out[existing_label_cols].astype(str).agg(" | ".join, axis=1)

    # Build summary table
    table_cols = ["Campaign", "CPI ($)", "installs", "ROAS D7 (%)", "ROAS D14 (%)", 
                  "ROI 100% By (Day)", "Retention D7 (%)", "ARPU ($)", "ROAS"]
    available_table_cols = [c for c in table_cols if c in out.columns]
    
    table = out[available_table_cols].rename(columns={
        "installs": "Installs"
    })

    return out, table

