"""
GPT-powered campaign recommendation helper.

Given a predictions dataframe, this module asks an LLM (GPT-5 by default if
configured) to classify each campaign into one of: Scale, Maintain, Reduce, Cut
and provide a short rationale. The function returns a mapping keyed by
`row_index` suitable for merging onto the dashboard table.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Any, List, Tuple

import pandas as pd

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    OpenAI = None  # type: ignore

# Prefer reusing the existing FAQ GPT singleton client if available
try:
    from glai.faq_gpt import get_faq_gpt  # type: ignore
except Exception:
    get_faq_gpt = None  # type: ignore


def _summarize_dataframe(pred_df: pd.DataFrame) -> Dict[str, Any]:
    """Return light summary stats and schema info for conditioning the LLM."""
    cols = list(pred_df.columns)
    info = {
        "num_rows": int(len(pred_df)),
        "columns": cols,
        "has_roas_cols": any(c.startswith("predicted_roas_") for c in cols),
        "has_cost": "cost" in cols,
        "has_revenue": "revenue" in cols,
    }
    if "cost" in pred_df.columns:
        info["total_cost"] = float(pd.to_numeric(pred_df["cost"], errors="coerce").fillna(0).sum())
    if "predicted_roas_p50" in pred_df.columns:
        info["mean_roas_p50"] = float(pd.to_numeric(pred_df["predicted_roas_p50"], errors="coerce").mean())
    return info


def _build_compact_payload(pred_df: pd.DataFrame, limit: int = 200) -> List[Dict[str, Any]]:
    """Build a compact list of campaign dicts for the prompt.

    Send raw data to GPT and let it parse columns itself - no preprocessing.
    Limit rows to `limit` for safety; the caller can pass a higher value if desired.
    """
    # Take top rows by any numeric column that might indicate spend/volume
    source = pred_df.head(limit).copy()
    
    # Convert to raw dict format for GPT to interpret - send everything as strings
    raw_data = []
    for _, row in source.iterrows():
        raw_row = {}
        for col, val in row.items():
            # Convert everything to string, let GPT figure out the data types
            raw_row[col] = str(val) if val is not None else ""
        raw_data.append(raw_row)
    
    return raw_data


def get_gpt_recommendations(
    pred_df: pd.DataFrame,
    model: str | None = None,
    limit: int = 200,
) -> Dict[int, Dict[str, Any]]:
    """Return GPT-powered recommendations per campaign.

    Parameters:
    - pred_df: dataframe returned by `load_predictions(...)` containing at
      least `row_index` and ROAS columns.
    - model: override model name. Defaults to env `OPENAI_MODEL` or 'gpt-5'.
    - limit: max number of campaigns to include in the prompt for cost control.

    Returns: mapping of row_index -> { action, rationale, budget_change_pct? }
    """
    # Reuse existing singleton client if possible
    client = None
    if get_faq_gpt is not None:
        try:
            faq = get_faq_gpt()
            client = getattr(faq, "client", None)
        except Exception:
            client = None

    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if OpenAI is None or not api_key:
            return {}
        client = OpenAI(api_key=api_key)
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    rows = _build_compact_payload(pred_df, limit=limit)
    if not rows:
        return {}

    meta = _summarize_dataframe(pred_df)

    system = (
        "You are a senior growth analyst optimizing mobile game UA. "
        "Your task is to classify campaigns into: Scale, Maintain, Reduce, Cut, "
        "and suggest an optional budget delta. You will receive raw campaign data "
        "with mixed data types. First, identify which columns contain numeric data "
        "(like cost, revenue, installs, ROAS, retention rates) and which are categorical "
        "(like game, platform, channel, country). Parse currency strings like '$78.00', "
        "'$-', and percentage strings like '31%' to numeric values. Calculate ROAS "
        "from cost/revenue data if ROAS columns are missing. Always be concise and actionable. "
        "IMPORTANT: You MUST provide recommendations for ALL campaigns in the data. "
        "Do not skip any campaigns - analyze each one and provide an action."
    )

    user = (
        "Decide actions for the campaigns below. Return STRICT JSON only, schema:"
        "\n{\n  \"recommendations\": [\n    {\n      \"row_index\": int,\n      \"action\": one of [\"Scale\", \"Maintain\", \"Reduce\", \"Cut\"],\n      \"rationale\": short, <= 20 words,\n      \"budget_change_pct\": number between -100 and 200 (optional)\n    }\n  ]\n}\n"
        "Guidelines: Favor Scale when ROAS >= 1.5 with narrow uncertainty; Maintain when near 1.0 and stable; "
        "Reduce when < 1.0 but promising; Cut when clearly unprofitable or highly uncertain.\n"
        "Data parsing rules:\n"
        "- Currency strings: '$78.00' → 78.0, '$-' → 0.0, '$0.00' → 0.0\n"
        "- Percentage strings: '31%' → 0.31, '0%' → 0.0\n"
        "- Numeric strings: '77' → 77, '0' → 0\n"
        "- Calculate ROAS = revenue/cost when ROAS columns are missing\n"
        "CRITICAL: You must analyze EVERY campaign and provide a recommendation. "
        "If cost is '$-' or 0, classify as 'Cut'. If revenue is '$-' or 0, be conservative.\n"
        f"Dataset summary: {json.dumps(meta)}\n"
        f"Campaign slice (max {limit} rows, prioritized by spend): {json.dumps(rows)}"
    )

    try:
        resp = client.chat.completions.create(
            model=model_name,
            temperature=0.2,
            max_tokens=1200,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = resp.choices[0].message.content
    except Exception as e:
        print(f"GPT API error: {e}")
        text = ""

    recs = []
    if text:
        try:
            data = json.loads(text)
            recs = data.get("recommendations", [])
            print(f"GPT returned {len(recs)} recommendations")
        except Exception as e:
            print(f"GPT response parsing failed: {e}")
            print(f"Raw response: {text}")
            recs = []
    else:
        print("GPT returned empty response")

    out: Dict[int, Dict[str, Any]] = {}
    for rec in recs:
        try:
            idx = int(rec.get("row_index"))
            action = str(rec.get("action", "")).strip()
            rationale = str(rec.get("rationale", "")).strip()
            budget = rec.get("budget_change_pct")
            payload: Dict[str, Any] = {"action": action, "rationale": rationale}
            if isinstance(budget, (int, float)):
                payload["budget_change_pct"] = float(budget)
            out[idx] = payload
        except Exception:
            continue
    if out:
        return out

    # Heuristic fallback when LLM response is empty or API failed
    try:
        fallback_out: Dict[int, Dict[str, Any]] = {}
        df = pd.DataFrame(rows)
        # Compute ROAS proxy
        roas = None
        if "predicted_roas_p50" in df.columns:
            roas = pd.to_numeric(df["predicted_roas_p50"], errors="coerce")
        else:
            cost_s = pd.to_numeric(df.get("cost", 0), errors="coerce").fillna(0)
            rev_s = pd.to_numeric(df.get("revenue", 0), errors="coerce").fillna(0)
            roas = (rev_s / (cost_s + 1e-9)).fillna(0)

        conf = pd.to_numeric(df.get("confidence_interval", 0.8), errors="coerce").fillna(0.8)
        for i, r in df.iterrows():
            idx = int(r.get("row_index", i))
            r_roas = float(roas.iloc[i]) if len(roas) > i else 0.0
            r_conf = float(conf.iloc[i]) if len(conf) > i else 0.8
            if r_roas >= 1.5 and r_conf < 0.5:
                act = "Scale"
            elif r_roas >= 1.0 and r_conf < 0.8:
                act = "Maintain"
            elif r_roas >= 0.5:
                act = "Reduce"
            else:
                act = "Cut"
            fallback_out[idx] = {
                "action": act,
                "rationale": f"Heuristic: roas~{r_roas:.2f}, conf~{r_conf:.2f}",
            }
        return fallback_out
    except Exception:
        return {}


