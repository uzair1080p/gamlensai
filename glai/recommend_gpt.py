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

    Only include the columns the model needs to reason about the decision to
    keep token usage low. Limit rows to `limit` for safety; the caller can pass
    a higher value if desired.
    """
    # Core fields
    base_candidates: List[str] = [
        "row_index",
        "predicted_roas_p50",
        "predicted_roas_p10",
        "predicted_roas_p90",
        "confidence_interval",
        "cost",
        "revenue",
        "ad_revenue",
        "installs",
    ]
    cols: List[str] = [c for c in base_candidates if c in pred_df.columns]

    # Add useful observed columns if available (roas_d*, retention_*, level_* aggregates)
    for pattern in ["roas_d", "retention_", "level_"]:
        for c in pred_df.columns:
            if isinstance(c, str) and c.startswith(pattern):
                cols.append(c)

    # De-duplicate while preserving order
    seen = set()
    cols = [x for x in cols if not (x in seen or seen.add(x))]
    # Prioritize top spend rows if cost exists, then append remaining head
    source = pred_df
    if "cost" in source.columns:
        try:
            source = source.sort_values("cost", ascending=False)
        except Exception:
            pass
    compact = source[cols].head(limit).copy()
    # Provide derived metrics when possible
    if "cost" in compact.columns and "installs" in compact.columns and "cpi" not in compact.columns:
        try:
            compact["cpi"] = (pd.to_numeric(compact["cost"], errors="coerce") / (pd.to_numeric(compact["installs"], errors="coerce") + 1e-9)).fillna(0.0)
        except Exception:
            pass
    # Ensure basic types for JSON
    for c in compact.columns:
        if isinstance(compact[c].dtype, pd.api.extensions.ExtensionDtype):
            compact[c] = compact[c].astype(object)
    return compact.to_dict(orient="records")


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
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-5")

    rows = _build_compact_payload(pred_df, limit=limit)
    if not rows:
        return {}

    meta = _summarize_dataframe(pred_df)

    system = (
        "You are a senior growth analyst optimizing mobile game UA. "
        "Your task is to classify campaigns into: Scale, Maintain, Reduce, Cut, "
        "and suggest an optional budget delta. Consider: predicted ROAS (p50), "
        "uncertainty (p90-p10 or confidence_interval), unit economics, and spend. "
        "If ROAS columns are missing, infer from cost, revenue and any available "
        "signals conservatively. Always be concise and actionable."
    )

    user = (
        "Decide actions for the campaigns below. Return STRICT JSON only, schema:"
        "\n{\n  \"recommendations\": [\n    {\n      \"row_index\": int,\n      \"action\": one of [\"Scale\", \"Maintain\", \"Reduce\", \"Cut\"],\n      \"rationale\": short, <= 20 words,\n      \"budget_change_pct\": number between -100 and 200 (optional)\n    }\n  ]\n}\n"
        "Guidelines: Favor Scale when p50 >= 1.5 with narrow uncertainty; Maintain when near 1.0 and stable; "
        "Reduce when < 1.0 but promising; Cut when clearly unprofitable or highly uncertain.\n"
        f"Dataset summary: {json.dumps(meta)}\n"
        f"Campaign slice (max {limit} rows, prioritized by spend): {json.dumps(rows)}"
    )

    try:
        resp = client.responses.create(
            model=model_name,
            temperature=0.2,
            max_output_tokens=1200,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = resp.output_text  # type: ignore[attr-defined]
    except Exception:
        # Fallback to empty mapping if API fails
        return {}

    try:
        data = json.loads(text)
        recs = data.get("recommendations", [])
    except Exception:
        return {}

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
    return out


