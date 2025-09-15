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
from typing import Dict, Any, List

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


def _build_compact_payload(pred_df: pd.DataFrame, limit: int = 50) -> List[Dict[str, Any]]:
    """Build a compact list of campaign dicts for the prompt.

    Only include the columns the model needs to reason about the decision to
    keep token usage low. Limit rows to `limit` for safety; the caller can pass
    a higher value if desired.
    """
    cols: List[str] = [
        c for c in [
            "row_index",
            "predicted_roas_p50",
            "predicted_roas_p10",
            "predicted_roas_p90",
            "confidence_interval",
            "cost",
            "revenue",
        ]
        if c in pred_df.columns
    ]
    compact = (
        pred_df[cols]
        .head(limit)
        .copy()
    )
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

    system = (
        "You are a performance marketing analyst for mobile game UA. "
        "Classify each campaign into one of: Scale, Maintain, Reduce, Cut. "
        "Use thresholds anchored on business logic: higher predicted ROAS and "
        "narrow confidence intervals favor Scale; borderline ROAS near 1.0 with "
        "moderate uncertainty favors Maintain; sub-1.0 but recoverable is Reduce; "
        "clearly unprofitable or highly uncertain is Cut. Always be concise."
    )

    user = (
        "Decide actions for the following campaigns. Return STRICT JSON with the schema: "
        "{\n  \"recommendations\": [\n    {\n      \"row_index\": int,\n      \"action\": one of [\"Scale\", \"Maintain\", \"Reduce\", \"Cut\"],\n      \"rationale\": short string (<= 20 words),\n      \"budget_change_pct\": optional number between -100 and 200\n    }\n  ]\n}\n"
        "Use predicted_roas_p50, p10/p90, confidence_interval, and cost when present.\n"
        f"Campaign data: {json.dumps(rows)}"
    )

    try:
        resp = client.responses.create(
            model=model_name,
            temperature=0.2,
            max_output_tokens=1000,
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


