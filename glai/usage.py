import json
import os
from datetime import datetime
from typing import Dict, Any

USAGE_FILE = os.path.join("artifacts", "gpt_usage.json")
LOG_DIR = os.path.join("artifacts", "gpt_logs")


def _ensure_paths() -> None:
    os.makedirs(os.path.dirname(USAGE_FILE), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def _load_usage() -> Dict[str, Any]:
    _ensure_paths()
    if os.path.exists(USAGE_FILE):
        try:
            with open(USAGE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {"total_input": 0, "total_output": 0, "total_cost": 0.0, "events": []}
    return {"total_input": 0, "total_output": 0, "total_cost": 0.0, "events": []}


def _save_usage(data: Dict[str, Any]) -> None:
    _ensure_paths()
    with open(USAGE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _calc_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    # Default to GPT-5 mini pricing provided by user
    # Input $0.25 / 1M, Output $2.00 / 1M
    if "gpt-5-mini" in model:
        return (input_tokens * 0.25 / 1_000_000.0) + (output_tokens * 2.0 / 1_000_000.0)
    # Fallback conservative rate
    return (input_tokens * 0.5 / 1_000_000.0) + (output_tokens * 3.0 / 1_000_000.0)


def record_event(kind: str, model: str, input_tokens: int, output_tokens: int, meta: Dict[str, Any]) -> None:
    data = _load_usage()
    cost = _calc_cost(input_tokens, output_tokens, model)
    evt = {
        "ts": datetime.utcnow().isoformat(),
        "kind": kind,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": round(cost, 6),
        "meta": meta,
    }
    data["total_input"] = int(data.get("total_input", 0)) + int(input_tokens)
    data["total_output"] = int(data.get("total_output", 0)) + int(output_tokens)
    data["total_cost"] = float(data.get("total_cost", 0.0)) + float(evt["cost"])
    data.setdefault("events", []).append(evt)
    _save_usage(data)

    # also write raw log with prompts / responses for deep debug
    if meta.get("save_raw"):
        fname = os.path.join(LOG_DIR, f"{kind}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.json")
        try:
            with open(fname, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass


def get_usage() -> Dict[str, Any]:
    return _load_usage()


