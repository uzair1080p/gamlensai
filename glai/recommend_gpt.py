"""
DEPRECATED: This module has been replaced by gamlens/lib/ai.py
All AI recommendations now use the gamlens pipeline.
"""

import warnings
from pathlib import Path
import sys

# Add gamlens to path
gamlens_path = Path(__file__).resolve().parents[1] / "gamlens"
if str(gamlens_path) not in sys.path:
    sys.path.insert(0, str(gamlens_path))

# Import from gamlens
from lib.ai import build_payload_for_ai, ask_one_question

def get_gpt_recommendations(*args, **kwargs):
    """DEPRECATED: Use gamlens/lib/ai.py instead"""
    warnings.warn(
        "glai.recommend_gpt.get_gpt_recommendations is deprecated. "
        "Use gamlens.lib.ai.ask_one_question instead.",
        DeprecationWarning,
        stacklevel=2
    )
    raise NotImplementedError(
        "This function has been deprecated. Please use gamlens/lib/ai.py for AI recommendations."
    )
