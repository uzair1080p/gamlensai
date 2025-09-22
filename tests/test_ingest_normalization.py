"""Regression tests for ingest column normalization."""

import pandas as pd

from glai.ingest import normalize_columns
from glai.models import PlatformEnum


def test_normalize_columns_preserves_cost_and_revenue_for_unknown_platform():
    """Whitespace-padded cost/revenue headers must retain their values."""

    df = pd.DataFrame(
        {
            " date ": ["2024-01-01", "2024-01-02"],
            "Installs": [100, 200],
            " cost ": [123.45, 67.89],
            " revenue ": [222.0, 333.0],
        }
    )

    normalized = normalize_columns(df, PlatformEnum.UNKNOWN)

    assert "cost" in normalized.columns
    assert "revenue" in normalized.columns
    assert normalized["cost"].tolist() == [123.45, 67.89]
    assert normalized["revenue"].tolist() == [222.0, 333.0]
    assert normalized["installs"].tolist() == [100, 200]
    assert list(normalized["date"].dt.date) == [pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-01-02").date()]
