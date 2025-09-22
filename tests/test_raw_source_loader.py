"""Tests for raw dataset loader utilities."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from glai.raw_source_loader import load_raw_source_dataframe


def test_load_raw_source_dataframe_reads_excel(tmp_path):
    """Datasets referencing .xlsx files should be retrievable for FAQ context."""

    df = pd.DataFrame({"cost": [10], "revenue": [20], "date": ["2024-01-01"]})
    source_path = tmp_path / "sample_dataset.xlsx"
    df.to_excel(source_path, index=False)

    dest_path = Path.cwd() / source_path.name
    dest_path.write_bytes(source_path.read_bytes())

    dataset = SimpleNamespace(raw_filename=source_path.name, source_platform="unknown", channel=None, storage_path=None)

    try:
        loaded = load_raw_source_dataframe(dataset)
        assert loaded is not None
        assert "cost" in loaded.columns
        assert loaded["cost"].iloc[0] == 10
    finally:
        if dest_path.exists():
            dest_path.unlink()
