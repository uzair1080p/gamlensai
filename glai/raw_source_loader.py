"""Utilities for locating and loading raw campaign data files."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd


def _candidate_directories(dataset) -> List[Path]:
    """Return directories to search for raw files."""
    dirs: List[Path] = [Path("."), Path("Campaign Data"), Path("data"), Path("data/raw"), Path("artifacts")]

    storage_path = getattr(dataset, "storage_path", None)
    if storage_path:
        try:
            storage_dir = Path(storage_path).resolve().parent
            if storage_dir.exists():
                dirs.append(storage_dir)
        except Exception:
            pass

    unique_dirs: List[Path] = []
    seen = set()
    for directory in dirs:
        try:
            resolved = directory.resolve()
        except Exception:
            continue
        if resolved in seen or not directory.exists():
            continue
        seen.add(resolved)
        unique_dirs.append(directory)
    return unique_dirs


def _gather_files_from_globs() -> List[Path]:
    """Fallback to broad glob searches mirroring legacy behaviour."""
    patterns = [
        "Campaign Data/**/*.csv",
        "Campaign Data/**/*.xlsx",
        "Campaign Data/**/*.xls",
        "data/raw/*.csv",
        "data/raw/*.xlsx",
        "data/raw/*.xls",
        "*.csv",
        "*.xlsx",
        "*.xls",
    ]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(Path().glob(pattern))
    filtered = [
        path
        for path in files
        if not any(token in path.name.lower() for token in ["template", "feature_importance", "gamlens_env", "demo_"])
        and not path.name.startswith("~$")  # Exclude temporary Excel files
    ]
    return filtered


def _read_dataframe(file_path: Path) -> Optional[pd.DataFrame]:
    """Load a dataframe from the provided file path."""
    try:
        if file_path.suffix.lower() == ".csv":
            return pd.read_csv(file_path)
        if file_path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(file_path)
    except Exception:
        return None
    return None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]
    
    # Handle duplicate columns by keeping the first occurrence
    df = df.loc[:, ~df.columns.duplicated()]
    
    return df


def load_raw_source_dataframe(dataset) -> Optional[pd.DataFrame]:
    """Load the original CSV/XLSX backing a dataset if it exists on disk."""

    raw_filename = getattr(dataset, "raw_filename", None)
    candidate_paths: List[Path] = []
    seen_paths = set()

    def add_candidate(path: Path) -> None:
        try:
            resolved = path.resolve()
        except Exception:
            return
        if not path.exists() or resolved in seen_paths:
            return
        seen_paths.add(resolved)
        candidate_paths.append(path)

    if raw_filename:
        raw_filename = str(raw_filename).strip()
        requested_path = Path(raw_filename)

        # Direct match (absolute or relative)
        add_candidate(requested_path)

        search_dirs = _candidate_directories(dataset)

        # Match by exact filename inside search directories
        for directory in search_dirs:
            add_candidate(directory / requested_path.name)

        # Try alternate extensions (csv/xlsx/xls) when only stem matches
        if not candidate_paths:
            stem = requested_path.stem or requested_path.name
            extensions = [requested_path.suffix.lower()] if requested_path.suffix else []
            for ext in [".csv", ".xlsx", ".xls"]:
                if ext not in extensions:
                    extensions.append(ext)
            for directory in search_dirs:
                for ext in extensions:
                    add_candidate(directory / f"{stem}{ext}")

        # Case-insensitive search inside directories
        if not candidate_paths:
            target_name = requested_path.name.lower()
            for directory in search_dirs:
                try:
                    for ext in ("*.csv", "*.xlsx", "*.xls"):
                        for path in directory.rglob(ext):
                            if path.name.lower() == target_name:
                                add_candidate(path)
                                break
                        if candidate_paths:
                            break
                except Exception:
                    continue
                if candidate_paths:
                    break

    # Fallback to legacy heuristics when raw filename lookup fails
    if not candidate_paths:
        specific_paths: List[str] = []
        platform = getattr(dataset, "source_platform", None)
        channel = getattr(dataset, "channel", None)

        if platform == "unity_ads" and channel == "android":
            specific_paths = [
                "Campaign Data/Unity Ads/Android/Adspend and Revenue data.csv",
                "Campaign Data/Unity Ads/Android/Adspend and Revenue data.xlsx",
            ]
        elif platform == "unity_ads" and channel == "ios":
            specific_paths = [
                "Campaign Data/Unity Ads/iOS/Adspend+ Revenue .csv",
                "Campaign Data/Unity Ads/iOS/Adspend+ Revenue .xlsx",
            ]
        elif platform == "mistplay" and channel == "android":
            specific_paths = [
                "Campaign Data/Mistplay/Android/Adspend & Revenue.csv",
                "Campaign Data/Mistplay/Android/Adspend & Revenue.xlsx",
            ]

        for file_path in specific_paths:
            add_candidate(Path(file_path))

        if not candidate_paths:
            for path in _gather_files_from_globs():
                add_candidate(path)

    for path in candidate_paths:
        df = _read_dataframe(path)
        if df is None or df.empty:
            continue
        return _normalize_columns(df)

    return None


__all__ = ["load_raw_source_dataframe"]

