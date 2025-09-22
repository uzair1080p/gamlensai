"""Utilities for locating and loading raw campaign data files."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
import json
import httpx


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


def load_cleaned_dataframe(dataset, endpoint_url: str = "http://170.64.236.80:5678/webhook/clean-ua") -> Optional[pd.DataFrame]:
    """Send the dataset's raw file to the n8n cleaner and return a cleaned dataframe.

    Fallbacks to normal raw loader if the cleaner is unavailable or returns invalid data.
    """
    # Reuse candidate resolution from raw loader to find the most likely file path
    raw_df = None
    try:
        raw_df = load_raw_source_dataframe(dataset)
    except Exception:
        raw_df = None

    # If we already got a dataframe, try to discover the exact file path used
    # by searching the same candidate list logic again. We'll resend that file
    # to the cleaner to ensure strict normalization (currency -> numeric, etc.).
    # When we cannot resolve a single file path, just return the raw_df.
    try:
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
            requested_path = Path(str(raw_filename).strip())
            add_candidate(requested_path)
            for directory in _candidate_directories(dataset):
                add_candidate(directory / requested_path.name)

        if not candidate_paths:
            for path in _gather_files_from_globs():
                add_candidate(path)

        if not candidate_paths:
            # No file path to send; return whatever raw_df we may have
            return raw_df

        # Prefer the first viable candidate
        file_path = candidate_paths[0]

        # Post to n8n cleaner
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "application/octet-stream")}
            try:
                with httpx.Client(timeout=60) as client:
                    resp = client.post(endpoint_url, files=files)
                if resp.status_code != 200:
                    return raw_df
                # Try JSON parse first; if not JSON, try to coerce
                try:
                    payload = resp.json()
                except json.JSONDecodeError:
                    payload = json.loads(resp.text)

                # Expect either a list of records or an object with a key like 'data'
                if isinstance(payload, list):
                    cleaned = pd.DataFrame(payload)
                elif isinstance(payload, dict):
                    # common patterns: { data: [...] } or { rows: [...] }
                    records = None
                    for key in ("data", "rows", "cleaned", "result"):
                        if key in payload and isinstance(payload[key], list):
                            records = payload[key]
                            break
                    if records is None:
                        # Try flattening dict-of-lists
                        try:
                            cleaned = pd.json_normalize(payload)
                        except Exception:
                            return raw_df
                    else:
                        cleaned = pd.DataFrame(records)
                else:
                    return raw_df

                if cleaned is None or cleaned.empty:
                    return raw_df

                return _normalize_columns(cleaned)
            except Exception:
                # Any failure -> fallback to raw_df
                return raw_df
    except Exception:
        return raw_df



__all__ = ["load_raw_source_dataframe"]

