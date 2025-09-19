"""
Data ingestion and normalization system for GameLens AI
"""

import os
import uuid
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import re

from sqlalchemy.orm import Session
from .db import get_db_session
from .models import Dataset, PlatformEnum
from .naming import make_canonical_name


def detect_platform_and_hierarchy(df: pd.DataFrame, filename: str) -> Dict[str, Any]:
    """
    Detect platform and extract hierarchy information from dataframe and filename
    
    Returns:
        Dict with platform, channel, game, countries
    """
    result = {
        'platform': PlatformEnum.UNKNOWN,
        'channel': None,
        'game': 'Unknown Game',
        'countries': None
    }
    
    # Detect platform from filename or data
    filename_lower = filename.lower()
    if 'unity' in filename_lower:
        result['platform'] = PlatformEnum.UNITY_ADS
    elif 'mistplay' in filename_lower:
        result['platform'] = PlatformEnum.MISTPLAY
    elif 'facebook' in filename_lower:
        result['platform'] = PlatformEnum.FACEBOOK
    elif 'google' in filename_lower:
        result['platform'] = PlatformEnum.GOOGLE
    elif 'tiktok' in filename_lower:
        result['platform'] = PlatformEnum.TIKTOK
    
    # Detect channel from filename or data
    if 'android' in filename_lower:
        result['channel'] = 'android'
    elif 'ios' in filename_lower:
        result['channel'] = 'ios'
    elif 'web' in filename_lower:
        result['channel'] = 'web'
    
    # Try to extract from data columns
    def _classify_platform(val: str) -> Optional[PlatformEnum]:
        v = str(val).strip().lower()
        if 'unity' in v:
            return PlatformEnum.UNITY_ADS
        if 'mistplay' in v:
            return PlatformEnum.MISTPLAY
        if 'facebook' in v or 'meta' in v:
            return PlatformEnum.FACEBOOK
        if 'google' in v:
            return PlatformEnum.GOOGLE
        if 'tiktok' in v:
            return PlatformEnum.TIKTOK
        return None

    def _classify_channel(val: str) -> Optional[str]:
        v = str(val).strip().lower()
        if v in {'android', 'ios', 'web', 'desktop'}:
            return v
        return None

    platform_col_val = None
    channel_col_val = None
    if 'platform' in df.columns:
        vals = df['platform'].dropna().unique()
        if len(vals) > 0:
            platform_col_val = str(vals[0])
    if 'channel' in df.columns:
        vals = df['channel'].dropna().unique()
        if len(vals) > 0:
            channel_col_val = str(vals[0])

    # Determine if columns are inverted; correct accordingly
    detected_platform_from_platform = _classify_platform(platform_col_val) if platform_col_val is not None else None
    detected_channel_from_platform = _classify_channel(platform_col_val) if platform_col_val is not None else None
    detected_platform_from_channel = _classify_platform(channel_col_val) if channel_col_val is not None else None
    detected_channel_from_channel = _classify_channel(channel_col_val) if channel_col_val is not None else None

    # Case 1: platform column actually contains channel like Android/iOS
    # and channel column contains platform like Unity/Mistplay
    if detected_channel_from_platform and detected_platform_from_channel:
        result['channel'] = detected_channel_from_platform
        result['platform'] = detected_platform_from_channel
    else:
        # Use normal classification if available
        if detected_platform_from_platform:
            result['platform'] = detected_platform_from_platform
        if detected_channel_from_channel:
            result['channel'] = detected_channel_from_channel
    
    # Extract game name
    game_columns = ['game', 'title', 'app', 'app_name']
    for col in game_columns:
        if col in df.columns:
            games = df[col].dropna().unique()
            if len(games) > 0:
                result['game'] = str(games[0])
                break
    
    # Extract countries
    country_columns = ['country', 'countries', 'geo', 'region']
    for col in country_columns:
        if col in df.columns:
            countries = df[col].dropna().unique()
            if len(countries) > 0:
                if len(countries) == 1:
                    result['countries'] = [str(countries[0])]
                else:
                    result['countries'] = [str(c) for c in countries[:5]]  # Limit to 5 countries
                break
    
    return result


def normalize_columns(df: pd.DataFrame, platform: PlatformEnum) -> pd.DataFrame:
    """
    Normalize column names and data types for consistent processing
    
    Args:
        df: Input dataframe
        platform: Detected platform
        
    Returns:
        Normalized dataframe
    """
    df_normalized = df.copy()
    
    # Column mapping for different platforms - includes whitespace variants
    column_mappings = {
        PlatformEnum.UNITY_ADS: {
            'date': ['date', 'Date', 'DATE', ' date ', ' Date ', ' DATE '],
            'installs': ['installs', 'Installs', 'INSTALLS', 'installs_count', ' installs ', ' Installs ', ' INSTALLS '],
            'cost': ['cost', 'Cost', 'COST', 'spend', 'Spend', 'SPEND', 'adspend', 'AdSpend', 'ADSPEND', ' cost ', ' Cost ', ' COST ', ' spend ', ' Spend ', ' SPEND '],
            'revenue': ['revenue', 'Revenue', 'REVENUE', 'ad_revenue', ' revenue ', ' Revenue ', ' REVENUE ', ' ad_revenue '],
            'roas_d0': ['roas_d0', 'ROAS_D0', 'roas_0', 'ROAS_0', ' roas_d0 ', ' ROAS_D0 ', ' roas_0 ', ' ROAS_0 '],
            'roas_d1': ['roas_d1', 'ROAS_D1', 'roas_1', 'ROAS_1', ' roas_d1 ', ' ROAS_D1 ', ' roas_1 ', ' ROAS_1 '],
            'roas_d3': ['roas_d3', 'ROAS_D3', 'roas_3', 'ROAS_3', ' roas_d3 ', ' ROAS_D3 ', ' roas_3 ', ' ROAS_3 '],
            'roas_d7': ['roas_d7', 'ROAS_D7', 'roas_7', 'ROAS_7', ' roas_d7 ', ' ROAS_D7 ', ' roas_7 ', ' ROAS_7 '],
            'roas_d14': ['roas_d14', 'ROAS_D14', 'roas_14', 'ROAS_14', ' roas_d14 ', ' ROAS_D14 ', ' roas_14 ', ' ROAS_14 '],
            'roas_d30': ['roas_d30', 'ROAS_D30', 'roas_30', 'ROAS_30', ' roas_d30 ', ' ROAS_D30 ', ' roas_30 ', ' ROAS_30 '],
            'roas_d60': ['roas_d60', 'ROAS_D60', 'roas_60', 'ROAS_60', ' roas_d60 ', ' ROAS_D60 ', ' roas_60 ', ' ROAS_60 '],
            'roas_d90': ['roas_d90', 'ROAS_D90', 'roas_90', 'ROAS_90', ' roas_d90 ', ' ROAS_D90 ', ' roas_90 ', ' ROAS_90 '],
            'retention_d1': ['retention_d1', 'RETENTION_D1', 'retention_1', 'RETENTION_1', ' retention_d1 ', ' RETENTION_D1 ', ' retention_1 ', ' RETENTION_1 '],
            'retention_d3': ['retention_d3', 'RETENTION_D3', 'retention_3', 'RETENTION_3', ' retention_d3 ', ' RETENTION_D3 ', ' retention_3 ', ' RETENTION_3 '],
            'retention_d7': ['retention_d7', 'RETENTION_D7', 'retention_7', 'RETENTION_7', ' retention_d7 ', ' RETENTION_D7 ', ' retention_7 ', ' RETENTION_7 '],
            'retention_d30': ['retention_d30', 'RETENTION_D30', 'retention_30', 'RETENTION_30', ' retention_d30 ', ' RETENTION_D30 ', ' retention_30 ', ' RETENTION_30 '],
        },
        PlatformEnum.MISTPLAY: {
            'date': ['date', 'Date', 'DATE', ' date ', ' Date ', ' DATE '],
            'installs': ['installs', 'Installs', 'INSTALLS', 'installs_count', ' installs ', ' Installs ', ' INSTALLS '],
            'cost': ['cost', 'Cost', 'COST', 'spend', 'Spend', 'SPEND', 'adspend', 'AdSpend', 'ADSPEND', ' cost ', ' Cost ', ' COST ', ' spend ', ' Spend ', ' SPEND '],
            'revenue': ['revenue', 'Revenue', 'REVENUE', 'ad_revenue', ' revenue ', ' Revenue ', ' REVENUE ', ' ad_revenue '],
            'roas_d0': ['roas_d0', 'ROAS_D0', 'roas_0', 'ROAS_0', ' roas_d0 ', ' ROAS_D0 ', ' roas_0 ', ' ROAS_0 '],
            'roas_d1': ['roas_d1', 'ROAS_D1', 'roas_1', 'ROAS_1', ' roas_d1 ', ' ROAS_D1 ', ' roas_1 ', ' ROAS_1 '],
            'roas_d3': ['roas_d3', 'ROAS_D3', 'roas_3', 'ROAS_3', ' roas_d3 ', ' ROAS_D3 ', ' roas_3 ', ' ROAS_3 '],
            'roas_d7': ['roas_d7', 'ROAS_D7', 'roas_7', 'ROAS_7', ' roas_d7 ', ' ROAS_D7 ', ' roas_7 ', ' ROAS_7 '],
            'roas_d14': ['roas_d14', 'ROAS_D14', 'roas_14', 'ROAS_14', ' roas_d14 ', ' ROAS_D14 ', ' roas_14 ', ' ROAS_14 '],
            'roas_d30': ['roas_d30', 'ROAS_D30', 'roas_30', 'ROAS_30', ' roas_d30 ', ' ROAS_D30 ', ' roas_30 ', ' ROAS_30 '],
            'roas_d60': ['roas_d60', 'ROAS_D60', 'roas_60', 'ROAS_60', ' roas_d60 ', ' ROAS_D60 ', ' roas_60 ', ' ROAS_60 '],
            'roas_d90': ['roas_d90', 'ROAS_D90', 'roas_90', 'ROAS_90', ' roas_d90 ', ' ROAS_D90 ', ' roas_90 ', ' ROAS_90 '],
        }
    }
    
    # Apply column mapping
    mapping = column_mappings.get(platform, {})
    for target_col, possible_cols in mapping.items():
        for col in possible_cols:
            if col in df_normalized.columns:
                df_normalized = df_normalized.rename(columns={col: target_col})
                break
    
    # CRITICAL FIX: Handle whitespace variants for ALL platforms (not just known ones)
    # This prevents cost/revenue data loss when platform detection fails
    whitespace_fixes = {
        ' cost ': 'cost',
        ' revenue ': 'revenue', 
        ' Cost ': 'cost',
        ' Revenue ': 'revenue',
        ' COST ': 'cost',
        ' REVENUE ': 'revenue',
        ' cost': 'cost',
        ' revenue': 'revenue',
        'cost ': 'cost',
        'revenue ': 'revenue',
        ' installs ': 'installs',
        ' Installs ': 'installs',
        ' INSTALLS ': 'installs',
        ' date ': 'date',
        ' Date ': 'date',
        ' DATE ': 'date'
    }
    
    for old_col, new_col in whitespace_fixes.items():
        if old_col in df_normalized.columns and new_col not in df_normalized.columns:
            df_normalized = df_normalized.rename(columns={old_col: new_col})
    
    # Generic normalization for ROAS/Retention regardless of platform (case-insensitive)
    import re as _re
    rename_map: dict = {}
    for col in list(df_normalized.columns):
        col_str = str(col).strip()
        lc = col_str.lower()
        # ROAS_Dxx -> roas_dxx
        m = _re.match(r"^roas[_\s]*d\s*(\d+)$", lc, flags=_re.IGNORECASE)
        if m:
            day = m.group(1)
            rename_map[col] = f"roas_d{day}"
            continue
        # RETENTION_Dxx -> retention_dxx
        m2 = _re.match(r"^retention[_\s]*d\s*(\d+)$", lc, flags=_re.IGNORECASE)
        if m2:
            day = m2.group(1)
            rename_map[col] = f"retention_d{day}"
            continue
        # Common alt headers
        if lc in ['adspend', 'ad_spend']:
            rename_map[col] = 'cost'
    if rename_map:
        df_normalized = df_normalized.rename(columns=rename_map)

    # Ensure required columns exist
    required_columns = ['date', 'installs', 'cost', 'revenue']
    for col in required_columns:
        if col not in df_normalized.columns:
            df_normalized[col] = 0
    
    # Normalize data types
    if 'date' in df_normalized.columns:
        df_normalized['date'] = pd.to_datetime(df_normalized['date'], errors='coerce')
    
    # Smart currency cleaning - detect and extract numeric values from currency strings
    def clean_currency_value(value):
        """Extract numeric value from currency string"""
        if pd.isna(value) or value == '' or value == '-':
            return 0.0
        
        # Convert to string and clean
        str_val = str(value).strip()
        
        # Handle common currency indicators
        if str_val in ['$-', '$', '€-', '€', '£-', '£', '¥-', '¥', '-', '']:
            return 0.0
        
        # Remove currency symbols and clean - keep only digits, dots, commas, and minus
        cleaned = re.sub(r'[^\d.,\-]', '', str_val)
        
        # Handle negative values
        is_negative = '-' in cleaned
        cleaned = cleaned.replace('-', '')
        
        # Handle comma as thousands separator
        if ',' in cleaned and '.' in cleaned:
            # Format: 1,234.56
            cleaned = cleaned.replace(',', '')
        elif ',' in cleaned and '.' not in cleaned:
            # Could be either 1,234 or 1,234.56 (European format)
            parts = cleaned.split(',')
            if len(parts) == 2 and len(parts[1]) <= 2:
                # Likely European decimal format: 1234,56
                cleaned = cleaned.replace(',', '.')
            else:
                # Likely thousands separator: 1,234
                cleaned = cleaned.replace(',', '')
        
        # Convert to float
        try:
            result = float(cleaned)
            return -result if is_negative else result
        except (ValueError, TypeError):
            return 0.0
    
    # Convert numeric columns
    numeric_columns = ['installs']
    currency_columns = ['cost', 'revenue']  # Clean currency values
    roas_columns = [col for col in df_normalized.columns if col.startswith('roas_d')]
    retention_columns = [col for col in df_normalized.columns if col.startswith('retention_d')]
    
    # Convert truly numeric columns
    for col in numeric_columns + roas_columns + retention_columns:
        if col in df_normalized.columns:
            df_normalized[col] = pd.to_numeric(df_normalized[col], errors='coerce')
    
    # Clean currency columns - extract numeric values from currency strings
    for col in currency_columns:
        if col in df_normalized.columns:
            df_normalized[col] = df_normalized[col].apply(clean_currency_value)
    
    # Keep currency columns as strings for GPT to parse - don't normalize here
    # The GPT recommendation system will handle parsing currency strings
    
    # Fill missing values
    df_normalized = df_normalized.fillna(0)
    
    return df_normalized


def infer_date_range(df: pd.DataFrame) -> Tuple[Optional[date], Optional[date]]:
    """
    Infer date range from dataframe
    
    Returns:
        Tuple of (start_date, end_date)
    """
    if 'date' not in df.columns:
        return None, None
    
    try:
        dates = pd.to_datetime(df['date'], errors='coerce').dropna()
        if len(dates) == 0:
            return None, None
        
        start_date = dates.min().date()
        end_date = dates.max().date()
        return start_date, end_date
    except Exception:
        return None, None


def compute_schema_fingerprint(df: pd.DataFrame) -> str:
    """
    Compute a fingerprint of the dataframe schema
    
    Returns:
        Hash string representing the schema
    """
    # Get column names and types
    schema_info = {
        'columns': sorted(df.columns.tolist()),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'shape': df.shape
    }
    
    # Create hash
    schema_str = str(sorted(schema_info.items()))
    return hashlib.md5(schema_str.encode()).hexdigest()


def ingest_file(file_path: str, notes: Optional[str] = None) -> Dataset:
    """
    Ingest a file and create a dataset record
    
    Args:
        file_path: Path to the file to ingest
        notes: Optional notes about the dataset
        
    Returns:
        Dataset object
    """
    # Read the file
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Determine file type and read
    if file_path.suffix.lower() in ['.csv']:
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    
    if df.empty:
        raise ValueError("File is empty")
    
    # Detect platform and hierarchy
    hierarchy = detect_platform_and_hierarchy(df, file_path.name)
    
    # Normalize columns
    df_normalized = normalize_columns(df, hierarchy['platform'])
    
    # Compute schema fingerprint
    schema_fingerprint = compute_schema_fingerprint(df_normalized)
    
    # Infer date range
    start_date, end_date = infer_date_range(df_normalized)
    
    # Create dataset record
    dataset_id = uuid.uuid4()
    dataset = Dataset(
        id=dataset_id,
        raw_filename=file_path.name,
        source_platform=hierarchy['platform'].value,
        channel=hierarchy['channel'],
        game=hierarchy['game'],
        countries=hierarchy['countries'],
        records=len(df_normalized),
        data_start_date=start_date,
        data_end_date=date.today(),  # Set to current date as required
        schema_fingerprint=schema_fingerprint,
        notes=notes
    )
    
    # Helper to ensure canonical name uniqueness
    def _ensure_unique_name(db_session, base_name: str) -> str:
        name = base_name
        # If name exists, append short uuid until unique
        while db_session.query(Dataset).filter(Dataset.canonical_name == name).first() is not None:
            name = f"{base_name}-{str(uuid.uuid4())[:8]}"
        return name

    # Save to database
    db = get_db_session()
    try:
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        
        # Create canonical name using GPT
        canonical_name = make_canonical_name({
            'game': hierarchy['game'],
            'platform': hierarchy['platform'].value,
            'channel': hierarchy['channel'],
            'countries': hierarchy['countries'],
            'start_date': start_date,
            'end_date': end_date,
            'columns': df_normalized.columns.tolist(),
            'sample_rows': df_normalized.head(3).to_dict('records')
        })
        # Ensure uniqueness in DB
        dataset.canonical_name = _ensure_unique_name(db, canonical_name)
        db.commit()
        db.refresh(dataset)
        
        # Create storage directory and save normalized data
        storage_dir = Path("data/normalized") / str(dataset_id)
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        storage_path = storage_dir / "data.parquet"
        df_normalized.to_parquet(storage_path, index=False)
        
        # Update storage path and completion timestamp
        dataset.storage_path = str(storage_path)
        dataset.ingest_completed_at = datetime.now()
        db.commit()
        db.refresh(dataset)
        
        return dataset
        
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


def get_datasets(platform: Optional[str] = None, game: Optional[str] = None) -> List[Dataset]:
    """
    Get datasets with optional filtering
    
    Args:
        platform: Filter by platform
        game: Filter by game
        
    Returns:
        List of Dataset objects
    """
    db = get_db_session()
    try:
        query = db.query(Dataset)
        
        if platform:
            query = query.filter(Dataset.source_platform == platform)
        
        if game:
            query = query.filter(Dataset.game.ilike(f"%{game}%"))
        
        return query.order_by(Dataset.ingest_started_at.desc()).all()
    finally:
        db.close()


def get_dataset_by_id(dataset_id: str) -> Optional[Dataset]:
    """
    Get dataset by ID
    
    Args:
        dataset_id: Dataset UUID
        
    Returns:
        Dataset object or None
    """
    import uuid as _uuid
    db = get_db_session()
    try:
        # Ensure we compare using UUID objects when the column is UUID-typed
        ds_id = dataset_id
        try:
            if isinstance(dataset_id, str):
                ds_id = _uuid.UUID(dataset_id)
        except Exception:
            ds_id = dataset_id
        return db.query(Dataset).filter(Dataset.id == ds_id).first()
    finally:
        db.close()


def load_dataset_data(dataset: Dataset) -> pd.DataFrame:
    """
    Load the normalized data for a dataset
    
    Args:
        dataset: Dataset object
        
    Returns:
        DataFrame with normalized data
    """
    if not dataset.storage_path or not Path(dataset.storage_path).exists():
        raise FileNotFoundError(f"Data file not found: {dataset.storage_path}")
    
    return pd.read_parquet(dataset.storage_path)
