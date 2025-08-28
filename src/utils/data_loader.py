import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GameLensDataLoader:
    """Data loader for GameLens AI Phase 1 - handles Unity Ads CSV ingestion and normalization"""
    
    def __init__(self, data_dir: Optional[str] = "Campaign Data"):
        # Resolve data directory robustly across different working directories (root vs notebooks)
        self.platforms = ['Android', 'iOS']
        self.data_dir = self._resolve_data_dir(data_dir)
        logger.info(f"Using data directory: {self.data_dir}")
        
    def _resolve_data_dir(self, data_dir: Optional[str]) -> str:
        """Resolve the data directory considering various common execution contexts."""
        candidates: List[str] = []
        if data_dir:
            candidates.append(data_dir)
        # Project root inferred from this file location: src/utils/ -> project root is two levels up
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        candidates.append(os.path.join(project_root, "Campaign Data"))
        # Current working directory
        cwd = os.getcwd()
        candidates.extend([
            os.path.join(cwd, "Campaign Data"),
            os.path.abspath(os.path.join(cwd, os.pardir, "Campaign Data")),
        ])
        
        for path in candidates:
            if os.path.exists(path):
                return path
        
        # Fall back to provided path even if not found; callers will see warnings later
        return data_dir if data_dir else "Campaign Data"
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV files from Unity Ads platforms"""
        data: Dict[str, Dict[str, pd.DataFrame]] = {}
        
        for platform in self.platforms:
            platform_dir = os.path.join(self.data_dir, "Unity Ads", platform)
            if os.path.exists(platform_dir):
                data[platform] = self._load_platform_data(platform_dir)
            else:
                logger.warning(f"Platform directory not found: {platform_dir}")
                
        return data
    
    def _load_platform_data(self, platform_dir: str) -> Dict[str, pd.DataFrame]:
        """Load all CSV files for a specific platform"""
        platform_data: Dict[str, pd.DataFrame] = {}
        
        # Map expected file patterns to standardized names
        file_patterns = {
            'adspend_revenue': ['Adspend', 'Adspend and Revenue', 'Adspend+ Revenue'],
            'level_progression': ['Level Progression'],
            'retention': ['retention'],
            'roas': ['ROAS']
        }
        
        for data_type, substrings in file_patterns.items():
            # Case-insensitive substring match against filenames in the directory
            try:
                filenames = os.listdir(platform_dir)
            except Exception as e:
                logger.error(f"Failed listing directory {platform_dir}: {e}")
                filenames = []
            match = next((f for f in filenames if any(s.lower() in f.lower() for s in substrings)), None)
            if match:
                file_path = os.path.join(platform_dir, match)
                try:
                    df = pd.read_csv(file_path)
                    platform_data[data_type] = self._clean_dataframe(df, data_type)
                    logger.info(f"Loaded {data_type} for {os.path.basename(platform_dir)}: {df.shape}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        return platform_data
    
    def _clean_dataframe(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Clean and standardize dataframe based on data type"""
        df = df.copy()
        
        # Remove any completely empty rows/columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        if data_type == 'adspend_revenue':
            # Standardize column names
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            # Convert date column if present
            if 'day' in df.columns:
                df['day'] = pd.to_datetime(df['day'])
                
            # Convert numeric columns
            numeric_cols = ['installs', 'cost', 'ad_revenue', 'revenue']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    
        elif data_type in ['retention', 'roas']:
            # Convert numeric columns
            for col in df.columns:
                if col not in ['country']:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    
        elif data_type == 'level_progression':
            # Convert event columns to numeric
            for col in df.columns:
                if col not in ['country']:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    
    def combine_platforms(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        """Combine data from multiple platforms into unified datasets"""
        combined: Dict[str, pd.DataFrame] = {}
        
        # Combine adspend_revenue data
        adspend_dfs: List[pd.DataFrame] = []
        for platform, platform_data in data.items():
            if 'adspend_revenue' in platform_data:
                df = platform_data['adspend_revenue'].copy()
                df['platform'] = platform
                adspend_dfs.append(df)
        
        if adspend_dfs:
            combined['adspend_revenue'] = pd.concat(adspend_dfs, ignore_index=True)
            
        # Combine retention data
        retention_dfs: List[pd.DataFrame] = []
        for platform, platform_data in data.items():
            if 'retention' in platform_data:
                df = platform_data['retention'].copy()
                df['platform'] = platform
                retention_dfs.append(df)
                
        if retention_dfs:
            combined['retention'] = pd.concat(retention_dfs, ignore_index=True)
            
        # Combine ROAS data
        roas_dfs: List[pd.DataFrame] = []
        for platform, platform_data in data.items():
            if 'roas' in platform_data:
                df = platform_data['roas'].copy()
                df['platform'] = platform
                roas_dfs.append(df)
                
        if roas_dfs:
            combined['roas'] = pd.concat(roas_dfs, ignore_index=True)
            
        # Combine level progression data
        level_dfs: List[pd.DataFrame] = []
        for platform, platform_data in data.items():
            if 'level_progression' in platform_data:
                df = platform_data['level_progression'].copy()
                df['platform'] = platform
                level_dfs.append(df)
                
        if level_dfs:
            combined['level_progression'] = pd.concat(level_dfs, ignore_index=True)
            
        return combined
    
    def validate_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """Validate data quality and return issues"""
        issues: Dict[str, List[str]] = {}
        
        for data_type, df in data.items():
            data_issues: List[str] = []
            
            # Check for missing values
            if df.empty:
                data_issues.append("Dataset is empty")
            else:
                missing_cols = df.columns[df.isnull().any()].tolist()
                if missing_cols:
                    data_issues.append(f"Missing values in columns: {missing_cols}")
                
            # Check for negative values in cost/revenue
            if data_type == 'adspend_revenue' and not df.empty:
                negative_cost = (df['cost'] < 0).sum()
                if negative_cost > 0:
                    data_issues.append(f"Negative cost values: {negative_cost} rows")
                    
            # Check for retention rates > 1
            if data_type == 'retention' and not df.empty:
                retention_cols = [col for col in df.columns if 'retention_rate' in col]
                for col in retention_cols:
                    invalid_retention = (df[col] > 1).sum()
                    if invalid_retention > 0:
                        data_issues.append(f"Invalid retention rates > 1 in {col}: {invalid_retention} rows")
                        
            if data_issues:
                issues[data_type] = data_issues
                
        return issues
