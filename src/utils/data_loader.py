import os
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GameLensDataLoader:
    """Data loader for GameLens AI Phase 1 - handles Unity Ads and Mistplay CSV ingestion and normalization"""

    def __init__(self, data_dir: str = "Campaign Data"):
        # Dynamically find the data directory
        current_dir = os.getcwd()
        project_root = current_dir
        # Check if running from notebooks/ or src/ or root
        if os.path.basename(current_dir) in ['notebooks', 'src']:
            project_root = os.path.dirname(current_dir)
        elif os.path.basename(current_dir) == 'utils': # If running from src/utils
            project_root = os.path.dirname(os.path.dirname(current_dir))

        potential_data_path = os.path.join(project_root, data_dir)
        if os.path.exists(potential_data_path):
            self.data_dir = potential_data_path
        else:
            # Fallback to current directory or parent if not found in project root
            if os.path.exists(os.path.join(current_dir, data_dir)):
                self.data_dir = os.path.join(current_dir, data_dir)
            elif os.path.exists(os.path.join(os.path.dirname(current_dir), data_dir)):
                self.data_dir = os.path.join(os.path.dirname(current_dir), data_dir)
            else:
                logger.warning(f"Data directory '{data_dir}' not found at '{potential_data_path}', '{current_dir}/{data_dir}', or '{os.path.dirname(current_dir)}/{data_dir}'. Using default: {data_dir}")
                self.data_dir = data_dir # Fallback to original if all else fails
        logger.info(f"Using data directory: {self.data_dir}")
        
        # Updated to include both platforms
        self.platforms = ['Unity Ads', 'Mistplay']
        self.supported_platforms = ['Unity Ads', 'Mistplay']
        
    def _get_platform_directories(self) -> Dict[str, List[str]]:
        """Get available platform directories and their subdirectories"""
        platform_dirs = {}
        
        for platform in self.supported_platforms:
            platform_path = os.path.join(self.data_dir, platform)
            if os.path.exists(platform_path):
                subdirs = [d for d in os.listdir(platform_path) 
                          if os.path.isdir(os.path.join(platform_path, d))]
                platform_dirs[platform] = subdirs
                logger.info(f"Found {platform}: {subdirs}")
            else:
                logger.warning(f"Platform directory not found: {platform_path}")
                
        return platform_dirs
    
    def _load_csv_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load CSV file with error handling"""
        try:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                logger.info(f"Loaded {file_path}: {df.shape}")
                return df
            else:
                logger.warning(f"File not found: {file_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def _standardize_unity_ads_data(self, df: pd.DataFrame, data_type: str, platform: str) -> pd.DataFrame:
        """Standardize Unity Ads data format with Game > Platform > Channel > Countries hierarchy"""
        df = df.copy()
        
        # Add platform and data type columns
        df['platform'] = platform
        df['data_type'] = data_type
        
        # Ensure Game > Platform > Channel > Countries hierarchy is present
        self._ensure_hierarchy_columns(df)
        
        # Standardize column names based on data type
        if data_type == 'adspend_revenue':
            # Handle different file naming conventions
            if 'ad_revenue' in df.columns:
                df['revenue'] = df['ad_revenue']
            if 'cost' in df.columns:
                df['spend'] = df['cost']
                
        elif data_type == 'retention':
            # Ensure retention columns are properly named
            retention_cols = [col for col in df.columns if 'retention_rate' in col]
            for col in retention_cols:
                if col not in df.columns:
                    df[col] = 0.0
                    
        elif data_type == 'roas':
            # Ensure ROAS columns are properly named
            roas_cols = [col for col in df.columns if 'roas_d' in col]
            for col in roas_cols:
                if col not in df.columns:
                    df[col] = 0.0
                    
        elif data_type == 'level_progression':
            # Handle level progression data
            if 'installs' in df.columns:
                df['total_installs'] = df['installs']
                
        return df
    
    def _standardize_mistplay_data(self, df: pd.DataFrame, data_type: str, platform: str) -> pd.DataFrame:
        """Standardize Mistplay data format with Game > Platform > Channel > Countries hierarchy"""
        df = df.copy()
        
        # Add platform and data type columns
        df['platform'] = platform
        df['data_type'] = data_type
        
        # Ensure Game > Platform > Channel > Countries hierarchy is present
        self._ensure_hierarchy_columns(df)
        
        # Standardize column names based on data type
        if data_type == 'adspend_revenue':
            # Mistplay uses 'cost' and 'ad_revenue'
            if 'cost' in df.columns:
                df['spend'] = df['cost']
            if 'ad_revenue' in df.columns:
                df['revenue'] = df['ad_revenue']
                
        elif data_type == 'retention':
            # Mistplay retention columns are already standardized
            retention_cols = [col for col in df.columns if 'retention_rate' in col]
            for col in retention_cols:
                if col not in df.columns:
                    df[col] = 0.0
                    
        elif data_type == 'roas':
            # Mistplay ROAS columns are already standardized
            roas_cols = [col for col in df.columns if 'roas_d' in col]
            for col in roas_cols:
                if col not in df.columns:
                    df[col] = 0.0
                    
        elif data_type == 'level_progression':
            # Handle Mistplay level progression data
            if 'installs' in df.columns:
                df['total_installs'] = df['installs']
                
        return df
    
    def _ensure_hierarchy_columns(self, df: pd.DataFrame) -> None:
        """Ensure Game > Platform > Channel > Countries hierarchy columns are present"""
        # Add missing hierarchy columns with default values if not present
        if 'game' not in df.columns:
            df['game'] = 'Unknown Game'
            logger.warning("Missing 'game' column - added with default value")
        
        if 'channel' not in df.columns:
            # Try to infer channel from subdirectory or use default
            if 'subdirectory' in df.columns:
                df['channel'] = df['subdirectory']
            else:
                df['channel'] = 'Unknown Channel'
            logger.warning("Missing 'channel' column - added with default value")
        
        if 'country' not in df.columns:
            df['country'] = 'Unknown Country'
            logger.warning("Missing 'country' column - added with default value")
        
        # Ensure platform column exists (should be set by calling function)
        if 'platform' not in df.columns:
            df['platform'] = 'Unknown Platform'
            logger.warning("Missing 'platform' column - added with default value")
    
    def load_platform_data(self, platform: str) -> Dict[str, pd.DataFrame]:
        """Load all data for a specific platform"""
        platform_data = {}
        platform_path = os.path.join(self.data_dir, platform)
        
        if not os.path.exists(platform_path):
            logger.warning(f"Platform directory not found: {platform_path}")
            return platform_data
            
        # Get subdirectories (Android, iOS, etc.)
        subdirs = [d for d in os.listdir(platform_path) 
                  if os.path.isdir(os.path.join(platform_path, d))]
        
        for subdir in subdirs:
            subdir_path = os.path.join(platform_path, subdir)
            logger.info(f"Processing {platform}/{subdir}")
            
            # Look for CSV files
            csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
            
            for csv_file in csv_files:
                file_path = os.path.join(subdir_path, csv_file)
                df = self._load_csv_file(file_path)
                
                if df is not None:
                    # Determine data type from filename
                    data_type = self._classify_data_type(csv_file)
                    
                    # Standardize data based on platform
                    if platform == 'Unity Ads':
                        df = self._standardize_unity_ads_data(df, data_type, platform)
                    elif platform == 'Mistplay':
                        df = self._standardize_mistplay_data(df, data_type, platform)
                    
                    # Add subdirectory info
                    df['subdirectory'] = subdir
                    
                    # Store with key: platform_subdirectory_datatype
                    key = f"{platform}_{subdir}_{data_type}"
                    platform_data[key] = df
                    
        return platform_data
    
    def _classify_data_type(self, filename: str) -> str:
        """Classify CSV file based on filename"""
        filename_lower = filename.lower()
        
        if any(keyword in filename_lower for keyword in ['adspend', 'revenue', 'cost']):
            return 'adspend_revenue'
        elif 'retention' in filename_lower:
            return 'retention'
        elif 'roas' in filename_lower:
            return 'roas'
        elif any(keyword in filename_lower for keyword in ['level', 'progression']):
            return 'level_progression'
        else:
            return 'unknown'
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all data from all platforms"""
        all_data = {}
        
        for platform in self.supported_platforms:
            platform_data = self.load_platform_data(platform)
            all_data.update(platform_data)
            
        logger.info(f"Loaded data from {len(all_data)} files")
        return all_data
    
    def combine_platform_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Combine data from different platforms by type"""
        combined_data = {
            'adspend_revenue': [],
            'retention': [],
            'roas': [],
            'level_progression': []
        }
        
        for key, df in data_dict.items():
            data_type = df['data_type'].iloc[0] if 'data_type' in df.columns else 'unknown'
            if data_type in combined_data:
                combined_data[data_type].append(df)
        
        # Concatenate dataframes for each type
        for data_type, dfs in combined_data.items():
            if dfs:
                combined_data[data_type] = pd.concat(dfs, ignore_index=True)
                logger.info(f"Combined {data_type}: {combined_data[data_type].shape}")
            else:
                combined_data[data_type] = pd.DataFrame()
                
        return combined_data
    
    def validate_data(self, combined_data: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """Validate combined data for completeness and quality with Game > Platform > Channel > Countries hierarchy"""
        validation_results = {}
        
        for data_type, df in combined_data.items():
            issues = []
            
            if df.empty:
                issues.append(f"No data found for {data_type}")
                continue
            
            # Check for Game > Platform > Channel > Countries hierarchy
            hierarchy_cols = ['game', 'platform', 'channel', 'country']
            missing_hierarchy = [col for col in hierarchy_cols if col not in df.columns]
            if missing_hierarchy:
                issues.append(f"Missing hierarchy columns: {missing_hierarchy}")
            
            # Check for required columns based on data type
            if data_type == 'adspend_revenue':
                required_cols = ['platform', 'spend', 'revenue']
                for col in required_cols:
                    if col not in df.columns:
                        issues.append(f"Missing required column: {col}")
                        
            elif data_type == 'retention':
                required_cols = ['platform', 'installs']
                retention_cols = [col for col in df.columns if 'retention_rate' in col]
                if not retention_cols:
                    issues.append("No retention rate columns found")
                    
            elif data_type == 'roas':
                required_cols = ['platform', 'installs']
                roas_cols = [col for col in df.columns if 'roas_d' in col]
                if not roas_cols:
                    issues.append("No ROAS columns found")
                    
            elif data_type == 'level_progression':
                required_cols = ['platform', 'installs']
                level_cols = [col for col in df.columns if 'level' in col and 'events' in col]
                if not level_cols:
                    issues.append("No level progression columns found")
            
            # Check for missing values in hierarchy columns
            for col in hierarchy_cols:
                if col in df.columns:
                    missing_count = df[col].isnull().sum()
                    if missing_count > 0:
                        issues.append(f"Missing {col} values: {missing_count}")
            
            # Check for missing values in key columns
            if 'platform' in df.columns:
                missing_platforms = df['platform'].isnull().sum()
                if missing_platforms > 0:
                    issues.append(f"Missing platform values: {missing_platforms}")
                    
            validation_results[data_type] = issues
            
        return validation_results
    
    def get_data_summary(self, combined_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Get summary statistics for all data types"""
        summary = {}
        
        for data_type, df in combined_data.items():
            if df.empty:
                summary[data_type] = {"status": "No data available"}
                continue
                
            summary[data_type] = {
                "shape": df.shape,
                "platforms": df['platform'].unique().tolist() if 'platform' in df.columns else [],
                "columns": df.columns.tolist(),
                "missing_values": df.isnull().sum().to_dict()
            }
            
        return summary
