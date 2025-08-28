import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class GameLensFeatureEngineer:
    """Feature engineering for GameLens AI - creates early-day features for ROAS forecasting"""
    
    def __init__(self):
        self.feature_columns = []
        
    def create_cohort_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create cohort-level features from early-day data (D1-D3)"""
        
        # Get base data
        retention_df = data.get('retention', pd.DataFrame())
        roas_df = data.get('roas', pd.DataFrame())
        level_df = data.get('level_progression', pd.DataFrame())
        adspend_df = data.get('adspend_revenue', pd.DataFrame())
        
        if retention_df.empty or roas_df.empty:
            logger.error("Missing required retention or ROAS data")
            return pd.DataFrame()
            
        # Start with retention data as base
        features_df = retention_df.copy()
        
        # Add early retention features (D1-D3)
        features_df = self._add_retention_features(features_df)
        
        # Add early ROAS features (D0-D3)
        features_df = self._add_roas_features(features_df, roas_df)
        
        # Add level progression features
        if not level_df.empty:
            features_df = self._add_level_features(features_df, level_df)
            
        # Add cost and volume features
        if not adspend_df.empty:
            features_df = self._add_cost_features(features_df, adspend_df)
            
        # Add derived features
        features_df = self._add_derived_features(features_df)
        
        # Add target variables (D15, D30, D45, D90 ROAS)
        features_df = self._add_targets(features_df, roas_df)
        
        return features_df
    
    def _add_retention_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add early retention features"""
        features = df.copy()
        
        # Early retention rates
        retention_cols = [col for col in df.columns if 'retention_rate' in col]
        
        # D1 retention
        if 'retention_rate_d1' in df.columns:
            features['d1_retention'] = df['retention_rate_d1']
            
        # D3 retention
        if 'retention_rate_d3' in df.columns:
            features['d3_retention'] = df['retention_rate_d3']
            
        # D7 retention
        if 'retention_rate_d7' in df.columns:
            features['d7_retention'] = df['retention_rate_d7']
            
        # Retention slope (D1 to D3)
        if 'retention_rate_d1' in df.columns and 'retention_rate_d3' in df.columns:
            features['retention_slope_d1_d3'] = (df['retention_rate_d3'] - df['retention_rate_d1']) / 2
            
        # Retention decay rate
        if 'retention_rate_d1' in df.columns and 'retention_rate_d7' in df.columns:
            features['retention_decay_rate'] = (df['retention_rate_d1'] - df['retention_rate_d7']) / 6
            
        return features
    
    def _add_roas_features(self, features_df: pd.DataFrame, roas_df: pd.DataFrame) -> pd.DataFrame:
        """Add early ROAS features"""
        features = features_df.copy()
        
        # Merge ROAS data
        roas_cols = [col for col in roas_df.columns if col not in ['platform']]
        roas_subset = roas_df[roas_cols].copy()
        
        # Merge on country and installs
        merge_cols = ['country', 'installs']
        features = features.merge(roas_subset, on=merge_cols, how='left', suffixes=('', '_roas'))
        
        # Early ROAS features
        roas_cols = [col for col in features.columns if 'roas_d' in col]
        
        # D0 ROAS
        if 'roas_d0' in features.columns:
            features['d0_roas'] = features['roas_d0']
            
        # D1 ROAS
        if 'roas_d1' in features.columns:
            features['d1_roas'] = features['roas_d1']
            
        # D3 ROAS
        if 'roas_d3' in features.columns:
            features['d3_roas'] = features['roas_d3']
            
        # ROAS growth rate (D0 to D3)
        if 'roas_d0' in features.columns and 'roas_d3' in features.columns:
            features['roas_growth_rate_d0_d3'] = (features['roas_d3'] - features['roas_d0']) / 3
            
        # ROAS acceleration
        if 'roas_d1' in features.columns and 'roas_d3' in features.columns:
            features['roas_acceleration'] = features['roas_d3'] - features['roas_d1']
            
        return features
    
    def _add_level_features(self, features_df: pd.DataFrame, level_df: pd.DataFrame) -> pd.DataFrame:
        """Add level progression features"""
        features = features_df.copy()
        
        # Merge level data
        level_cols = [col for col in level_df.columns if col not in ['platform']]
        level_subset = level_df[level_cols].copy()
        
        # Merge on country and installs
        merge_cols = ['country', 'installs']
        features = features.merge(level_subset, on=merge_cols, how='left', suffixes=('', '_level'))
        
        # Level completion rates
        level_cols = [col for col in features.columns if 'level' in col and 'events' in col]
        
        for col in level_cols:
            level_num = col.split('_')[0].replace('level', '')
            if level_num.isdigit():
                features[f'level_{level_num}_completion_rate'] = features[col] / features['installs']
                
        # Early level completion (levels 10-50)
        early_levels = ['level_10_completion_rate', 'level_20_completion_rate', 
                       'level_30_completion_rate', 'level_40_completion_rate', 'level_50_completion_rate']
        
        available_early = [col for col in early_levels if col in features.columns]
        if available_early:
            features['avg_early_level_completion'] = features[available_early].mean(axis=1)
            
        # Drop-off points
        if 'level_10_completion_rate' in features.columns and 'level_20_completion_rate' in features.columns:
            features['dropoff_10_to_20'] = features['level_10_completion_rate'] - features['level_20_completion_rate']
            
        if 'level_20_completion_rate' in features.columns and 'level_30_completion_rate' in features.columns:
            features['dropoff_20_to_30'] = features['level_20_completion_rate'] - features['level_30_completion_rate']
            
        # Ad engagement
        if 'rv_ads_events' in features.columns:
            features['ad_engagement_rate'] = features['rv_ads_events'] / features['installs']
            
        # Purchase events
        purchase_cols = [col for col in features.columns if 'purchase' in col or 'iap' in col or 'subscription' in col]
        if purchase_cols:
            features['total_purchase_events'] = features[purchase_cols].sum(axis=1)
            features['purchase_rate'] = features['total_purchase_events'] / features['installs']
            
        return features
    
    def _add_cost_features(self, features_df: pd.DataFrame, adspend_df: pd.DataFrame) -> pd.DataFrame:
        """Add cost and volume features"""
        features = features_df.copy()
        
        # Aggregate daily data by country
        daily_agg = adspend_df.groupby('country').agg({
            'installs': 'sum',
            'cost': 'sum',
            'ad_revenue': 'sum',
            'revenue': 'sum'
        }).reset_index()
        
        # Calculate metrics
        daily_agg['cpi'] = daily_agg['cost'] / daily_agg['installs']
        daily_agg['arpu'] = (daily_agg['ad_revenue'] + daily_agg['revenue']) / daily_agg['installs']
        daily_agg['ad_arpu'] = daily_agg['ad_revenue'] / daily_agg['installs']
        daily_agg['iap_arpu'] = daily_agg['revenue'] / daily_agg['installs']
        
        # Merge with features
        features = features.merge(daily_agg, on='country', how='left', suffixes=('', '_daily'))
        
        return features
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features and interactions"""
        features = df.copy()
        
        # Volume features
        if 'installs' in features.columns:
            features['log_installs'] = np.log1p(features['installs'])
            
        # Cost efficiency
        if 'cpi' in features.columns and 'd1_roas' in features.columns:
            features['cost_efficiency'] = features['d1_roas'] / features['cpi']
            
        # Retention-ROAS interaction
        if 'd1_retention' in features.columns and 'd1_roas' in features.columns:
            features['retention_roas_interaction'] = features['d1_retention'] * features['d1_roas']
            
        # Platform features (if available)
        if 'platform' in features.columns:
            features['is_android'] = (features['platform'] == 'Android').astype(int)
            features['is_ios'] = (features['platform'] == 'iOS').astype(int)
            
        return features
    
    def _add_targets(self, features_df: pd.DataFrame, roas_df: pd.DataFrame) -> pd.DataFrame:
        """Add target variables for forecasting"""
        features = features_df.copy()
        
        # Target ROAS values
        target_days = [15, 30, 45, 90]
        
        for day in target_days:
            col_name = f'roas_d{day}'
            if col_name in roas_df.columns:
                features[f'target_roas_d{day}'] = roas_df[col_name]
                
        return features
    
    def get_feature_columns(self) -> List[str]:
        """Get list of engineered feature columns"""
        return self.feature_columns
    
    def prepare_training_data(self, features_df: pd.DataFrame, target_day: int = 30) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for model training"""
        
        # Define feature columns (exclude targets and metadata)
        exclude_cols = ['country', 'platform', 'installs'] + [col for col in features_df.columns if 'target_' in col]
        
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # Remove rows with missing target
        target_col = f'target_roas_d{target_day}'
        if target_col not in features_df.columns:
            raise ValueError(f"Target column {target_col} not found")
            
        valid_data = features_df.dropna(subset=[target_col])
        
        # Prepare X and y
        X = valid_data[feature_cols].fillna(0)
        y = valid_data[target_col]
        
        # Store feature columns for later use
        self.feature_columns = feature_cols
        
        return X, y
