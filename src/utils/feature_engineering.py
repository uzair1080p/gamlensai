import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class GameLensFeatureEngineer:
    """Feature engineering for GameLens AI Phase 1 - creates features from early-day data for ROAS forecasting"""
    
    def __init__(self):
        self.feature_columns = []
        self.feature_importance = {}
        
    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create comprehensive feature set from all data types"""
        logger.info("Starting feature engineering process...")
        
        # Initialize with retention data as base
        if 'retention' not in data or data['retention'].empty:
            raise ValueError("Retention data is required for feature engineering")
            
        features = data['retention'].copy()
        
        # Add features from different data types
        features = self._add_retention_features(features)
        features = self._add_roas_features(features, data.get('roas', pd.DataFrame()))
        features = self._add_level_progression_features(features, data.get('level_progression', pd.DataFrame()))
        features = self._add_cost_features(features, data.get('adspend_revenue', pd.DataFrame()))
        features = self._add_derived_features(features)
        features = self._add_platform_features(features)
        
        # Clean and validate features
        features = self._clean_features(features)
        
        logger.info(f"Feature engineering complete. Final shape: {features.shape}")
        self.feature_columns = [col for col in features.columns if col not in ['platform', 'subdirectory', 'data_type']]
        
        return features
    
    def _add_retention_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add retention-based features"""
        features = df.copy()
        
        # Basic retention features
        if 'retention_rate_d1' in df.columns:
            features['d1_retention'] = df['retention_rate_d1']
        if 'retention_rate_d3' in df.columns:
            features['d3_retention'] = df['retention_rate_d3']
        if 'retention_rate_d7' in df.columns:
            features['d7_retention'] = df['retention_rate_d7']
            
        # Retention ratios and slopes
        if 'retention_rate_d1' in df.columns and 'retention_rate_d3' in df.columns:
            features['retention_slope_d1_d3'] = (df['retention_rate_d3'] - df['retention_rate_d1']) / 2
            features['retention_ratio_d3_d1'] = df['retention_rate_d3'] / (df['retention_rate_d1'] + 1e-8)
            
        if 'retention_rate_d1' in df.columns and 'retention_rate_d7' in df.columns:
            features['retention_slope_d1_d7'] = (df['retention_rate_d7'] - df['retention_rate_d1']) / 6
            features['retention_ratio_d7_d1'] = df['retention_rate_d7'] / (df['retention_rate_d1'] + 1e-8)
            
        # Additional retention days if available
        for day in [2, 4, 5, 6, 10, 14, 18, 21, 24]:
            col_name = f'retention_rate_d{day}'
            if col_name in df.columns:
                features[f'd{day}_retention'] = df[col_name]
                
        return features
    
    def _add_roas_features(self, features: pd.DataFrame, roas_df: pd.DataFrame) -> pd.DataFrame:
        """Add ROAS-based features"""
        if roas_df.empty:
            logger.warning("No ROAS data available for feature engineering")
            return features
            
        # Merge ROAS data with features
        merge_cols = ['platform', 'subdirectory']
        available_cols = [col for col in merge_cols if col in roas_df.columns and col in features.columns]
        
        if not available_cols:
            logger.warning("No common columns found for ROAS data merge")
            return features
            
        # Merge on available columns
        merged = features.merge(roas_df, on=available_cols, how='left', suffixes=('', '_roas'))
        
        # Add ROAS features
        roas_cols = [col for col in roas_df.columns if 'roas_d' in col]
        
        for col in roas_cols:
            if col in merged.columns:
                day = col.split('_')[-1]  # Extract day number
                merged[f'roas_d{day}'] = merged[col]
                
        # Early ROAS features (most important for forecasting)
        early_roas_days = ['d0', 'd1', 'd2', 'd3', 'd7']
        for day in early_roas_days:
            col_name = f'roas_{day}'
            if col_name in merged.columns:
                merged[f'roas_{day}'] = merged[col_name]
                
        # ROAS ratios and slopes
        if 'roas_d1' in merged.columns and 'roas_d3' in merged.columns:
            merged['roas_slope_d1_d3'] = (merged['roas_d3'] - merged['roas_d1']) / 2
            merged['roas_ratio_d3_d1'] = merged['roas_d3'] / (merged['roas_d1'] + 1e-8)
            
        if 'roas_d1' in merged.columns and 'roas_d7' in merged.columns:
            merged['roas_slope_d1_d7'] = (merged['roas_d7'] - merged['roas_d1']) / 6
            merged['roas_ratio_d7_d1'] = merged['roas_d7'] / (merged['roas_d1'] + 1e-8)
            
        return merged
    
    def _add_level_progression_features(self, features: pd.DataFrame, level_df: pd.DataFrame) -> pd.DataFrame:
        """Add level progression features"""
        if level_df.empty:
            logger.warning("No level progression data available for feature engineering")
            return features
            
        # Merge level progression data
        merge_cols = ['platform', 'subdirectory']
        available_cols = [col for col in merge_cols if col in level_df.columns and col in features.columns]
        
        if not available_cols:
            logger.warning("No common columns found for level progression data merge")
            return features
            
        merged = features.merge(level_df, on=available_cols, how='left', suffixes=('', '_level'))
        
        # Find level progression columns
        level_cols = [col for col in level_df.columns if 'level' in col.lower() and 'events' in col.lower()]
        
        # Add level progression features
        for col in level_cols:
            if col in merged.columns:
                # Extract level number from column name
                level_match = col.split('_')[0] if '_' in col else col
                merged[f'level_{level_match}_events'] = merged[col]
                
        # Calculate level progression ratios
        level_numbers = []
        for col in level_cols:
            if col in merged.columns:
                try:
                    # Extract numeric level from column name
                    level_num = int(''.join(filter(str.isdigit, col.split('_')[0])))
                    level_numbers.append(level_num)
                except:
                    continue
                    
        level_numbers.sort()
        
        # Create progression ratios between consecutive levels
        for i in range(len(level_numbers) - 1):
            current_level = level_numbers[i]
            next_level = level_numbers[i + 1]
            
            current_col = f'level_{current_level}_events'
            next_col = f'level_{next_level}_events'
            
            if current_col in merged.columns and next_col in merged.columns:
                merged[f'progression_ratio_{current_level}_{next_level}'] = (
                    merged[next_col] / (merged[current_col] + 1e-8)
                )
                
        # Add total events feature
        event_cols = [col for col in merged.columns if 'level_' in col and 'events' in col]
        if event_cols:
            merged['total_level_events'] = merged[event_cols].sum(axis=1)
            
        return merged
    
    def _add_cost_features(self, features: pd.DataFrame, cost_df: pd.DataFrame) -> pd.DataFrame:
        """Add cost and monetization features"""
        if cost_df.empty:
            logger.warning("No cost/revenue data available for feature engineering")
            return features
            
        # Merge cost data
        merge_cols = ['platform', 'subdirectory']
        available_cols = [col for col in merge_cols if col in cost_df.columns and col in features.columns]
        
        if not available_cols:
            logger.warning("No common columns found for cost data merge")
            return features
            
        merged = features.merge(cost_df, on=available_cols, how='left', suffixes=('', '_cost'))
        
        # Add cost features
        if 'spend' in merged.columns:
            merged['total_spend'] = merged['spend']
        elif 'cost' in merged.columns:
            merged['total_spend'] = merged['cost']
            
        if 'revenue' in merged.columns:
            merged['total_revenue'] = merged['revenue']
        elif 'ad_revenue' in merged.columns:
            merged['total_revenue'] = merged['ad_revenue']
            
        # Calculate monetization metrics
        if 'total_spend' in merged.columns and 'total_revenue' in merged.columns:
            merged['roi'] = merged['total_revenue'] / (merged['total_spend'] + 1e-8)
            merged['profit'] = merged['total_revenue'] - merged['total_spend']
            merged['profit_margin'] = merged['profit'] / (merged['total_revenue'] + 1e-8)
            
        # Add install-based metrics
        if 'installs' in merged.columns:
            merged['total_installs'] = merged['installs']
            if 'total_spend' in merged.columns:
                merged['cpi'] = merged['total_spend'] / (merged['total_installs'] + 1e-8)
            if 'total_revenue' in merged.columns:
                merged['arpu'] = merged['total_revenue'] / (merged['total_installs'] + 1e-8)
                
        return merged
    
    def _add_derived_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add derived features and interactions"""
        derived = features.copy()
        
        # Retention-based derived features
        if 'd1_retention' in derived.columns:
            derived['d1_retention_squared'] = derived['d1_retention'] ** 2
            derived['d1_retention_log'] = np.log(derived['d1_retention'] + 1e-8)
            
        if 'd3_retention' in derived.columns:
            derived['d3_retention_squared'] = derived['d3_retention'] ** 2
            derived['d3_retention_log'] = np.log(derived['d3_retention'] + 1e-8)
            
        # ROAS-based derived features
        if 'roas_d1' in derived.columns:
            derived['roas_d1_squared'] = derived['roas_d1'] ** 2
            derived['roas_d1_log'] = np.log(derived['roas_d1'] + 1e-8)
            
        if 'roas_d3' in derived.columns:
            derived['roas_d3_squared'] = derived['roas_d3'] ** 2
            derived['roas_d3_log'] = np.log(derived['roas_d3'] + 1e-8)
            
        # Interaction features
        if 'd1_retention' in derived.columns and 'roas_d1' in derived.columns:
            derived['retention_roas_interaction'] = derived['d1_retention'] * derived['roas_d1']
            
        if 'd3_retention' in derived.columns and 'roas_d3' in derived.columns:
            derived['retention_roas_interaction_d3'] = derived['d3_retention'] * derived['roas_d3']
            
        # Cost-based derived features
        if 'cpi' in derived.columns:
            derived['cpi_log'] = np.log(derived['cpi'] + 1e-8)
            
        if 'arpu' in derived.columns:
            derived['arpu_log'] = np.log(derived['arpu'] + 1e-8)
            
        return derived
    
    def _add_platform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add platform-specific features"""
        platform_features = features.copy()
        
        # Platform encoding
        if 'platform' in platform_features.columns:
            platform_dummies = pd.get_dummies(platform_features['platform'], prefix='platform')
            platform_features = pd.concat([platform_features, platform_dummies], axis=1)
            
        # Subdirectory encoding (Android/iOS)
        if 'subdirectory' in platform_features.columns:
            subdir_dummies = pd.get_dummies(platform_features['subdirectory'], prefix='subdir')
            platform_features = pd.concat([platform_features, subdir_dummies], axis=1)
            
        return platform_features
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features"""
        cleaned = features.copy()
        
        # Remove duplicate columns
        cleaned = cleaned.loc[:, ~cleaned.columns.duplicated()]
        
        # Handle infinite values
        numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            cleaned[col] = cleaned[col].replace([np.inf, -np.inf], np.nan)
            
        # Fill missing values with 0 for numeric columns
        for col in numeric_cols:
            cleaned[col] = cleaned[col].fillna(0)
            
        # Remove rows with all missing values
        cleaned = cleaned.dropna(how='all')
        
        # Ensure all numeric columns are float
        for col in numeric_cols:
            cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce').fillna(0)
        
        # Remove any remaining object columns that aren't properly encoded
        # Keep only numeric columns for model training
        object_cols = cleaned.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            logger.warning(f"Removing object columns that weren't properly encoded: {list(object_cols)}")
            cleaned = cleaned.drop(columns=object_cols)
            
        return cleaned
    
    def get_feature_importance(self, model) -> Dict[str, float]:
        """Extract feature importance from trained model"""
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(self.feature_columns, model.feature_importances_))
            self.feature_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            return self.feature_importance
        else:
            logger.warning("Model does not have feature_importances_ attribute")
            return {}
    
    def select_top_features(self, features: pd.DataFrame, n_features: int = 20) -> pd.DataFrame:
        """Select top features based on importance"""
        if not self.feature_importance:
            logger.warning("No feature importance available. Returning all features.")
            return features
            
        top_features = list(self.feature_importance.keys())[:n_features]
        available_features = [f for f in top_features if f in features.columns]
        
        # Always include platform and subdirectory columns
        base_cols = ['platform', 'subdirectory', 'data_type']
        available_base_cols = [col for col in base_cols if col in features.columns]
        
        selected_cols = available_base_cols + available_features
        return features[selected_cols]
    
    def get_feature_summary(self, features: pd.DataFrame) -> Dict:
        """Get summary of engineered features"""
        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = features.select_dtypes(include=['object']).columns.tolist()
        
        summary = {
            'total_features': len(features.columns),
            'numeric_features': len(numeric_cols),
            'categorical_features': len(categorical_cols),
            'total_samples': len(features),
            'feature_types': {
                'retention_features': len([col for col in features.columns if 'retention' in col.lower()]),
                'roas_features': len([col for col in features.columns if 'roas' in col.lower()]),
                'level_features': len([col for col in features.columns if 'level' in col.lower()]),
                'cost_features': len([col for col in features.columns if any(x in col.lower() for x in ['spend', 'cost', 'revenue', 'cpi', 'arpu'])]),
                'platform_features': len([col for col in features.columns if 'platform' in col.lower() or 'subdir' in col.lower()])
            }
        }
        
        return summary
