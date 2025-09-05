import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import gc

logger = logging.getLogger(__name__)

class MemoryEfficientFeatureEngineer:
    """Memory-efficient feature engineering for GameLens AI - optimized for low-RAM servers"""
    
    def __init__(self):
        self.feature_columns = []
        self.feature_importance = {}
        
    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create comprehensive feature set from all data types with memory optimization"""
        logger.info("Starting memory-efficient feature engineering process...")
        
        # Initialize with retention data as base
        if 'retention' not in data or data['retention'].empty:
            raise ValueError("Retention data is required for feature engineering")
            
        # Start with a copy of retention data
        features = data['retention'].copy()
        logger.info(f"Base features shape: {features.shape}")
        
        # Process each data type separately and merge incrementally
        try:
            # Add retention features (already in base)
            features = self._add_retention_features(features)
            logger.info(f"After retention features: {features.shape}")
            gc.collect()  # Force garbage collection
            
            # Add ROAS features
            if 'roas' in data and not data['roas'].empty:
                features = self._add_roas_features(features, data['roas'])
                logger.info(f"After ROAS features: {features.shape}")
                gc.collect()
            
            # Add level progression features
            if 'level_progression' in data and not data['level_progression'].empty:
                features = self._add_level_progression_features(features, data['level_progression'])
                logger.info(f"After level progression features: {features.shape}")
                gc.collect()
            
            # Add cost features
            if 'adspend_revenue' in data and not data['adspend_revenue'].empty:
                features = self._add_cost_features(features, data['adspend_revenue'])
                logger.info(f"After cost features: {features.shape}")
                gc.collect()
            
            # Add derived features
            features = self._add_derived_features(features)
            logger.info(f"After derived features: {features.shape}")
            gc.collect()
            
            # Final cleanup
            features = self._cleanup_features(features)
            logger.info(f"Final features shape: {features.shape}")
            
        except MemoryError as e:
            logger.error(f"Memory error during feature engineering: {e}")
            # Return minimal features if memory runs out
            features = self._create_minimal_features(data['retention'])
            logger.warning("Returning minimal feature set due to memory constraints")
        
        return features
    
    def _add_retention_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add retention-specific features"""
        try:
            # Basic retention features
            retention_cols = [col for col in features.columns if 'retention' in col.lower()]
            
            if retention_cols:
                # Early retention metrics
                if 'retention_d1' in features.columns:
                    features['early_retention_d1'] = features['retention_d1']
                
                if 'retention_d3' in features.columns:
                    features['early_retention_d3'] = features['retention_d3']
                    features['retention_d1_to_d3_ratio'] = features['retention_d1'] / (features['retention_d3'] + 1e-6)
                
                if 'retention_d7' in features.columns:
                    features['early_retention_d7'] = features['retention_d7']
                    features['retention_d3_to_d7_ratio'] = features['retention_d3'] / (features['retention_d7'] + 1e-6)
            
            return features
        except Exception as e:
            logger.warning(f"Error adding retention features: {e}")
            return features
    
    def _add_roas_features(self, features: pd.DataFrame, roas_data: pd.DataFrame) -> pd.DataFrame:
        """Add ROAS-specific features"""
        try:
            if roas_data.empty:
                return features
            
            # Merge on common columns
            merge_cols = ['platform', 'campaign_id', 'date'] if 'campaign_id' in roas_data.columns else ['platform', 'date']
            available_merge_cols = [col for col in merge_cols if col in features.columns and col in roas_data.columns]
            
            if available_merge_cols:
                # Select only essential ROAS columns to save memory
                roas_cols_to_merge = available_merge_cols + [col for col in roas_data.columns if 'roas_d' in col and col.endswith(('0', '1', '3', '7'))]
                roas_subset = roas_data[roas_cols_to_merge]
                
                features = features.merge(roas_subset, on=available_merge_cols, how='left', suffixes=('', '_roas'))
                
                # Add early ROAS features
                if 'roas_d0' in features.columns:
                    features['early_roas_d0'] = features['roas_d0']
                if 'roas_d1' in features.columns:
                    features['early_roas_d1'] = features['roas_d1']
                if 'roas_d3' in features.columns:
                    features['early_roas_d3'] = features['roas_d3']
            
            return features
        except Exception as e:
            logger.warning(f"Error adding ROAS features: {e}")
            return features
    
    def _add_level_progression_features(self, features: pd.DataFrame, level_data: pd.DataFrame) -> pd.DataFrame:
        """Add level progression features"""
        try:
            if level_data.empty:
                return features
            
            # Merge on common columns
            merge_cols = ['platform', 'campaign_id', 'date'] if 'campaign_id' in level_data.columns else ['platform', 'date']
            available_merge_cols = [col for col in merge_cols if col in features.columns and col in level_data.columns]
            
            if available_merge_cols:
                # Select only essential level columns to save memory
                level_cols_to_merge = available_merge_cols + [col for col in level_data.columns if 'level' in col.lower() and col.endswith(('1', '3', '7'))]
                level_subset = level_data[level_cols_to_merge]
                
                features = features.merge(level_subset, on=available_merge_cols, how='left', suffixes=('', '_level'))
                
                # Add early level features
                level_cols = [col for col in features.columns if 'level' in col.lower()]
                if level_cols:
                    features['avg_early_level'] = features[level_cols].mean(axis=1)
                    features['max_early_level'] = features[level_cols].max(axis=1)
            
            return features
        except Exception as e:
            logger.warning(f"Error adding level progression features: {e}")
            return features
    
    def _add_cost_features(self, features: pd.DataFrame, cost_data: pd.DataFrame) -> pd.DataFrame:
        """Add cost and revenue features"""
        try:
            if cost_data.empty:
                return features
            
            # Merge on common columns
            merge_cols = ['platform', 'campaign_id', 'date'] if 'campaign_id' in cost_data.columns else ['platform', 'date']
            available_merge_cols = [col for col in merge_cols if col in features.columns and col in cost_data.columns]
            
            if available_merge_cols:
                # Select only essential cost columns to save memory
                cost_cols_to_merge = available_merge_cols + [col for col in cost_data.columns if any(keyword in col.lower() for keyword in ['cost', 'revenue', 'spend', 'install'])]
                cost_subset = cost_data[cost_cols_to_merge]
                
                features = features.merge(cost_subset, on=available_merge_cols, how='left', suffixes=('', '_cost'))
                
                # Add cost efficiency features
                if 'cost' in features.columns and 'revenue' in features.columns:
                    features['cost_efficiency'] = features['revenue'] / (features['cost'] + 1e-6)
                if 'spend' in features.columns and 'revenue' in features.columns:
                    features['spend_efficiency'] = features['revenue'] / (features['spend'] + 1e-6)
            
            return features
        except Exception as e:
            logger.warning(f"Error adding cost features: {e}")
            return features
    
    def _add_derived_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add derived features"""
        try:
            # Platform-specific features
            if 'platform' in features.columns:
                features['is_android'] = (features['platform'] == 'Android').astype(int)
                features['is_ios'] = (features['platform'] == 'iOS').astype(int)
                features['is_unity_ads'] = features['platform'].str.contains('Unity', case=False, na=False).astype(int)
                features['is_mistplay'] = features['platform'].str.contains('Mistplay', case=False, na=False).astype(int)
            
            # Date features
            if 'date' in features.columns:
                features['date'] = pd.to_datetime(features['date'], errors='coerce')
                features['day_of_week'] = features['date'].dt.dayofweek
                features['month'] = features['date'].dt.month
                features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
            
            return features
        except Exception as e:
            logger.warning(f"Error adding derived features: {e}")
            return features
    
    def _cleanup_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean up features and remove unnecessary columns"""
        try:
            # Remove columns with all NaN values
            features = features.dropna(axis=1, how='all')
            
            # Fill remaining NaN values with 0 for numeric columns
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            features[numeric_cols] = features[numeric_cols].fillna(0)
            
            # Remove duplicate columns
            features = features.loc[:, ~features.columns.duplicated()]
            
            return features
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
            return features
    
    def _create_minimal_features(self, retention_data: pd.DataFrame) -> pd.DataFrame:
        """Create minimal feature set when memory is constrained"""
        try:
            features = retention_data.copy()
            
            # Add only the most essential features
            if 'platform' in features.columns:
                features['is_android'] = (features['platform'] == 'Android').astype(int)
                features['is_ios'] = (features['platform'] == 'iOS').astype(int)
            
            # Add basic retention features
            if 'retention_d1' in features.columns:
                features['early_retention_d1'] = features['retention_d1']
            if 'retention_d3' in features.columns:
                features['early_retention_d3'] = features['retention_d3']
            
            return features
        except Exception as e:
            logger.error(f"Error creating minimal features: {e}")
            return retention_data
