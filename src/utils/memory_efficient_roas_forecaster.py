import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import joblib
import logging
import gc

# Try to import LightGBM, fallback to XGBoost if it fails
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ LightGBM imported successfully")
except Exception as e:
    LIGHTGBM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ LightGBM not available: {e}")
    logger.info("🔄 Falling back to XGBoost for ROAS forecasting")
    
    try:
        import xgboost as xgb
        XGBOOST_AVAILABLE = True
        logger.info("✅ XGBoost imported successfully as fallback")
    except Exception as e2:
        XGBOOST_AVAILABLE = False
        logger.error(f"❌ Neither LightGBM nor XGBoost available: {e2}")
        raise ImportError("Neither LightGBM nor XGBoost could be imported. Please install one of them.")

logger = logging.getLogger(__name__)

class MemoryEfficientROASForecaster:
    """ROAS forecasting model for GameLens AI - optimized for 32GB RAM servers with full dataset training"""
    
    def __init__(self, target_day: int = 30):
        self.target_day = target_day
        self.models = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   quantiles: List[float] = [0.1, 0.5, 0.9],
                   n_estimators: int = 50,  # Reduced default
                   learning_rate: float = 0.1,  # Higher learning rate for faster convergence
                   max_depth: int = 4,  # Reduced depth to save memory
                   random_state: int = 42) -> Dict[str, any]:
        """Train ML models with memory optimization"""
        logger.info("Starting memory-efficient model training...")
        logger.info(f"Target variable stats: mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}")
        logger.info(f"Features: {len(X.columns)} columns, samples: {len(X)}")
        
        # Validate data quality
        if y.std() == 0:
            logger.warning("Target variable has zero variance - this will cause poor model performance")
        if X.isnull().sum().sum() > 0:
            logger.warning(f"Found {X.isnull().sum().sum()} missing values in features")
        if y.isnull().sum() > 0:
            logger.warning(f"Found {y.isnull().sum()} missing values in target")
        
        # Ensure only numeric features are used
        X_numeric = X.select_dtypes(include=[np.number])
        if len(X_numeric.columns) != len(X.columns):
            object_cols = X.select_dtypes(include=['object']).columns
            logger.warning(f"Removing object columns for training: {list(object_cols)}")
            X = X_numeric
        
        # Use standard training for all datasets with 32GB RAM - no chunking needed
        logger.info(f"Training on full dataset ({len(X)} samples) with 32GB RAM - no chunking required.")
        return self._train_model_standard(X, y, quantiles, n_estimators, learning_rate, max_depth, random_state)
    
    def _train_model_standard(self, X: pd.DataFrame, y: pd.Series, 
                             quantiles: List[float], n_estimators: int,
                             learning_rate: float, max_depth: int, 
                             random_state: int) -> Dict[str, any]:
        """Standard training for smaller datasets"""
        models = {}
        
        if LIGHTGBM_AVAILABLE:
            # Use LightGBM with memory-efficient parameters
            for q in quantiles:
                logger.info(f"Training LightGBM model for quantile {q}")
                
                params = {
                    'objective': 'quantile',
                    'alpha': q,
                    'metric': 'quantile',
                    'boosting_type': 'gbdt',
                    'num_leaves': 15,  # Reduced from 31
                    'learning_rate': learning_rate,
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'feature_fraction': 0.8,  # Reduced from 0.9
                    'bagging_fraction': 0.7,  # Reduced from 0.8
                    'bagging_freq': 5,
                    'verbose': -1,
                    'random_state': random_state,
                    'force_col_wise': True,  # Memory optimization
                    'min_data_in_leaf': 20,  # Prevent overfitting
                }
                
                model = lgb.LGBMRegressor(**params)
                model.fit(X, y)
                models[f'q{q}'] = model
                
                # Force garbage collection after each model
                gc.collect()
                
        elif XGBOOST_AVAILABLE:
            # Use XGBoost with memory-efficient parameters
            for q in quantiles:
                logger.info(f"Training XGBoost model for quantile {q}")
                
                params = {
                    'objective': 'reg:quantileerror',
                    'quantile_alpha': q,
                    'learning_rate': learning_rate,
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'subsample': 0.8,  # Increased for better performance
                    'colsample_bytree': 0.9,  # Increased for better performance
                    'random_state': random_state,
                    'verbosity': 0,
                    'tree_method': 'hist',  # Memory efficient
                    'grow_policy': 'depthwise',  # Memory efficient
                    'min_child_weight': 1,  # Added for better performance
                    'reg_alpha': 0.1,  # L1 regularization
                    'reg_lambda': 1.0,  # L2 regularization
                }
                
                model = xgb.XGBRegressor(**params)
                model.fit(X, y)
                models[f'q{q}'] = model
                
                # Force garbage collection after each model
                gc.collect()
        else:
            raise ImportError("Neither LightGBM nor XGBoost is available for training")
            
        self.models = models
        
        # Store feature importance from median model
        if 'q0.5' in models:
            self.feature_importance = dict(zip(X.columns, models['q0.5'].feature_importances_))
            
        return models
    
    
    
    def predict_with_confidence(self, X: pd.DataFrame) -> pd.DataFrame:
        """Make predictions with confidence intervals on dataset"""
        
        if not self.models:
            raise ValueError("Models not trained. Call train_model() first.")
        
        # Ensure only numeric features are used (same as training)
        X_numeric = X.select_dtypes(include=[np.number])
        if len(X_numeric.columns) != len(X.columns):
            X = X_numeric
        
        # Limit predictions for very large datasets to prevent hanging
        max_prediction_samples = 5000  # Even smaller limit to prevent WebSocket timeouts
        if len(X) > max_prediction_samples:
            logger.info(f"Large dataset detected ({len(X)} samples). Using subset for predictions ({max_prediction_samples} samples).")
            X = X.sample(n=max_prediction_samples, random_state=42)
        
        logger.info(f"Making predictions on dataset ({len(X)} samples) with 32GB RAM.")
        return self._predict_standard(X)
    
    def _predict_standard(self, X: pd.DataFrame) -> pd.DataFrame:
        """Standard prediction for smaller datasets with proper index handling"""
        predictions = {}
        
        for quantile, model in self.models.items():
            try:
                pred = model.predict(X)
                # Ensure predictions are finite and reasonable
                pred = np.where(np.isfinite(pred), pred, 0.0)
                pred = np.clip(pred, 0.0, 10.0)  # Clip to reasonable ROAS range
                predictions[f'roas_pred_{quantile}'] = pred
            except Exception as e:
                logger.error(f"Error predicting with quantile {quantile}: {e}")
                predictions[f'roas_pred_{quantile}'] = np.zeros(len(X))
        
        # Create result DataFrame with proper index from X
        result = pd.DataFrame(index=X.index)
        
        # Add predictions with proper index alignment
        if 'roas_pred_q0.5' in predictions:
            result['roas_prediction'] = predictions['roas_pred_q0.5']
        else:
            result['roas_prediction'] = np.zeros(len(X))
            
        if 'roas_pred_q0.1' in predictions:
            result['roas_lower_bound'] = predictions['roas_pred_q0.1']
        else:
            result['roas_lower_bound'] = np.zeros(len(X))
            
        if 'roas_pred_q0.9' in predictions:
            result['roas_upper_bound'] = predictions['roas_pred_q0.9']
        else:
            result['roas_upper_bound'] = np.zeros(len(X))
        
        # Calculate confidence interval
        result['confidence_interval'] = result['roas_upper_bound'] - result['roas_lower_bound']
        
        # Ensure all values are finite and reasonable
        for col in result.columns:
            result[col] = np.where(np.isfinite(result[col]), result[col], 0.0)
            result[col] = np.clip(result[col], 0.0, 10.0)
        
        return result
    
    
    def evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance on full dataset"""
        logger.info("Evaluating model performance...")
        
        try:
            # Use larger evaluation dataset for better metrics
            max_eval_samples = 10000  # Increased for better evaluation
            if len(X) > max_eval_samples:
                logger.info(f"Large dataset detected ({len(X)} samples). Using subset for evaluation ({max_eval_samples} samples).")
                X_eval, _, y_eval, _ = train_test_split(X, y, test_size=1-max_eval_samples/len(X), random_state=42)
            else:
                logger.info("Using full dataset for evaluation with 32GB RAM.")
                X_eval, y_eval = X, y
            
            # Make predictions
            predictions = self.predict_with_confidence(X_eval)
            y_pred = predictions['roas_prediction']
            
            # Ensure proper alignment by using common indices
            common_indices = y_eval.index.intersection(predictions.index)
            if len(common_indices) == 0:
                logger.error("No common indices between y_eval and predictions")
                return {'r2': 0, 'rmse': float('inf'), 'mape': float('inf'), 'mae': float('inf'), 'confidence_coverage': 0}
            
            y_eval_aligned = y_eval.loc[common_indices]
            y_pred_aligned = y_pred.loc[common_indices]
            
            # Remove any NaN or infinite values
            valid_mask = np.isfinite(y_eval_aligned) & np.isfinite(y_pred_aligned) & (y_eval_aligned > 0)
            y_eval_clean = y_eval_aligned[valid_mask]
            y_pred_clean = y_pred_aligned[valid_mask]
            
            if len(y_eval_clean) == 0:
                logger.error("No valid data points after cleaning")
                return {'r2': 0, 'rmse': float('inf'), 'mape': float('inf'), 'mae': float('inf'), 'confidence_coverage': 0}
            
            # Calculate metrics with clean data
            mse = mean_squared_error(y_eval_clean, y_pred_clean)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(y_eval_clean, y_pred_clean)
            mae = np.mean(np.abs(y_eval_clean - y_pred_clean))  # Mean Absolute Error
            
            # R-squared
            ss_res = np.sum((y_eval_clean - y_pred_clean) ** 2)
            ss_tot = np.sum((y_eval_clean - np.mean(y_eval_clean)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Confidence interval coverage
            lower_bound = predictions['roas_lower_bound'].loc[common_indices][valid_mask]
            upper_bound = predictions['roas_upper_bound'].loc[common_indices][valid_mask]
            coverage = np.mean((y_eval_clean >= lower_bound) & (y_eval_clean <= upper_bound))
            
            metrics = {
                'r2': r2,
                'rmse': rmse,
                'mape': mape,
                'mae': mae,
                'confidence_coverage': coverage
            }
            
            self.performance_metrics = metrics
            
            # Clear evaluation data
            del X_eval, y_eval, predictions, y_pred
            gc.collect()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {'r2': 0, 'rmse': float('inf'), 'mape': float('inf'), 'mae': float('inf'), 'confidence_coverage': 0}
    
    def generate_recommendations(self, X: pd.DataFrame, target_roas: float = 0.5, 
                                num_recommendations: int = 10, confidence_threshold: float = 0.8) -> pd.DataFrame:
        """Generate recommendations based on model predictions"""
        logger.info("Generating recommendations...")
        
        if not self.models:
            raise ValueError("Models not trained. Call train_model() first.")
        
        try:
            # Make predictions on the dataset
            predictions = self.predict_with_confidence(X)
            
            # Calculate confidence interval width
            predictions['confidence_width'] = predictions['roas_upper_bound'] - predictions['roas_lower_bound']
            
            # Filter by confidence threshold (narrower confidence intervals = higher confidence)
            max_confidence_width = predictions['confidence_width'].quantile(1 - confidence_threshold)
            high_confidence = predictions[predictions['confidence_width'] <= max_confidence_width].copy()
            
            # Calculate potential ROAS improvement
            high_confidence['roas_improvement'] = high_confidence['roas_prediction'] - target_roas
            
            # Sort by potential improvement (descending)
            recommendations = high_confidence.nlargest(num_recommendations, 'roas_improvement')
            
            # Add recommendation reasons
            recommendations['recommendation_reason'] = recommendations.apply(
                lambda row: f"Predicted ROAS {row['roas_prediction']:.3f} vs target {target_roas:.3f} (improvement: {row['roas_improvement']:.3f})", 
                axis=1
            )
            
            # Select relevant columns for recommendations
            result_columns = ['roas_prediction', 'roas_lower_bound', 'roas_upper_bound', 
                            'confidence_width', 'roas_improvement', 'recommendation_reason']
            
            # Only include columns that exist
            available_columns = [col for col in result_columns if col in recommendations.columns]
            recommendations_result = recommendations[available_columns].copy()
            
            # Add index as campaign identifier if available
            if hasattr(X, 'index'):
                recommendations_result.index.name = 'campaign_id'
            
            logger.info(f"Generated {len(recommendations_result)} recommendations")
            return recommendations_result
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['roas_prediction', 'roas_lower_bound', 'roas_upper_bound', 
                                       'confidence_width', 'roas_improvement', 'recommendation_reason'])
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Get feature importance as DataFrame"""
        if not self.feature_importance:
            return pd.DataFrame()
        
        # Sort by importance
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Take top N features
        top_features = sorted_features[:top_n]
        
        # Create DataFrame
        df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
        df['Importance'] = df['Importance'].round(4)
        
        return df
    
    def save_model(self, filepath: str):
        """Save trained models"""
        try:
            joblib.dump({
                'models': self.models,
                'feature_importance': self.feature_importance,
                'performance_metrics': self.performance_metrics,
                'target_day': self.target_day
            }, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load trained models"""
        try:
            data = joblib.load(filepath)
            self.models = data['models']
            self.feature_importance = data['feature_importance']
            self.performance_metrics = data['performance_metrics']
            self.target_day = data['target_day']
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
