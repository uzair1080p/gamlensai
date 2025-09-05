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
    """Memory-efficient ROAS forecasting model for GameLens AI - optimized for low-RAM servers"""
    
    def __init__(self, target_day: int = 30):
        self.target_day = target_day
        self.models = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        self.training_chunk_size = 1000  # Process data in chunks
        
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   quantiles: List[float] = [0.1, 0.5, 0.9],
                   n_estimators: int = 50,  # Reduced default
                   learning_rate: float = 0.1,  # Higher learning rate for faster convergence
                   max_depth: int = 4,  # Reduced depth to save memory
                   random_state: int = 42) -> Dict[str, any]:
        """Train ML models with memory optimization"""
        logger.info("Starting memory-efficient model training...")
        
        # Ensure only numeric features are used
        X_numeric = X.select_dtypes(include=[np.number])
        if len(X_numeric.columns) != len(X.columns):
            object_cols = X.select_dtypes(include=['object']).columns
            logger.warning(f"Removing object columns for training: {list(object_cols)}")
            X = X_numeric
        
        # Check if we need to use chunked training
        if len(X) > self.training_chunk_size:
            logger.info(f"Large dataset detected ({len(X)} samples). Using chunked training.")
            return self._train_model_chunked(X, y, quantiles, n_estimators, learning_rate, max_depth, random_state)
        else:
            logger.info(f"Small dataset ({len(X)} samples). Using standard training.")
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
                    'subsample': 0.7,  # Reduced from 0.8
                    'colsample_bytree': 0.8,  # Reduced from 0.9
                    'random_state': random_state,
                    'verbosity': 0,
                    'tree_method': 'hist',  # Memory efficient
                    'grow_policy': 'depthwise',  # Memory efficient
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
    
    def _train_model_chunked(self, X: pd.DataFrame, y: pd.Series, 
                            quantiles: List[float], n_estimators: int,
                            learning_rate: float, max_depth: int, 
                            random_state: int) -> Dict[str, any]:
        """Chunked training for large datasets"""
        logger.info(f"Training with chunked approach (chunk size: {self.training_chunk_size})")
        
        # Split data into chunks
        n_chunks = (len(X) + self.training_chunk_size - 1) // self.training_chunk_size
        logger.info(f"Processing {len(X)} samples in {n_chunks} chunks")
        
        models = {}
        
        if LIGHTGBM_AVAILABLE:
            # Train models using chunked approach
            for q in quantiles:
                logger.info(f"Training LightGBM model for quantile {q} using chunked data")
                
                # Use more conservative parameters for chunked training
                params = {
                    'objective': 'quantile',
                    'alpha': q,
                    'metric': 'quantile',
                    'boosting_type': 'gbdt',
                    'num_leaves': 10,  # Further reduced for chunked training
                    'learning_rate': learning_rate * 1.5,  # Higher learning rate
                    'n_estimators': n_estimators // 2,  # Fewer estimators
                    'max_depth': max_depth - 1,  # Shallower trees
                    'feature_fraction': 0.7,
                    'bagging_fraction': 0.6,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'random_state': random_state,
                    'force_col_wise': True,
                    'min_data_in_leaf': 30,
                }
                
                model = lgb.LGBMRegressor(**params)
                
                # Train on chunks
                for i in range(0, len(X), self.training_chunk_size):
                    end_idx = min(i + self.training_chunk_size, len(X))
                    X_chunk = X.iloc[i:end_idx]
                    y_chunk = y.iloc[i:end_idx]
                    
                    logger.info(f"Training on chunk {i//self.training_chunk_size + 1}/{n_chunks} ({len(X_chunk)} samples)")
                    model.fit(X_chunk, y_chunk)
                    
                    # Clear chunk data
                    del X_chunk, y_chunk
                    gc.collect()
                
                models[f'q{q}'] = model
                gc.collect()
                
        elif XGBOOST_AVAILABLE:
            # Train models using chunked approach
            for q in quantiles:
                logger.info(f"Training XGBoost model for quantile {q} using chunked data")
                
                params = {
                    'objective': 'reg:quantileerror',
                    'quantile_alpha': q,
                    'learning_rate': learning_rate * 1.5,
                    'n_estimators': n_estimators // 2,
                    'max_depth': max_depth - 1,
                    'subsample': 0.6,
                    'colsample_bytree': 0.7,
                    'random_state': random_state,
                    'verbosity': 0,
                    'tree_method': 'hist',
                    'grow_policy': 'depthwise',
                }
                
                model = xgb.XGBRegressor(**params)
                
                # Train on chunks
                for i in range(0, len(X), self.training_chunk_size):
                    end_idx = min(i + self.training_chunk_size, len(X))
                    X_chunk = X.iloc[i:end_idx]
                    y_chunk = y.iloc[i:end_idx]
                    
                    logger.info(f"Training on chunk {i//self.training_chunk_size + 1}/{n_chunks} ({len(X_chunk)} samples)")
                    model.fit(X_chunk, y_chunk)
                    
                    # Clear chunk data
                    del X_chunk, y_chunk
                    gc.collect()
                
                models[f'q{q}'] = model
                gc.collect()
        else:
            raise ImportError("Neither LightGBM nor XGBoost is available for training")
            
        self.models = models
        
        # Store feature importance from median model
        if 'q0.5' in models:
            self.feature_importance = dict(zip(X.columns, models['q0.5'].feature_importances_))
            
        return models
    
    def predict_with_confidence(self, X: pd.DataFrame) -> pd.DataFrame:
        """Make predictions with confidence intervals using chunked approach if needed"""
        
        if not self.models:
            raise ValueError("Models not trained. Call train_model() first.")
        
        # Ensure only numeric features are used (same as training)
        X_numeric = X.select_dtypes(include=[np.number])
        if len(X_numeric.columns) != len(X.columns):
            X = X_numeric
        
        # Check if we need chunked prediction
        if len(X) > self.training_chunk_size:
            logger.info(f"Large prediction set ({len(X)} samples). Using chunked prediction.")
            return self._predict_chunked(X)
        else:
            return self._predict_standard(X)
    
    def _predict_standard(self, X: pd.DataFrame) -> pd.DataFrame:
        """Standard prediction for smaller datasets"""
        predictions = {}
        
        for quantile, model in self.models.items():
            pred = model.predict(X)
            predictions[f'roas_pred_{quantile}'] = pred
        
        # Create result DataFrame
        result = pd.DataFrame(predictions)
        result['roas_prediction'] = result['roas_pred_q0.5']  # Median as main prediction
        result['roas_lower_bound'] = result['roas_pred_q0.1']  # 10th percentile
        result['roas_upper_bound'] = result['roas_pred_q0.9']  # 90th percentile
        result['confidence_interval'] = result['roas_upper_bound'] - result['roas_lower_bound']
        
        return result
    
    def _predict_chunked(self, X: pd.DataFrame) -> pd.DataFrame:
        """Chunked prediction for large datasets"""
        logger.info(f"Making predictions in chunks (chunk size: {self.training_chunk_size})")
        
        all_predictions = []
        n_chunks = (len(X) + self.training_chunk_size - 1) // self.training_chunk_size
        
        for i in range(0, len(X), self.training_chunk_size):
            end_idx = min(i + self.training_chunk_size, len(X))
            X_chunk = X.iloc[i:end_idx]
            
            logger.info(f"Predicting on chunk {i//self.training_chunk_size + 1}/{n_chunks} ({len(X_chunk)} samples)")
            
            chunk_predictions = self._predict_standard(X_chunk)
            all_predictions.append(chunk_predictions)
            
            # Clear chunk data
            del X_chunk, chunk_predictions
            gc.collect()
        
        # Combine all predictions
        result = pd.concat(all_predictions, ignore_index=True)
        del all_predictions
        gc.collect()
        
        return result
    
    def evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance with memory optimization"""
        logger.info("Evaluating model performance...")
        
        try:
            # Use smaller test set for evaluation to save memory
            if len(X) > 2000:
                logger.info("Large dataset detected. Using subset for evaluation.")
                X_eval, _, y_eval, _ = train_test_split(X, y, test_size=0.8, random_state=42)
            else:
                X_eval, y_eval = X, y
            
            # Make predictions
            predictions = self.predict_with_confidence(X_eval)
            y_pred = predictions['roas_prediction']
            
            # Calculate metrics
            mse = mean_squared_error(y_eval, y_pred)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(y_eval, y_pred)
            
            # R-squared
            ss_res = np.sum((y_eval - y_pred) ** 2)
            ss_tot = np.sum((y_eval - np.mean(y_eval)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Confidence interval coverage
            lower_bound = predictions['roas_lower_bound']
            upper_bound = predictions['roas_upper_bound']
            coverage = np.mean((y_eval >= lower_bound) & (y_eval <= upper_bound))
            
            metrics = {
                'r2': r2,
                'rmse': rmse,
                'mape': mape,
                'confidence_coverage': coverage
            }
            
            self.performance_metrics = metrics
            
            # Clear evaluation data
            del X_eval, y_eval, predictions, y_pred
            gc.collect()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {'r2': 0, 'rmse': float('inf'), 'mape': float('inf'), 'confidence_coverage': 0}
    
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
