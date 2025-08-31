import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import joblib
import logging

logger = logging.getLogger(__name__)

class GameLensROASForecaster:
    """ROAS forecasting model for GameLens AI - predicts D15/D30/D45/D90 ROAS from early data"""
    
    def __init__(self, target_day: int = 30):
        self.target_day = target_day
        self.models = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   quantiles: List[float] = [0.1, 0.5, 0.9],
                   n_estimators: int = 100,
                   learning_rate: float = 0.05,
                   max_depth: int = 6,
                   random_state: int = 42) -> Dict[str, lgb.LGBMRegressor]:
        """Train LightGBM models for different quantiles to get confidence intervals"""
        
        models = {}
        
        for q in quantiles:
            logger.info(f"Training model for quantile {q}")
            
            # LightGBM parameters optimized for quantile regression
            params = {
                'objective': 'quantile',
                'alpha': q,
                'metric': 'quantile',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': learning_rate,
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': random_state
            }
            
            # Train model
            model = lgb.LGBMRegressor(**params)
            model.fit(X, y)
            
            models[f'q{q}'] = model
            
        self.models = models
        
        # Store feature importance from median model
        if 'q0.5' in models:
            self.feature_importance = dict(zip(X.columns, models['q0.5'].feature_importances_))
            
        return models
    
    def predict_with_confidence(self, X: pd.DataFrame) -> pd.DataFrame:
        """Make predictions with confidence intervals"""
        
        if not self.models:
            raise ValueError("Models not trained. Call train_model() first.")
            
        predictions = {}
        
        # Get predictions from each quantile model
        for q_name, model in self.models.items():
            pred = model.predict(X)
            predictions[q_name] = pred
            
        # Create results dataframe
        results = pd.DataFrame(predictions)
        
        # Rename columns for clarity
        results.columns = [f'roas_pred_{col}' for col in results.columns]
        
        # Add point prediction (median)
        if 'roas_pred_q0.5' in results.columns:
            results['roas_prediction'] = results['roas_pred_q0.5']
            
        # Add confidence interval width
        if 'roas_pred_q0.1' in results.columns and 'roas_pred_q0.9' in results.columns:
            results['confidence_interval_width'] = results['roas_pred_q0.9'] - results['roas_pred_q0.1']
            
        return results
    
    def evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        
        if not self.models:
            raise ValueError("Models not trained. Call train_model() first.")
            
        # Get predictions
        predictions = self.predict_with_confidence(X)
        y_pred = predictions['roas_prediction']
        
        # Calculate metrics
        metrics = {
            'mape': mean_absolute_percentage_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': np.mean(np.abs(y - y_pred)),
            'r2': 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
        }
        
        # Calculate confidence interval coverage
        if 'roas_pred_q0.1' in predictions.columns and 'roas_pred_q0.9' in predictions.columns:
            coverage = np.mean((y >= predictions['roas_pred_q0.1']) & 
                             (y <= predictions['roas_pred_q0.9']))
            metrics['confidence_coverage'] = coverage
            
        self.performance_metrics = metrics
        
        return metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Dict[str, List[float]]:
        """Perform cross-validation"""
        
        if not self.models:
            raise ValueError("Models not trained. Call train_model() first.")
            
        # Use median model for CV
        median_model = self.models.get('q0.5')
        if not median_model:
            raise ValueError("Median model not found")
            
        # Cross-validation scores
        cv_scores = {
            'mape': cross_val_score(median_model, X, y, cv=cv_folds, 
                                  scoring='neg_mean_absolute_percentage_error'),
            'rmse': cross_val_score(median_model, X, y, cv=cv_folds, 
                                  scoring='neg_root_mean_squared_error')
        }
        
        # Convert to positive values
        cv_scores['mape'] = -cv_scores['mape']
        cv_scores['rmse'] = -cv_scores['rmse']
        
        return cv_scores
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance ranking"""
        
        if not self.feature_importance:
            raise ValueError("Feature importance not available. Train model first.")
            
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': list(self.feature_importance.keys()),
            'importance': list(self.feature_importance.values())
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_model(self, filepath: str):
        """Save trained models"""
        
        if not self.models:
            raise ValueError("No models to save. Train model first.")
            
        model_data = {
            'models': self.models,
            'feature_importance': self.feature_importance,
            'performance_metrics': self.performance_metrics,
            'target_day': self.target_day
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Models saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained models"""
        
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.feature_importance = model_data['feature_importance']
        self.performance_metrics = model_data['performance_metrics']
        self.target_day = model_data['target_day']
        
        logger.info(f"Models loaded from {filepath}")
    
    def generate_recommendations(self, X: pd.DataFrame, y_actual: Optional[pd.Series] = None,
                               target_roas: float = 1.0) -> pd.DataFrame:
        """Generate actionable recommendations based on predictions"""
        
        if not self.models:
            raise ValueError("Models not trained. Call train_model() first.")
            
        # Get predictions
        predictions = self.predict_with_confidence(X)
        
        # Create recommendations dataframe
        recommendations = pd.DataFrame()
        
        # Add predictions
        recommendations['predicted_roas'] = predictions['roas_prediction']
        recommendations['confidence_lower'] = predictions['roas_pred_q0.1']
        recommendations['confidence_upper'] = predictions['roas_pred_q0.9']
        
        # Add actual values if available
        if y_actual is not None:
            recommendations['actual_roas'] = y_actual
            recommendations['prediction_error'] = y_actual - predictions['roas_prediction']
        
        # Generate recommendations
        recommendations['roas_gap'] = target_roas - recommendations['predicted_roas']
        recommendations['recommendation'] = recommendations['roas_gap'].apply(self._get_recommendation)
        
        # Add confidence level
        recommendations['confidence_level'] = recommendations.apply(self._get_confidence_level, axis=1)
        
        return recommendations
    
    def _get_recommendation(self, roas_gap: float) -> str:
        """Get recommendation based on ROAS gap"""
        
        if roas_gap > 0.2:
            return "Scale aggressively - high potential"
        elif roas_gap > 0.1:
            return "Scale moderately - good potential"
        elif roas_gap > 0:
            return "Scale cautiously - marginal potential"
        elif roas_gap > -0.1:
            return "Maintain current spend"
        elif roas_gap > -0.2:
            return "Reduce spend - underperforming"
        else:
            return "Cut spend - poor performance"
    
    def _get_confidence_level(self, row: pd.Series) -> str:
        """Get confidence level based on prediction interval width"""
        
        ci_width = row['confidence_upper'] - row['confidence_lower']
        
        if ci_width < 0.1:
            return "High"
        elif ci_width < 0.2:
            return "Medium"
        else:
            return "Low"
