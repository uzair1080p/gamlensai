"""
Prediction system for GameLens AI
"""

import os
import json
import uuid
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from sqlalchemy.orm import Session
from .db import get_db_session
from .models import PredictionRun, ModelVersion, Dataset
from .ingest import load_dataset_data
from .train import create_features, load_model_artifacts


def run_predictions(
    model_version_id: str,
    dataset_id: str,
    targets: List[int] = [30, 45, 90]
) -> PredictionRun:
    """
    Run predictions using a trained model on a dataset
    
    Args:
        model_version_id: Model version UUID
        dataset_id: Dataset UUID
        targets: List of target days for prediction
        
    Returns:
        PredictionRun object
    """
    # Get model version and dataset
    model_version = get_model_version_by_id(model_version_id)
    if not model_version:
        raise ValueError(f"Model version {model_version_id} not found")
    
    dataset = get_dataset_by_id(dataset_id)
    if not dataset:
        raise ValueError(f"Dataset {dataset_id} not found")
    
    # Load model artifacts
    artifacts = load_model_artifacts(model_version)
    models = artifacts['models']
    metadata = artifacts['metadata']
    
    if not models:
        raise ValueError("No models found in artifacts")
    
    # Load dataset data
    df = load_dataset_data(dataset)
    
    # Create features (same as training)
    features_df = create_features(df)
    
    # Prepare features for prediction
    feature_list = metadata.get('feature_list', [])
    if not feature_list:
        raise ValueError("Feature list not found in model metadata")
    
    # Select only the features used in training
    available_features = [col for col in feature_list if col in features_df.columns]
    missing_features = [col for col in feature_list if col not in features_df.columns]
    
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
    
    # Create feature matrix
    X = features_df[available_features].fillna(0)
    
    # Add missing features with zeros
    for feature in missing_features:
        X[feature] = 0
    
    # Reorder columns to match training order
    X = X[feature_list]
    
    # Make predictions for each target day
    predictions = {}
    summary_data = []
    
    for target_day in targets:
        target_col = f'roas_d{target_day}'
        
        # Get predictions from quantile models
        pred_p10 = models.get('p10', models.get('p50'))  # Fallback to p50 if p10 not available
        pred_p50 = models.get('p50')
        pred_p90 = models.get('p90', models.get('p50'))  # Fallback to p50 if p90 not available
        
        if not pred_p50:
            print(f"Warning: No p50 model found for target day {target_day}")
            continue
        
        # Make predictions
        pred_10 = pred_p10.predict(X) if pred_p10 else pred_p50.predict(X)
        pred_50 = pred_p50.predict(X)
        pred_90 = pred_p90.predict(X) if pred_p90 else pred_p50.predict(X)
        
        # Store predictions
        predictions[target_col] = {
            'p10': pred_10,
            'p50': pred_50,
            'p90': pred_90
        }
        
        # Create summary for each row
        for i in range(len(X)):
            summary_data.append({
                'row_index': i,
                'target_day': target_day,
                'predicted_roas_p10': float(pred_10[i]),
                'predicted_roas_p50': float(pred_50[i]),
                'predicted_roas_p90': float(pred_90[i]),
                'confidence_interval': float(pred_90[i] - pred_10[i])
            })
    
    # Create prediction run record
    prediction_run = PredictionRun(
        model_version_id=uuid.UUID(model_version_id) if isinstance(model_version_id, str) else model_version_id,
        dataset_id=uuid.UUID(dataset_id) if isinstance(dataset_id, str) else dataset_id,
        n_rows=len(X),
        targets=targets
    )
    
    # Save to database
    db = get_db_session()
    try:
        db.add(prediction_run)
        db.commit()
        db.refresh(prediction_run)
        
        # Save predictions to file
        output_dir = Path(model_version.artifact_path) / "pred_runs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        prediction_run_id = str(prediction_run.id)
        output_path = output_dir / f"{prediction_run_id}.parquet"
        
        # Create prediction dataframe
        pred_df = pd.DataFrame(summary_data)
        
        # Add original data columns for context
        context_cols = ['date', 'platform', 'channel', 'country', 'game', 'cost', 'revenue', 'installs']
        for col in context_cols:
            if col in features_df.columns:
                pred_df[col] = features_df[col].iloc[pred_df['row_index']].values
        
        pred_df.to_parquet(output_path, index=False)
        
        # Update prediction run with output path
        prediction_run.output_path = str(output_path)
        
        # Generate summary statistics
        summary_json = generate_prediction_summary(pred_df, targets)
        prediction_run.summary_json = summary_json
        
        # Mark as completed
        prediction_run.completed_at = datetime.now()
        
        db.commit()
        db.refresh(prediction_run)
        
        return prediction_run
        
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


def generate_prediction_summary(pred_df: pd.DataFrame, targets: List[int]) -> Dict[str, Any]:
    """
    Generate summary statistics for predictions
    
    Args:
        pred_df: Prediction dataframe
        targets: List of target days
        
    Returns:
        Summary dictionary
    """
    summary = {}
    
    for target_day in targets:
        target_data = pred_df[pred_df['target_day'] == target_day]
        
        if len(target_data) == 0:
            continue
        
        # Basic statistics
        summary[f'd{target_day}'] = {
            'total_campaigns': len(target_data),
            'mean_predicted_roas': float(target_data['predicted_roas_p50'].mean()),
            'median_predicted_roas': float(target_data['predicted_roas_p50'].median()),
            'std_predicted_roas': float(target_data['predicted_roas_p50'].std()),
            'min_predicted_roas': float(target_data['predicted_roas_p50'].min()),
            'max_predicted_roas': float(target_data['predicted_roas_p50'].max()),
        }
        
        # ROAS thresholds
        for threshold in [0.5, 1.0, 1.5, 2.0]:
            above_threshold = (target_data['predicted_roas_p50'] >= threshold).sum()
            percentage = (above_threshold / len(target_data)) * 100
            summary[f'd{target_day}'][f'above_roas_{threshold}'] = {
                'count': int(above_threshold),
                'percentage': float(percentage)
            }
        
        # Confidence intervals
        summary[f'd{target_day}']['confidence_stats'] = {
            'mean_confidence_width': float(target_data['confidence_interval'].mean()),
            'median_confidence_width': float(target_data['confidence_interval'].median()),
        }
        
        # Recommendations
        recommendations = generate_recommendations(target_data)
        summary[f'd{target_day}']['recommendations'] = recommendations
    
    return summary


def generate_recommendations(pred_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate campaign recommendations based on predictions
    
    Args:
        pred_df: Prediction dataframe for a specific target day
        
    Returns:
        Recommendations dictionary
    """
    recommendations = {
        'scale': {'count': 0, 'spend_share': 0.0, 'campaigns': []},
        'maintain': {'count': 0, 'spend_share': 0.0, 'campaigns': []},
        'reduce': {'count': 0, 'spend_share': 0.0, 'campaigns': []},
        'cut': {'count': 0, 'spend_share': 0.0, 'campaigns': []}
    }
    
    total_spend = pred_df['cost'].sum() if 'cost' in pred_df.columns else len(pred_df)
    
    for _, row in pred_df.iterrows():
        pred_roas = row['predicted_roas_p50']
        confidence_width = row['confidence_interval']
        spend = row.get('cost', 1)
        
        # Determine recommendation based on predicted ROAS and confidence
        if pred_roas >= 1.5 and confidence_width < 0.5:
            action = 'scale'
        elif pred_roas >= 1.0 and confidence_width < 0.8:
            action = 'maintain'
        elif pred_roas >= 0.5:
            action = 'reduce'
        else:
            action = 'cut'
        
        recommendations[action]['count'] += 1
        recommendations[action]['spend_share'] += spend / total_spend
        recommendations[action]['campaigns'].append({
            'row_index': int(row['row_index']),
            'predicted_roas': float(pred_roas),
            'confidence_width': float(confidence_width),
            'spend': float(spend)
        })
    
    # Convert spend shares to percentages
    for action in recommendations:
        recommendations[action]['spend_share'] = float(recommendations[action]['spend_share'] * 100)
    
    return recommendations


def get_dataset_by_id(dataset_id: str) -> Optional[Dataset]:
    """
    Get dataset by ID
    
    Args:
        dataset_id: Dataset UUID
        
    Returns:
        Dataset object or None
    """
    db = get_db_session()
    try:
        # Convert string to UUID if needed
        if isinstance(dataset_id, str):
            dataset_id = uuid.UUID(dataset_id)
        return db.query(Dataset).filter(Dataset.id == dataset_id).first()
    finally:
        db.close()


def get_model_version_by_id(model_version_id: str) -> Optional[ModelVersion]:
    """
    Get model version by ID
    
    Args:
        model_version_id: Model version UUID
        
    Returns:
        ModelVersion object or None
    """
    db = get_db_session()
    try:
        # Convert string to UUID if needed
        if isinstance(model_version_id, str):
            model_version_id = uuid.UUID(model_version_id)
        return db.query(ModelVersion).filter(ModelVersion.id == model_version_id).first()
    finally:
        db.close()


def get_prediction_runs(
    model_version_id: Optional[str] = None,
    dataset_id: Optional[str] = None
) -> List[PredictionRun]:
    """
    Get prediction runs with optional filtering
    
    Args:
        model_version_id: Filter by model version
        dataset_id: Filter by dataset
        
    Returns:
        List of PredictionRun objects
    """
    import uuid as _uuid
    db = get_db_session()
    try:
        query = db.query(PredictionRun)
        
        if model_version_id:
            mv_id = model_version_id
            try:
                if isinstance(model_version_id, str):
                    mv_id = _uuid.UUID(model_version_id)
            except Exception:
                mv_id = model_version_id
            query = query.filter(PredictionRun.model_version_id == mv_id)
        
        if dataset_id:
            ds_id = dataset_id
            try:
                if isinstance(dataset_id, str):
                    ds_id = _uuid.UUID(dataset_id)
            except Exception:
                ds_id = dataset_id
            query = query.filter(PredictionRun.dataset_id == ds_id)
        
        return query.order_by(PredictionRun.requested_at.desc()).all()
    finally:
        db.close()


def load_predictions(prediction_run: PredictionRun) -> pd.DataFrame:
    """
    Load predictions from a prediction run
    
    Args:
        prediction_run: PredictionRun object
        
    Returns:
        DataFrame with predictions
    """
    if not prediction_run.output_path or not Path(prediction_run.output_path).exists():
        raise FileNotFoundError(f"Prediction file not found: {prediction_run.output_path}")
    
    return pd.read_parquet(prediction_run.output_path)
