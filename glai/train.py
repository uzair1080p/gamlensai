"""
Training and model registry system for GameLens AI
"""

import os
import json
import uuid
import hashlib
import re
import pandas as pd
import numpy as np
from datetime import date
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

from sqlalchemy.orm import Session
from .db import get_db_session
from .models import ModelVersion, Dataset
from .ingest import load_dataset_data
from .naming import generate_model_name, generate_artifact_path


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from the dataset for training
    
    Args:
        df: Input dataframe with normalized data
        
    Returns:
        DataFrame with engineered features
    """
    features_df = df.copy()

    # Normalize ROAS/Retention column labels to canonical lowercase form
    # Accept variants like ROAS_D30, roas d30, ROAS_d 30, etc.
    import re as _re
    rename_map: dict = {}
    for col in list(features_df.columns):
        col_str = str(col).strip()
        lc = col_str.lower()
        m = _re.match(r"^roas[_\s]*d\s*(\d+)$", lc)
        if m:
            day = m.group(1)
            rename_map[col] = f"roas_d{day}"
            continue
        m2 = _re.match(r"^retention[_\s]*d\s*(\d+)$", lc)
        if m2:
            day = m2.group(1)
            rename_map[col] = f"retention_d{day}"
            continue
    if rename_map:
        features_df = features_df.rename(columns=rename_map)
    
    # Basic features
    if 'cost' in features_df.columns and 'installs' in features_df.columns:
        features_df['cpi'] = features_df['cost'] / features_df['installs'].replace(0, 1)
        features_df['cost_per_install'] = features_df['cpi']
    
    if 'revenue' in features_df.columns and 'cost' in features_df.columns:
        features_df['roas'] = features_df['revenue'] / features_df['cost'].replace(0, 1)
    
    # ROAS progression features
    roas_cols = [col for col in features_df.columns if col.startswith('roas_d')]
    for i, col in enumerate(roas_cols):
        if i > 0:
            prev_col = roas_cols[i-1]
            if prev_col in features_df.columns:
                features_df[f'{col}_vs_{prev_col}'] = features_df[col] - features_df[prev_col]
    
    # Retention features
    retention_cols = [col for col in features_df.columns if col.startswith('retention_d')]
    for col in retention_cols:
        if col in features_df.columns:
            features_df[f'{col}_rate'] = features_df[col] / 100.0  # Convert percentage to rate
    
    # Date features
    if 'date' in features_df.columns:
        features_df['date'] = pd.to_datetime(features_df['date'])
        features_df['day_of_week'] = features_df['date'].dt.dayofweek
        features_df['month'] = features_df['date'].dt.month
        features_df['quarter'] = features_df['date'].dt.quarter
    
    # Platform and channel features (one-hot encoding)
    if 'platform' in features_df.columns:
        platform_dummies = pd.get_dummies(features_df['platform'], prefix='platform')
        features_df = pd.concat([features_df, platform_dummies], axis=1)
    
    if 'channel' in features_df.columns:
        channel_dummies = pd.get_dummies(features_df['channel'], prefix='channel')
        features_df = pd.concat([features_df, channel_dummies], axis=1)
    
    # Country features
    if 'country' in features_df.columns:
        # Top countries only to avoid too many features
        top_countries = features_df['country'].value_counts().head(10).index
        for country in top_countries:
            features_df[f'country_{country}'] = (features_df['country'] == country).astype(int)
    
    return features_df


def prepare_training_data(features_df: pd.DataFrame, target_day: int) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare training data with features and target
    
    Args:
        features_df: DataFrame with features
        target_day: Target day for prediction
        
    Returns:
        Tuple of (X, y) for training
    """
    target_col = f'roas_d{target_day}'
    
    if target_col not in features_df.columns:
        raise ValueError(f"Target column {target_col} not found in data")
    
    # Remove rows with missing target
    valid_data = features_df.dropna(subset=[target_col])
    
    if len(valid_data) == 0:
        raise ValueError(f"No valid data found for target {target_col}")
    
    # Exclude future signals and metadata columns
    exclude_cols = [
        'date', 'platform', 'channel', 'country', 'game', 'countries',
        target_col
    ]
    
    # Remove future ROAS columns (only use earlier days)
    future_roas_cols = [col for col in features_df.columns 
                       if col.startswith('roas_d') and 
                       int(re.search(r'roas_d(\d+)', col).group(1)) >= target_day]
    
    exclude_cols.extend(future_roas_cols)
    
    # Select feature columns
    feature_cols = [col for col in valid_data.columns if col not in exclude_cols]
    
    # Remove non-numeric columns
    numeric_cols = valid_data[feature_cols].select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in feature_cols if col in numeric_cols]
    
    X = valid_data[feature_cols].fillna(0)
    y = valid_data[target_col]
    
    return X, y


def train_lgbm_quantile(
    dataset_ids: List[str], 
    target_day: int, 
    params: Dict[str, Any],
    notes: Optional[str] = None
) -> ModelVersion:
    """
    Train LightGBM quantile regression models
    
    Args:
        dataset_ids: List of dataset UUIDs to use for training
        target_day: Target day for prediction (15, 30, 45, 90)
        params: Model parameters
        notes: Optional notes about the model
        
    Returns:
        ModelVersion object
    """
    # Load and combine datasets
    all_data = []
    max_end_date = None
    
    for dataset_id in dataset_ids:
        dataset = get_dataset_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        df = load_dataset_data(dataset)
        all_data.append(df)
        
        if dataset.data_end_date:
            if max_end_date is None or dataset.data_end_date > max_end_date:
                max_end_date = dataset.data_end_date
    
    if not all_data:
        raise ValueError("No valid datasets found")
    
    # Combine all datasets
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Create features
    features_df = create_features(combined_df)
    
    # Prepare training data
    X, y = prepare_training_data(features_df, target_day)
    
    if len(X) == 0:
        raise ValueError("No training data available after feature engineering")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train quantile models
    quantiles = [0.1, 0.5, 0.9]  # p10, p50, p90
    models = {}
    
    for q in quantiles:
        model_params = {
            'objective': 'quantile',
            'alpha': q,
            'metric': 'quantile',
            'boosting_type': 'gbdt',
            'num_leaves': params.get('num_leaves', 31),
            'learning_rate': params.get('learning_rate', 0.05),
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        model = lgb.LGBMRegressor(**model_params)
        model.fit(X_train, y_train)
        models[f'p{int(q*100)}'] = model
    
    # Evaluate models
    predictions = {}
    metrics = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
        
        metrics[name] = {
            'r2': float(r2),
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape)
        }
    
    # Get feature importance
    feature_importance = models['p50'].feature_importances_
    feature_names = X.columns.tolist()
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # Generate model name and version
    platform = "multi"  # Could be determined from datasets
    model_name = generate_model_name(target_day, platform)
    
    # Get next version number
    db = get_db_session()
    try:
        existing_versions = db.query(ModelVersion).filter(
            ModelVersion.model_name == model_name
        ).order_by(ModelVersion.version.desc()).all()
        
        next_version = 1
        if existing_versions:
            next_version = existing_versions[0].version + 1
        
        # Create artifact directory
        artifact_path = generate_artifact_path(model_name, next_version)
        artifact_dir = Path(artifact_path)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for name, model in models.items():
            model_file = artifact_dir / f"model_{name}.pkl"
            joblib.dump(model, model_file)
        
        # Save metadata
        metadata = {
            'feature_list': feature_names,
            'params': params,
            'metrics': metrics,
            'feature_importance': importance_df.to_dict('records')
        }
        
        with open(artifact_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save feature importance
        importance_df.to_csv(artifact_dir / "feature_importance.csv", index=False)
        
        # Compute feature set fingerprint
        feature_set_fingerprint = hashlib.md5(
            str(sorted(feature_names)).encode()
        ).hexdigest()
        
        # Create model version record
        model_version = ModelVersion(
            model_name=model_name,
            version=next_version,
            target_day=target_day,
            train_dataset_ids=dataset_ids,
            feature_set_fingerprint=feature_set_fingerprint,
            train_end_date=max_end_date or date.today(),
            metrics_json=metrics,
            artifact_path=artifact_path,
            params_json=params,
            notes=notes
        )
        
        db.add(model_version)
        db.commit()
        db.refresh(model_version)
        
        return model_version
        
    except Exception as e:
        db.rollback()
        raise e
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
    db = get_db_session()
    try:
        # Convert string to UUID if needed
        if isinstance(dataset_id, str):
            dataset_id = uuid.UUID(dataset_id)
        return db.query(Dataset).filter(Dataset.id == dataset_id).first()
    finally:
        db.close()


def get_model_versions(model_name: Optional[str] = None) -> List[ModelVersion]:
    """
    Get model versions with optional filtering
    
    Args:
        model_name: Filter by model name
        
    Returns:
        List of ModelVersion objects
    """
    db = get_db_session()
    try:
        query = db.query(ModelVersion)
        
        if model_name:
            query = query.filter(ModelVersion.model_name == model_name)
        
        return query.order_by(ModelVersion.created_at.desc()).all()
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
        return db.query(ModelVersion).filter(ModelVersion.id == model_version_id).first()
    finally:
        db.close()


def load_model_artifacts(model_version: ModelVersion) -> Dict[str, Any]:
    """
    Load model artifacts from storage
    
    Args:
        model_version: ModelVersion object
        
    Returns:
        Dictionary with loaded models and metadata
    """
    artifact_dir = Path(model_version.artifact_path)
    
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Artifact directory not found: {artifact_dir}")
    
    # Load models
    models = {}
    for model_file in artifact_dir.glob("model_*.pkl"):
        model_name = model_file.stem.replace("model_", "")
        models[model_name] = joblib.load(model_file)
    
    # Load metadata
    metadata_file = artifact_dir / "metadata.json"
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    # Load feature importance
    importance_file = artifact_dir / "feature_importance.csv"
    feature_importance = None
    if importance_file.exists():
        feature_importance = pd.read_csv(importance_file)
    
    return {
        'models': models,
        'metadata': metadata,
        'feature_importance': feature_importance
    }
