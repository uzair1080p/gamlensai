"""
SQLAlchemy models for GameLens AI
"""

import uuid
from datetime import datetime, date
from enum import Enum
from typing import List, Optional, Dict, Any

from sqlalchemy import (
    Column, String, Integer, DateTime, Date, Text, JSON, 
    ForeignKey, UniqueConstraint, Index, Boolean
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy import JSON as SQLAlchemyJSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class PlatformEnum(str, Enum):
    """Supported advertising platforms"""
    UNITY_ADS = "unity_ads"
    MISTPLAY = "mistplay"
    FACEBOOK = "facebook"
    GOOGLE = "google"
    TIKTOK = "tiktok"
    UNKNOWN = "unknown"


class Dataset(Base):
    """Dataset table for storing ingested data metadata"""
    __tablename__ = "dataset"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    raw_filename = Column(String(255), nullable=False)
    source_platform = Column(String(50), nullable=False)  # PlatformEnum
    channel = Column(String(50), nullable=True)  # android/ios/web
    game = Column(String(255), nullable=False)
    countries = Column(JSON, nullable=True)  # List of countries or "MULTI"
    records = Column(Integer, nullable=False, default=0)
    
    # Timestamps
    ingest_started_at = Column(DateTime(timezone=True), server_default=func.now())
    ingest_completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Date range
    data_start_date = Column(Date, nullable=True)
    data_end_date = Column(Date, nullable=False, default=date.today)
    
    # Storage and metadata
    storage_path = Column(String(500), nullable=True)
    canonical_name = Column(String(255), nullable=True, unique=True)
    schema_fingerprint = Column(String(64), nullable=True)  # Hash of columns+dtypes
    notes = Column(Text, nullable=True)
    
    # Relationships
    prediction_runs = relationship("PredictionRun", back_populates="dataset")
    
    # Indexes
    __table_args__ = (
        Index('idx_dataset_platform', 'source_platform'),
        Index('idx_dataset_game', 'game'),
        Index('idx_dataset_canonical_name', 'canonical_name'),
        Index('idx_dataset_ingest_date', 'ingest_started_at'),
    )


class ModelVersion(Base):
    """Model version table for tracking trained models"""
    __tablename__ = "model_version"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(255), nullable=False)  # e.g., lgbm_roas_d30_quantile
    version = Column(Integer, nullable=False, default=1)
    target_day = Column(Integer, nullable=False)  # 15/30/45/90 etc.
    
    # Training data
    train_dataset_ids = Column(JSON, nullable=False)  # Array of UUIDs
    feature_set_fingerprint = Column(String(64), nullable=True)
    train_end_date = Column(Date, nullable=False, default=date.today)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(String(100), nullable=False, default="system")
    
    # Model artifacts and metrics
    metrics_json = Column(JSON, nullable=True)  # R2, MAPE, RMSE, MAE
    artifact_path = Column(String(500), nullable=True)
    params_json = Column(JSON, nullable=True)
    notes = Column(Text, nullable=True)
    
    # Relationships
    prediction_runs = relationship("PredictionRun", back_populates="model_version")
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint('model_name', 'version', name='uq_model_name_version'),
        Index('idx_model_name', 'model_name'),
        Index('idx_model_target_day', 'target_day'),
        Index('idx_model_created_at', 'created_at'),
    )


class PredictionRun(Base):
    """Prediction run table for tracking prediction executions"""
    __tablename__ = "prediction_run"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_version_id = Column(UUID(as_uuid=True), ForeignKey('model_version.id'), nullable=False)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey('dataset.id'), nullable=False)
    
    # Execution tracking
    requested_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    n_rows = Column(Integer, nullable=False, default=0)
    
    # Prediction configuration and results
    targets = Column(JSON, nullable=True)  # Horizons requested [30, 45, 90]
    output_path = Column(String(500), nullable=True)
    summary_json = Column(JSON, nullable=True)  # Aggregates & recommendation summary
    
    # Relationships
    model_version = relationship("ModelVersion", back_populates="prediction_runs")
    dataset = relationship("Dataset", back_populates="prediction_runs")
    
    # Indexes
    __table_args__ = (
        Index('idx_prediction_model', 'model_version_id'),
        Index('idx_prediction_dataset', 'dataset_id'),
        Index('idx_prediction_requested', 'requested_at'),
    )
