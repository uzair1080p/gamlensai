"""
Integration tests for GameLens AI
"""

import pytest
import pandas as pd
import tempfile
import os
from datetime import date
from pathlib import Path

from glai.db import init_database, get_db_session
from glai.models import Dataset
from glai.ingest import ingest_file, get_datasets
from glai.naming import make_canonical_name


@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing"""
    data = {
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'platform': ['Unity Ads', 'Unity Ads', 'Unity Ads'],
        'channel': ['Android', 'Android', 'Android'],
        'game': ['Test Game', 'Test Game', 'Test Game'],
        'country': ['United States', 'United States', 'United States'],
        'installs': [100, 150, 200],
        'cost': [50.0, 75.0, 100.0],
        'revenue': [60.0, 90.0, 120.0],
        'roas_d0': [1.2, 1.2, 1.2],
        'roas_d1': [1.3, 1.3, 1.3],
        'roas_d3': [1.4, 1.4, 1.4],
        'roas_d7': [1.5, 1.5, 1.5],
        'roas_d30': [1.6, 1.6, 1.6]
    }
    return pd.DataFrame(data)


def test_database_initialization():
    """Test database initialization"""
    # This should not raise an exception
    init_database()
    
    # Test connection
    db = get_db_session()
    try:
        # Should be able to query
        datasets = db.query(Dataset).all()
        assert isinstance(datasets, list)
    finally:
        db.close()


def test_ingest_file_integration(sample_csv_data):
    """Test complete file ingestion workflow"""
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_csv_data.to_csv(f.name, index=False)
        temp_path = f.name
    
    try:
        # Ingest the file
        dataset = ingest_file(temp_path, notes="Integration test")
        
        # Verify dataset was created
        assert dataset is not None
        assert dataset.raw_filename == os.path.basename(temp_path)
        assert dataset.source_platform == "unity_ads"
        assert dataset.channel == "android"
        assert dataset.game == "Test Game"
        assert dataset.records == 3
        assert dataset.data_start_date == date(2024, 1, 1)
        assert dataset.data_end_date == date.today()  # Should be set to current date
        assert dataset.ingest_completed_at is not None
        assert dataset.storage_path is not None
        assert dataset.canonical_name is not None
        
        # Verify parquet file exists
        assert Path(dataset.storage_path).exists()
        
        # Verify data can be loaded
        from glai.ingest import load_dataset_data
        loaded_df = load_dataset_data(dataset)
        assert len(loaded_df) == 3
        assert 'date' in loaded_df.columns
        assert 'cost' in loaded_df.columns
        assert 'revenue' in loaded_df.columns
        
        # Verify dataset appears in database query
        datasets = get_datasets()
        assert len(datasets) >= 1
        assert any(d.id == dataset.id for d in datasets)
        
    finally:
        # Clean up
        os.unlink(temp_path)


def test_canonical_name_generation():
    """Test canonical name generation with real data"""
    meta = {
        'game': 'Test Game',
        'platform': 'unity_ads',
        'channel': 'android',
        'countries': ['United States'],
        'start_date': date(2024, 1, 1),
        'end_date': date(2024, 1, 31),
        'columns': ['date', 'cost', 'revenue', 'roas_d30'],
        'sample_rows': [
            {'date': '2024-01-01', 'cost': 50.0, 'revenue': 60.0, 'roas_d30': 1.6},
            {'date': '2024-01-02', 'cost': 75.0, 'revenue': 90.0, 'roas_d30': 1.6}
        ]
    }
    
    canonical_name = make_canonical_name(meta)
    
    # Verify name format
    assert canonical_name is not None
    assert len(canonical_name) > 0
    assert len(canonical_name) <= 80
    assert '_' in canonical_name  # Should contain underscores
    assert 'test_game' in canonical_name.lower()
    assert 'unity_ads' in canonical_name.lower()
    assert 'android' in canonical_name.lower()
    assert '20240101-20240131' in canonical_name


def test_dataset_retrieval():
    """Test dataset retrieval and filtering"""
    # Get all datasets
    datasets = get_datasets()
    assert isinstance(datasets, list)
    
    # Test filtering by platform
    unity_datasets = get_datasets(platform="unity_ads")
    assert isinstance(unity_datasets, list)
    
    # Test filtering by game
    test_datasets = get_datasets(game="Test Game")
    assert isinstance(test_datasets, list)


def test_data_end_date_setting():
    """Test that data_end_date is set to current date during ingestion"""
    # Create sample data
    data = {
        'date': ['2024-01-01', '2024-01-02'],
        'platform': ['Unity Ads', 'Unity Ads'],
        'channel': ['Android', 'Android'],
        'game': ['Test Game', 'Test Game'],
        'country': ['United States', 'United States'],
        'installs': [100, 150],
        'cost': [50.0, 75.0],
        'revenue': [60.0, 90.0],
        'roas_d30': [1.6, 1.6]
    }
    df = pd.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name
    
    try:
        dataset = ingest_file(temp_path, notes="Date test")
        
        # Verify data_end_date is set to today
        assert dataset.data_end_date == date.today()
        
    finally:
        os.unlink(temp_path)


def test_schema_fingerprint():
    """Test that schema fingerprint is generated correctly"""
    # Create sample data
    data = {
        'date': ['2024-01-01'],
        'cost': [50.0],
        'revenue': [60.0]
    }
    df = pd.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name
    
    try:
        dataset = ingest_file(temp_path, notes="Fingerprint test")
        
        # Verify schema fingerprint is generated
        assert dataset.schema_fingerprint is not None
        assert len(dataset.schema_fingerprint) == 32  # MD5 hash length
        
    finally:
        os.unlink(temp_path)
