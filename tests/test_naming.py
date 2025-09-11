"""
Tests for GPT-assisted naming system
"""

import pytest
import os
from datetime import date
from glai.naming import make_canonical_name, _make_canonical_name_deterministic, _validate_canonical_name


def test_deterministic_naming():
    """Test deterministic naming fallback"""
    meta = {
        'game': 'Test Game',
        'platform': 'unity_ads',
        'channel': 'android',
        'countries': ['United States'],
        'start_date': date(2024, 1, 1),
        'end_date': date(2024, 1, 31)
    }
    
    name = _make_canonical_name_deterministic(meta)
    
    # Should follow the pattern: game_platform_channel_country_startdate-enddate
    assert 'test_game' in name
    assert 'unity_ads' in name
    assert 'android' in name
    assert '20240101-20240131' in name
    assert len(name) <= 80


def test_deterministic_naming_multi_country():
    """Test deterministic naming with multiple countries"""
    meta = {
        'game': 'Test Game',
        'platform': 'mistplay',
        'channel': 'ios',
        'countries': ['United States', 'Canada', 'Mexico'],
        'start_date': date(2024, 1, 1),
        'end_date': date(2024, 1, 31)
    }
    
    name = _make_canonical_name_deterministic(meta)
    
    assert 'test_game' in name
    assert 'mistplay' in name
    assert 'ios' in name
    assert 'multi' in name  # Should use 'multi' for multiple countries
    assert '20240101-20240131' in name


def test_deterministic_naming_missing_data():
    """Test deterministic naming with missing data"""
    meta = {
        'game': None,
        'platform': None,
        'channel': None,
        'countries': None,
        'start_date': None,
        'end_date': None
    }
    
    name = _make_canonical_name_deterministic(meta)
    
    # Should handle missing data gracefully
    assert len(name) > 0
    assert len(name) <= 80


def test_validate_canonical_name():
    """Test canonical name validation"""
    # Valid names
    assert _validate_canonical_name("test_game_unity_ads_android_us_20240101-20240131")
    assert _validate_canonical_name("game_platform_channel_country_20240101-20240131")
    
    # Invalid names
    assert not _validate_canonical_name("")  # Empty
    assert not _validate_canonical_name("a" * 100)  # Too long
    assert not _validate_canonical_name("invalid name with spaces")  # Spaces
    assert not _validate_canonical_name("invalid@name#with$special%chars")  # Special chars
    assert not _validate_canonical_name("short")  # Too few underscores


def test_make_canonical_name_fallback():
    """Test that make_canonical_name falls back to deterministic when GPT is unavailable"""
    meta = {
        'game': 'Test Game',
        'platform': 'unity_ads',
        'channel': 'android',
        'countries': ['United States'],
        'start_date': date(2024, 1, 1),
        'end_date': date(2024, 1, 31),
        'columns': ['date', 'cost', 'revenue'],
        'sample_rows': [{'date': '2024-01-01', 'cost': 100, 'revenue': 150}]
    }
    
    # Mock missing OpenAI key
    original_key = os.environ.get('OPENAI_API_KEY')
    if 'OPENAI_API_KEY' in os.environ:
        del os.environ['OPENAI_API_KEY']
    
    try:
        name = make_canonical_name(meta)
        
        # Should fall back to deterministic naming
        assert 'test_game' in name
        assert 'unity_ads' in name
        assert 'android' in name
        assert '20240101-20240131' in name
        assert _validate_canonical_name(name)
        
    finally:
        # Restore original key
        if original_key:
            os.environ['OPENAI_API_KEY'] = original_key


def test_generate_model_name():
    """Test model name generation"""
    from glai.naming import generate_model_name
    
    name = generate_model_name(30, "unity_ads")
    assert name == "lgbm_roas_d30_unity_ads_quantile"
    
    name = generate_model_name(90, "multi")
    assert name == "lgbm_roas_d90_multi_quantile"


def test_generate_artifact_path():
    """Test artifact path generation"""
    from glai.naming import generate_artifact_path
    
    path = generate_artifact_path("lgbm_roas_d30_unity_ads_quantile", 1)
    assert path == "artifacts/lgbm_roas_d30_unity_ads_quantile/v1"
    
    path = generate_artifact_path("test_model", 5)
    assert path == "artifacts/test_model/v5"
