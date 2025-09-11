"""
GPT-assisted canonical naming system for GameLens AI
"""

import os
import re
from typing import Dict, Any, Optional
from datetime import date
import hashlib

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def make_canonical_name(meta: Dict[str, Any]) -> str:
    """
    Generate a canonical name for a dataset using GPT or deterministic fallback
    
    Args:
        meta: Metadata dictionary containing:
            - game: Game name
            - platform: Platform name
            - channel: Channel (android/ios/web)
            - countries: List of countries
            - start_date: Start date
            - end_date: End date
            - columns: List of column names
            - sample_rows: Sample data rows
            
    Returns:
        Canonical name string
    """
    # Try GPT first if available
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        try:
            return _make_canonical_name_gpt(meta)
        except Exception as e:
            print(f"GPT naming failed: {e}, falling back to deterministic")
    
    # Fallback to deterministic naming
    return _make_canonical_name_deterministic(meta)


def _make_canonical_name_gpt(meta: Dict[str, Any]) -> str:
    """
    Generate canonical name using GPT
    
    Args:
        meta: Metadata dictionary
        
    Returns:
        Canonical name string
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Prepare the prompt
    game = meta.get('game', 'Unknown Game')
    platform = meta.get('platform', 'unknown')
    channel = meta.get('channel', 'unknown')
    countries = meta.get('countries', [])
    start_date = meta.get('start_date')
    end_date = meta.get('end_date')
    columns = meta.get('columns', [])
    sample_rows = meta.get('sample_rows', [])
    
    # Format countries
    if not countries:
        countries_str = "UNKNOWN"
    elif len(countries) == 1:
        countries_str = countries[0].upper()
    else:
        countries_str = "MULTI"
    
    # Format dates
    start_str = start_date.strftime("%Y%m%d") if start_date else "UNKNOWN"
    end_str = end_date.strftime("%Y%m%d") if end_date else "UNKNOWN"
    
    prompt = f"""Generate a canonical dataset name following this exact pattern:
[game]_[platform]_[channel]_[countriesOrMULTI]_[startYYYYMMDD]-[endYYYYMMDD]

Constraints:
- Return ONLY the name, no explanation
- Use slug-style formatting (lowercase, underscores, no spaces)
- Maximum 80 characters
- ASCII characters only
- No special characters except underscores and hyphens

Dataset information:
- Game: {game}
- Platform: {platform}
- Channel: {channel}
- Countries: {countries_str}
- Date range: {start_str} to {end_str}
- Columns: {', '.join(columns[:10])}  # Show first 10 columns
- Sample data: {str(sample_rows[:2])}  # Show first 2 rows

Generate the canonical name:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a data naming expert. Generate concise, descriptive dataset names following the specified pattern."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0.1
    )
    
    canonical_name = response.choices[0].message.content.strip()
    
    # Clean up the response
    canonical_name = re.sub(r'[^a-zA-Z0-9_-]', '_', canonical_name)
    canonical_name = re.sub(r'_+', '_', canonical_name)  # Replace multiple underscores with single
    canonical_name = canonical_name.strip('_')
    
    # Ensure it follows the pattern
    if not _validate_canonical_name(canonical_name):
        # Fallback to deterministic if GPT response is invalid
        return _make_canonical_name_deterministic(meta)
    
    return canonical_name


def _make_canonical_name_deterministic(meta: Dict[str, Any]) -> str:
    """
    Generate canonical name using deterministic rules
    
    Args:
        meta: Metadata dictionary
        
    Returns:
        Canonical name string
    """
    # Extract components
    game = meta.get('game', 'unknown_game')
    platform = meta.get('platform', 'unknown')
    channel = meta.get('channel', 'unknown')
    countries = meta.get('countries', [])
    start_date = meta.get('start_date')
    end_date = meta.get('end_date')
    
    # Clean and format game name
    game_clean = re.sub(r'[^a-zA-Z0-9]', '_', str(game).lower())
    game_clean = re.sub(r'_+', '_', game_clean).strip('_')
    game_clean = game_clean[:20]  # Limit length
    
    # Clean platform name
    platform_clean = re.sub(r'[^a-zA-Z0-9]', '_', str(platform).lower())
    platform_clean = re.sub(r'_+', '_', platform_clean).strip('_')
    
    # Clean channel name
    channel_clean = re.sub(r'[^a-zA-Z0-9]', '_', str(channel).lower())
    channel_clean = re.sub(r'_+', '_', channel_clean).strip('_')
    
    # Format countries
    if not countries:
        countries_str = "unknown"
    elif len(countries) == 1:
        country_clean = re.sub(r'[^a-zA-Z0-9]', '_', str(countries[0]).lower())
        countries_str = re.sub(r'_+', '_', country_clean).strip('_')
    else:
        countries_str = "multi"
    
    # Format dates
    if start_date and end_date:
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        date_range = f"{start_str}-{end_str}"
    else:
        date_range = "unknown"
    
    # Construct canonical name
    canonical_name = f"{game_clean}_{platform_clean}_{channel_clean}_{countries_str}_{date_range}"
    
    # Ensure it's within length limit
    if len(canonical_name) > 80:
        # Truncate game name if needed
        max_game_len = 80 - len(f"_{platform_clean}_{channel_clean}_{countries_str}_{date_range}")
        if max_game_len > 0:
            game_clean = game_clean[:max_game_len]
            canonical_name = f"{game_clean}_{platform_clean}_{channel_clean}_{countries_str}_{date_range}"
        else:
            # Use hash if still too long
            hash_suffix = hashlib.md5(canonical_name.encode()).hexdigest()[:8]
            canonical_name = f"dataset_{hash_suffix}_{date_range}"
    
    return canonical_name


def _validate_canonical_name(name: str) -> bool:
    """
    Validate that a canonical name follows the required pattern
    
    Args:
        name: Canonical name to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not name or len(name) > 80:
        return False
    
    # Check for invalid characters
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        return False
    
    # Check for required components (at least 3 underscores for the pattern)
    if name.count('_') < 3:
        return False
    
    return True


def generate_model_name(target_day: int, platform: str = "multi") -> str:
    """
    Generate a model name for training
    
    Args:
        target_day: Target day for prediction (15, 30, 45, 90)
        platform: Platform name
        
    Returns:
        Model name string
    """
    platform_clean = re.sub(r'[^a-zA-Z0-9]', '_', str(platform).lower())
    platform_clean = re.sub(r'_+', '_', platform_clean).strip('_')
    
    return f"lgbm_roas_d{target_day}_{platform_clean}_quantile"


def generate_artifact_path(model_name: str, version: int) -> str:
    """
    Generate artifact path for model storage
    
    Args:
        model_name: Model name
        version: Model version
        
    Returns:
        Artifact path string
    """
    return f"artifacts/{model_name}/v{version}"
