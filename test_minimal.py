#!/usr/bin/env python3
"""
Minimal test to isolate Bus error cause
"""
import sys
import os

print("Testing imports one by one...")

try:
    print("1. Testing basic imports...")
    import streamlit as st
    print("✅ Streamlit imported successfully")
except Exception as e:
    print(f"❌ Streamlit import failed: {e}")
    sys.exit(1)

try:
    print("2. Testing pandas...")
    import pandas as pd
    print("✅ Pandas imported successfully")
except Exception as e:
    print(f"❌ Pandas import failed: {e}")
    sys.exit(1)

try:
    print("3. Testing numpy...")
    import numpy as np
    print("✅ Numpy imported successfully")
except Exception as e:
    print(f"❌ Numpy import failed: {e}")
    sys.exit(1)

try:
    print("4. Testing scikit-learn...")
    import sklearn
    print("✅ Scikit-learn imported successfully")
except Exception as e:
    print(f"❌ Scikit-learn import failed: {e}")
    sys.exit(1)

try:
    print("5. Testing LightGBM...")
    import lightgbm as lgb
    print("✅ LightGBM imported successfully")
except Exception as e:
    print(f"❌ LightGBM import failed: {e}")
    sys.exit(1)

try:
    print("6. Testing our data loader...")
    sys.path.append('src')
    from utils.data_loader import GameLensDataLoader
    print("✅ Data loader imported successfully")
except Exception as e:
    print(f"❌ Data loader import failed: {e}")
    sys.exit(1)

try:
    print("7. Testing feature engineering...")
    from utils.feature_engineering import GameLensFeatureEngineer
    print("✅ Feature engineering imported successfully")
except Exception as e:
    print(f"❌ Feature engineering import failed: {e}")
    sys.exit(1)

try:
    print("8. Testing ROAS forecaster...")
    from utils.roas_forecaster import GameLensROASForecaster
    print("✅ ROAS forecaster imported successfully")
except Exception as e:
    print(f"❌ ROAS forecaster import failed: {e}")
    sys.exit(1)

print("\n🎉 All core imports successful! The issue is likely in the LLM service or DOCX imports.")
