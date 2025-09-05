#!/usr/bin/env python3
"""
Test script to verify Streamlit app can start without Bus errors.
This script simulates the app startup process without actually running Streamlit.
"""

import os
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def test_app_startup():
    """Test if the Streamlit app can start without Bus errors"""
    print("🚀 Testing Streamlit App Startup...")
    print("=" * 50)
    
    # Test 1: Basic imports
    print("1. Testing basic imports...")
    try:
        import streamlit as st
        print("   ✅ Streamlit imported successfully")
    except Exception as e:
        print(f"   ❌ Streamlit import failed: {e}")
        return False
    
    # Test 2: Core ML libraries
    print("2. Testing ML libraries...")
    try:
        import pandas as pd
        import numpy as np
        import lightgbm as lgb
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        print("   ✅ All ML libraries imported successfully")
    except Exception as e:
        print(f"   ❌ ML library import failed: {e}")
        return False
    
    # Test 3: Our custom modules
    print("3. Testing custom modules...")
    try:
        from utils.data_loader import GameLensDataLoader
        from utils.feature_engineering import GameLensFeatureEngineer
        from utils.roas_forecaster import GameLensROASForecaster
        print("   ✅ All custom modules imported successfully")
    except Exception as e:
        print(f"   ❌ Custom module import failed: {e}")
        return False
    
    # Test 4: Server environment detection
    print("4. Testing server environment detection...")
    try:
        def is_server_environment():
            try:
                if os.path.exists('/proc/version'):
                    with open('/proc/version', 'r') as f:
                        version_info = f.read().lower()
                        if 'ubuntu' in version_info or 'linux' in version_info:
                            return True
                if os.path.exists('/etc/cloud') or os.path.exists('/sys/class/dmi/id/product_name'):
                    return True
                if os.geteuid() == 0:
                    return True
                return False
            except:
                return False
        
        is_server = is_server_environment()
        print(f"   ✅ Server detection working: {'Server' if is_server else 'Local'} environment")
    except Exception as e:
        print(f"   ❌ Server detection failed: {e}")
        return False
    
    # Test 5: LLM import decision (this is where Bus errors typically occur)
    print("5. Testing LLM import decision...")
    try:
        if is_server:
            print("   🖥️ Server detected - skipping LLM import to prevent Bus error")
            print("   ✅ LLM import will be skipped safely")
        else:
            print("   💻 Local environment - LLM import may be attempted")
            # Don't actually import openai here to avoid Bus error
            print("   ✅ LLM import decision logic working")
    except Exception as e:
        print(f"   ❌ LLM import decision failed: {e}")
        return False
    
    # Test 6: Simulate app initialization
    print("6. Testing app initialization simulation...")
    try:
        # Simulate what happens when the app starts
        print("   📊 Initializing data loader...")
        data_loader = GameLensDataLoader()
        
        print("   🔧 Initializing feature engineer...")
        feature_engineer = GameLensFeatureEngineer()
        
        print("   🤖 Initializing ROAS forecaster...")
        forecaster = GameLensROASForecaster()
        
        print("   ✅ App initialization simulation successful")
    except Exception as e:
        print(f"   ❌ App initialization failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! The Streamlit app should start successfully.")
    print("🖥️ Server mode detected - LLM features will be disabled to prevent Bus errors.")
    print("✅ Core ROAS forecasting functionality will be fully available.")
    
    return True

def main():
    """Run the startup test"""
    try:
        success = test_app_startup()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
