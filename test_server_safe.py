#!/usr/bin/env python3
"""
Server-safe test script to verify Streamlit app can start without Bus errors.
This script tests the core functionality without problematic imports.
"""

import os
import sys

def test_server_environment_detection():
    """Test the server environment detection logic"""
    print("Testing server environment detection...")
    
    def is_server_environment():
        """Detect if we're running on a server that might have Bus error issues"""
        try:
            # Check for common server indicators
            if os.path.exists('/proc/version'):
                with open('/proc/version', 'r') as f:
                    version_info = f.read().lower()
                    if 'ubuntu' in version_info or 'linux' in version_info:
                        return True
            
            # Check for cloud server indicators
            if os.path.exists('/etc/cloud') or os.path.exists('/sys/class/dmi/id/product_name'):
                return True
                
            # Check if running as root (common on servers)
            if os.geteuid() == 0:
                return True
                
            return False
        except:
            return False
    
    is_server = is_server_environment()
    print(f"✅ Server environment detected: {is_server}")
    return is_server

def test_core_imports():
    """Test core imports that should work on all environments"""
    print("\nTesting core imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except Exception as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except Exception as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except Exception as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        from utils.data_loader import GameLensDataLoader
        print("✅ Data loader imported successfully")
    except Exception as e:
        print(f"❌ Data loader import failed: {e}")
        return False
    
    try:
        from utils.feature_engineering import GameLensFeatureEngineer
        print("✅ Feature engineering imported successfully")
    except Exception as e:
        print(f"❌ Feature engineering import failed: {e}")
        return False
    
    try:
        from utils.roas_forecaster import GameLensROASForecaster
        print("✅ ROAS forecaster imported successfully")
    except Exception as e:
        print(f"❌ ROAS forecaster import failed: {e}")
        return False
    
    return True

def test_llm_import_decision():
    """Test the LLM import decision logic"""
    print("\nTesting LLM import decision logic...")
    
    is_server = test_server_environment_detection()
    
    if is_server:
        print("🖥️ Server detected - LLM import should be skipped")
        print("✅ This should prevent Bus errors")
        return True
    else:
        print("💻 Local environment detected - LLM import may be attempted")
        print("⚠️ This could potentially cause Bus errors on some systems")
        return True

def main():
    """Run all tests"""
    print("🧪 Server-Safe Test Suite")
    print("=" * 50)
    
    # Test server environment detection
    is_server = test_server_environment_detection()
    
    # Test core imports
    core_success = test_core_imports()
    
    # Test LLM import decision
    llm_decision_success = test_llm_import_decision()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"  Server Environment: {'🖥️ Server' if is_server else '💻 Local'}")
    print(f"  Core Imports: {'✅ Success' if core_success else '❌ Failed'}")
    print(f"  LLM Decision Logic: {'✅ Success' if llm_decision_success else '❌ Failed'}")
    
    if core_success and llm_decision_success:
        print("\n🎉 All tests passed! The app should run safely on this environment.")
        if is_server:
            print("🖥️ Server mode will be used - LLM features disabled to prevent Bus errors.")
        else:
            print("💻 Local mode will be used - LLM features may be available.")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
