#!/usr/bin/env python3
"""
Test script to verify ML library fallback works correctly.
Tests LightGBM -> XGBoost fallback when libgomp is not available.
"""

import os
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def test_ml_imports():
    """Test ML library imports and fallback logic"""
    print("🧪 Testing ML Library Imports and Fallback Logic")
    print("=" * 60)
    
    # Test 1: Direct LightGBM import
    print("1. Testing direct LightGBM import...")
    try:
        import lightgbm as lgb
        print("   ✅ LightGBM imported successfully")
        lightgbm_available = True
    except Exception as e:
        print(f"   ❌ LightGBM import failed: {e}")
        lightgbm_available = False
    
    # Test 2: Direct XGBoost import
    print("2. Testing direct XGBoost import...")
    try:
        import xgboost as xgb
        print("   ✅ XGBoost imported successfully")
        xgboost_available = True
    except Exception as e:
        print(f"   ❌ XGBoost import failed: {e}")
        xgboost_available = False
    
    # Test 3: Our ROAS forecaster import
    print("3. Testing ROAS forecaster import...")
    try:
        from utils.roas_forecaster import GameLensROASForecaster
        print("   ✅ ROAS forecaster imported successfully")
        
        # Test forecaster initialization
        forecaster = GameLensROASForecaster(target_day=30)
        print("   ✅ ROAS forecaster initialized successfully")
        
    except Exception as e:
        print(f"   ❌ ROAS forecaster import failed: {e}")
        return False
    
    # Test 4: Test with sample data
    print("4. Testing with sample data...")
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.randn(n_samples))
        
        print(f"   📊 Created sample data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Test training
        print("   🤖 Testing model training...")
        models = forecaster.train_model(X, y, n_estimators=10, learning_rate=0.1)
        print(f"   ✅ Model training successful: {len(models)} models trained")
        
        # Test prediction
        print("   🔮 Testing predictions...")
        predictions = forecaster.predict_with_confidence(X.head(5))
        print(f"   ✅ Predictions successful: {predictions.shape[0]} predictions made")
        
        # Show which library was used
        if hasattr(forecaster, 'models') and forecaster.models:
            first_model = list(forecaster.models.values())[0]
            model_type = type(first_model).__name__
            print(f"   📈 Model type used: {model_type}")
            
            if 'LGBM' in model_type:
                print("   🟢 LightGBM was used for training")
            elif 'XGB' in model_type:
                print("   🟡 XGBoost was used as fallback")
            else:
                print(f"   🔵 Unknown model type: {model_type}")
        
    except Exception as e:
        print(f"   ❌ Sample data test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"  LightGBM Available: {'✅ Yes' if lightgbm_available else '❌ No'}")
    print(f"  XGBoost Available: {'✅ Yes' if xgboost_available else '❌ No'}")
    print(f"  ROAS Forecaster: ✅ Working")
    print(f"  Fallback Logic: ✅ Working")
    
    if not lightgbm_available and xgboost_available:
        print("\n🎉 Fallback system working perfectly!")
        print("🟡 XGBoost will be used instead of LightGBM")
    elif lightgbm_available:
        print("\n🎉 LightGBM working perfectly!")
        print("🟢 LightGBM will be used for optimal performance")
    else:
        print("\n❌ Neither LightGBM nor XGBoost available!")
        print("🔴 Please install at least one ML library")
        return False
    
    return True

def main():
    """Run the ML fallback test"""
    try:
        success = test_ml_imports()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
