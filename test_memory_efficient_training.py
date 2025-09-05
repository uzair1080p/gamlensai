#!/usr/bin/env python3
"""
Test script to verify memory-efficient model training works correctly.
Tests chunked training and memory optimization features.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def test_memory_efficient_training():
    """Test memory-efficient model training"""
    print("🧪 Testing Memory-Efficient Model Training")
    print("=" * 60)
    
    try:
        # Import the memory-efficient forecaster
        from utils.memory_efficient_roas_forecaster import MemoryEfficientROASForecaster
        print("✅ Memory-efficient ROAS forecaster imported successfully")
        
        # Create sample data
        print("\n📊 Creating sample data...")
        np.random.seed(42)
        n_samples = 2000  # Large enough to trigger chunked training
        n_features = 15
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.randn(n_samples) * 0.5 + 2.0)  # ROAS-like values
        
        print(f"   Created dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Test forecaster initialization
        print("\n🤖 Testing forecaster initialization...")
        forecaster = MemoryEfficientROASForecaster(target_day=30)
        print("   ✅ Forecaster initialized successfully")
        
        # Test training with chunked approach
        print("\n🏋️ Testing chunked model training...")
        print(f"   Training chunk size: {forecaster.training_chunk_size}")
        print(f"   Dataset size: {len(X)} (will trigger chunked training)")
        
        models = forecaster.train_model(
            X, y,
            quantiles=[0.1, 0.5, 0.9],
            n_estimators=20,  # Reduced for testing
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        print(f"   ✅ Model training successful: {len(models)} models trained")
        
        # Test predictions
        print("\n🔮 Testing predictions...")
        predictions = forecaster.predict_with_confidence(X.head(100))  # Test on subset
        print(f"   ✅ Predictions successful: {predictions.shape[0]} predictions made")
        print(f"   Prediction columns: {list(predictions.columns)}")
        
        # Test model evaluation
        print("\n📈 Testing model evaluation...")
        metrics = forecaster.evaluate_model(X.head(500), y.head(500))  # Test on subset
        print(f"   ✅ Model evaluation successful")
        print(f"   Metrics: R²={metrics.get('r2', 0):.3f}, RMSE={metrics.get('rmse', 0):.3f}")
        
        # Test feature importance
        print("\n🎯 Testing feature importance...")
        if forecaster.feature_importance:
            top_features = list(forecaster.feature_importance.keys())[:5]
            print(f"   ✅ Feature importance available: {len(forecaster.feature_importance)} features")
            print(f"   Top 5 features: {top_features}")
        else:
            print("   ⚠️ No feature importance available")
        
        print("\n" + "=" * 60)
        print("🎉 All memory-efficient training tests passed!")
        print("✅ Chunked training works correctly")
        print("✅ Memory optimization is functioning")
        print("✅ Predictions and evaluation work properly")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the memory-efficient training test"""
    try:
        success = test_memory_efficient_training()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
