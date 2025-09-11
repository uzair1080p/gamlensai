"""
Demo setup script for GameLens AI
"""

import pandas as pd
import os
from datetime import date, timedelta
from glai.db import init_database
from glai.ingest import ingest_file
from glai.train import train_lgbm_quantile
from glai.predict import run_predictions

def create_demo_data():
    """Create demo CSV data"""
    # Create sample Unity Ads data
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    
    data = []
    for i, date_val in enumerate(dates):
        data.append({
            'date': date_val.strftime('%Y-%m-%d'),
            'platform': 'Unity Ads',
            'channel': 'Android',
            'game': 'Demo Game',
            'country': 'United States',
            'installs': 100 + i * 5,
            'cost': 50.0 + i * 2.5,
            'revenue': 60.0 + i * 3.0,
            'roas_d0': 1.2 + (i % 10) * 0.1,
            'roas_d1': 1.3 + (i % 10) * 0.1,
            'roas_d3': 1.4 + (i % 10) * 0.1,
            'roas_d7': 1.5 + (i % 10) * 0.1,
            'roas_d14': 1.6 + (i % 10) * 0.1,
            'roas_d30': 1.7 + (i % 10) * 0.1,
            'roas_d60': 1.8 + (i % 10) * 0.1,
            'roas_d90': 1.9 + (i % 10) * 0.1,
            'retention_d1': 80 + (i % 5),
            'retention_d3': 70 + (i % 5),
            'retention_d7': 60 + (i % 5),
            'retention_d30': 50 + (i % 5)
        })
    
    df = pd.DataFrame(data)
    return df

def setup_demo():
    """Set up demo data and models"""
    print("🚀 Setting up GameLens AI Demo...")
    
    # Initialize database
    print("📊 Initializing database...")
    init_database()
    
    # Create demo data
    print("📝 Creating demo data...")
    demo_df = create_demo_data()
    
    # Save demo data
    demo_file = "demo_unity_ads_android.csv"
    demo_df.to_csv(demo_file, index=False)
    print(f"💾 Saved demo data to {demo_file}")
    
    # Ingest demo data
    print("📥 Ingesting demo data...")
    try:
        dataset = ingest_file(demo_file, notes="Demo dataset for testing")
        print(f"✅ Dataset ingested: {dataset.canonical_name}")
        print(f"   - Records: {dataset.records}")
        print(f"   - Date range: {dataset.data_start_date} to {dataset.data_end_date}")
        
        # Train a demo model
        print("🤖 Training demo model...")
        try:
            model_version = train_lgbm_quantile(
                dataset_ids=[str(dataset.id)],
                target_day=30,
                params={
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'n_estimators': 100
                },
                notes="Demo model for testing"
            )
            print(f"✅ Model trained: {model_version.model_name} v{model_version.version}")
            
            # Run predictions
            print("📈 Running demo predictions...")
            try:
                prediction_run = run_predictions(
                    model_version_id=str(model_version.id),
                    dataset_id=str(dataset.id),
                    targets=[30]
                )
                print(f"✅ Predictions completed: {prediction_run.n_rows} rows processed")
                
            except Exception as e:
                print(f"⚠️ Prediction failed: {e}")
                
        except Exception as e:
            print(f"⚠️ Training failed: {e}")
            
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
    
    print("\n🎉 Demo setup complete!")
    print("\nTo run the application:")
    print("1. streamlit run pages/2_🚀_Train_Predict_Validate_FAQ.py")
    print("2. Or use: make dev")

if __name__ == "__main__":
    setup_demo()
