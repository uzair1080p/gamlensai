#!/usr/bin/env python3
"""
Debug script to extract and test the Adaptive AI Recommendations logic
This will help identify where the data processing is going wrong
"""

import os
import sys
import pandas as pd
import json
from pathlib import Path

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the necessary modules
from glai.db import get_db_session
from glai.models import Dataset
from glai.recommend_gpt import get_gpt_recommendations
from glai.faq_gpt import GameLensFAQGPT

def load_raw_csv_data(dataset):
    """Load raw CSV data when normalized data has zeros"""
    try:
        # Try to find the original CSV file based on dataset info
        csv_path = None
        
        # First try specific paths based on platform/channel
        if dataset.source_platform == "unity_ads" and dataset.channel == "android":
            csv_path = "Campaign Data/Unity Ads/Android/Adspend and Revenue data.csv"
        elif dataset.source_platform == "unity_ads" and dataset.channel == "ios":
            csv_path = "Campaign Data/Unity Ads/iOS/Adspend+ Revenue .csv"
        elif dataset.source_platform == "mistplay" and dataset.channel == "android":
            csv_path = "Campaign Data/Mistplay/Android/Adspend & Revenue.csv"
        
        # If specific path doesn't exist, try to find any CSV file
        if not csv_path or not os.path.exists(csv_path):
            # Search in Campaign Data directory recursively
            campaign_data_path = "Campaign Data"
            if os.path.exists(campaign_data_path):
                for root, _, files in os.walk(campaign_data_path):
                    for file in files:
                        if file.endswith((".csv", ".xlsx", ".xls")) and dataset.raw_filename in file:
                            csv_path = os.path.join(root, file)
                            break
                    else:
                        continue
                    break
                else:
                    csv_path = None
            else:
                csv_path = None

        if not csv_path:
            # Fallback: For other datasets, try to find any CSV file in data/raw or root directory
            import glob
            # Exclude template files and artifact files
            csv_files = [f for f in glob.glob("data/raw/*.csv") + glob.glob("*.csv") + glob.glob("*.xlsx") + glob.glob("*.xls")
                         if "Data_Template" not in f and "gpt_usage" not in f and "gpt_logs" not in f and "demo_" not in f and not f.startswith("~$")]
            
            # Prioritize files matching the dataset's canonical name or raw filename
            matching_files = [f for f in csv_files if dataset.canonical_name in f or dataset.raw_filename in f]
            if matching_files:
                csv_path = matching_files[0]
            elif csv_files:
                csv_path = csv_files[0] # Use the first available CSV if no specific match
            else:
                return None
            
        if csv_path and os.path.exists(csv_path):
            print(f"📄 Loading file: {csv_path}")
            
            # Handle different file types
            if csv_path.endswith('.csv'):
                df = pd.read_csv(csv_path)
            elif csv_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(csv_path)
            else:
                return None
                
            # Normalize column names (strip spaces, lowercase)
            df.columns = [col.strip().lower() for col in df.columns]
            print(f"📄 Loaded {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"📄 Columns: {list(df.columns)}")
            
            # Handle duplicate columns by keeping the first occurrence
            df = df.loc[:, ~df.columns.duplicated()]
            print(f"📄 After removing duplicates: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"📄 Final columns: {list(df.columns)}")
            
            # Clean up cost and revenue columns if they contain lists
            if 'cost' in df.columns:
                # If cost column contains lists, extract the first element
                if df['cost'].dtype == 'object' and df['cost'].iloc[0] is not None and isinstance(df['cost'].iloc[0], list):
                    df['cost'] = df['cost'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
                    print("📄 Cleaned cost column (extracted first element from lists)")
            
            if 'revenue' in df.columns:
                # If revenue column contains lists, extract the first element
                if df['revenue'].dtype == 'object' and df['revenue'].iloc[0] is not None and isinstance(df['revenue'].iloc[0], list):
                    df['revenue'] = df['revenue'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
                    print("📄 Cleaned revenue column (extracted first element from lists)")
            
            # Show sample data
            if 'cost' in df.columns and 'revenue' in df.columns:
                print(f"📄 Sample 'cost' values: {df['cost'].head(5).values.tolist()}")
                print(f"📄 Sample 'revenue' values: {df['revenue'].head(5).values.tolist()}")
            
            return df
        else:
            print(f"❌ File not found: {csv_path}")
            return None
    except Exception as e:
        print(f"❌ Error loading raw CSV data: {e}")
        return None

def debug_adaptive_ai_logic():
    """Debug the Adaptive AI Recommendations logic"""
    print("🔍 Starting Adaptive AI Recommendations Debug")
    print("=" * 60)
    
    # Get the session and find the "test" dataset
    session = get_db_session()
    try:
        # Find the test dataset
        test_dataset = session.query(Dataset).filter(Dataset.canonical_name == "test").first()
        if not test_dataset:
            print("❌ No 'test' dataset found in database")
            return
        
        print(f"✅ Found dataset: {test_dataset.canonical_name}")
        print(f"   - Raw filename: {test_dataset.raw_filename}")
        print(f"   - Platform: {test_dataset.source_platform}")
        print(f"   - Channel: {test_dataset.channel}")
        print(f"   - Records: {test_dataset.records}")
        print()
        
        # Load raw CSV data
        print("📄 Loading raw CSV data...")
        raw_df = load_raw_csv_data(test_dataset)
        
        if raw_df is None or raw_df.empty:
            print("❌ Could not load raw CSV data")
            return
        
        print(f"✅ Raw data loaded: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
        print()
        
        # Show data summary
        print("📊 Data Summary:")
        print(f"   - Total rows: {len(raw_df)}")
        print(f"   - Columns: {list(raw_df.columns)}")
        
        # Check for cost and revenue columns
        cost_cols = [col for col in raw_df.columns if 'cost' in col.lower()]
        revenue_cols = [col for col in raw_df.columns if 'revenue' in col.lower()]
        
        print(f"   - Cost columns found: {cost_cols}")
        print(f"   - Revenue columns found: {revenue_cols}")
        
        # Show sample data
        print("\n📋 Sample Data (first 5 rows):")
        print(raw_df.head())
        
        # Check for non-zero values
        if cost_cols:
            cost_col = cost_cols[0]
            non_zero_costs = raw_df[cost_col][raw_df[cost_col] != 0].count()
            print(f"\n💰 Cost Analysis:")
            print(f"   - Non-zero cost values: {non_zero_costs}")
            print(f"   - Sample cost values: {raw_df[cost_col].head(10).values.tolist()}")
        
        if revenue_cols:
            revenue_col = revenue_cols[0]
            non_zero_revenue = raw_df[revenue_col][raw_df[revenue_col] != 0].count()
            print(f"\n💵 Revenue Analysis:")
            print(f"   - Non-zero revenue values: {non_zero_revenue}")
            print(f"   - Sample revenue values: {raw_df[revenue_col].head(10).values.tolist()}")
        
        print("\n" + "=" * 60)
        
        # Test GPT recommendations
        print("🤖 Testing GPT Recommendations...")
        gpt_map = {}  # Initialize gpt_map
        try:
            # Add row_index column like in the Streamlit app
            gpt_df = raw_df.copy()
            gpt_df['row_index'] = gpt_df.index
            
            gpt_map = get_gpt_recommendations(gpt_df)
            print(f"✅ GPT recommendations generated: {len(gpt_map)} recommendations")
            
            # Show sample recommendations
            if gpt_map:
                print("\n📋 Sample GPT Recommendations:")
                for i, (idx, rec) in enumerate(list(gpt_map.items())[:5]):
                    print(f"   Row {idx}: {rec.get('action', 'Unknown')} - {rec.get('rationale', 'No rationale')}")
            
        except Exception as e:
            print(f"❌ Error generating GPT recommendations: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 60)
        
        # Test FAQ system
        print("❓ Testing FAQ System...")
        try:
            # Build compact dataset for FAQ
            compact_for_faq = []
            for i, row in raw_df.iterrows():
                # Parse cost and revenue from raw strings
                cost_str = str(row.get("cost", "0")).replace("$", "").replace(",", "").strip()
                revenue_str = str(row.get("revenue", "0")).replace("$", "").replace(",", "").strip()
                
                try:
                    cost_val = float(cost_str) if cost_str and cost_str != "0" and cost_str != "-" else 0
                    revenue_val = float(revenue_str) if revenue_str and revenue_str != "0" and revenue_str != "-" else 0
                except (ValueError, TypeError):
                    cost_val = 0
                    revenue_val = 0
                
                installs = row.get("installs", 0)
                # Handle NaT and other non-numeric values
                try:
                    installs = float(installs) if pd.notna(installs) and installs != 0 else 0
                except (ValueError, TypeError):
                    installs = 0
                cpi = cost_val / installs if installs > 0 else 0
                
                compact_row = {
                    "Campaign": i + 1,
                    "CPI ($)": round(cpi, 2),
                    "Installs": installs,
                    "ROAS D7 (%)": 0,  # Simplified for debug
                    "ROAS D14 (%)": 0,  # Simplified for debug
                    "Recommended Action": gpt_map.get(i, {}).get("action", "Unknown") if gpt_map else "Unknown"
                }
                compact_for_faq.append(compact_row)
            
            print(f"✅ FAQ compact data built: {len(compact_for_faq)} rows")
            
            # Show sample FAQ data
            print("\n📋 Sample FAQ Data:")
            for i, row in enumerate(compact_for_faq[:5]):
                print(f"   Campaign {row['Campaign']}: CPI=${row['CPI ($)']}, Installs={row['Installs']}, Action={row['Recommended Action']}")
            
            # Test FAQ GPT
            faq_gpt = GameLensFAQGPT()
            test_question = "When will ROI of 100% be achieved on this channel?"
            
            # Build context
            context = {
                "system_info": {"app_name": "GameLens AI"},
                "current_context": {"page": "predictions"},
                "available_data": {"datasets": [test_dataset.canonical_name]},
                "ai_recommendations": {"count": len(gpt_map) if gpt_map else 0},
                "dataset_compact": compact_for_faq[:10],  # Limit for debug
                "has_predictions": True,
                "model_type": "Adaptive AI Recommendations (GPT-powered)"
            }
            
            print(f"\n🤖 Testing FAQ with question: '{test_question}'")
            print(f"   Context size: {len(json.dumps(context))} characters")
            print(f"   AI recommendations count: {context['ai_recommendations']['count']}")
            
            answer = faq_gpt.generate_faq_answer(test_question, context)
            if answer:
                print(f"✅ FAQ answer generated: {len(answer)} characters")
                print(f"   Preview: {answer[:200]}...")
            else:
                print("❌ No FAQ answer generated")
                
        except Exception as e:
            print(f"❌ Error testing FAQ system: {e}")
            import traceback
            traceback.print_exc()
        
    finally:
        session.close()
    
    print("\n" + "=" * 60)
    print("🏁 Debug completed")

if __name__ == "__main__":
    debug_adaptive_ai_logic()
