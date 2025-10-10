"""
Unified GameLens AI page: Train, Predict, Validate, FAQ
Now integrated with the new gamlens functionality
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import io
import uuid
import json
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add gamlens to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
gamlens_path = PROJECT_ROOT / "gamlens"
if str(gamlens_path) not in sys.path:
    sys.path.insert(0, str(gamlens_path))

# Import gamlens modules
from lib.utils import ensure_dirs, read_csv_strict, DATA_DIR
from lib.kpis import add_core_kpis
from lib.ai import build_payload_for_ai, ask_one_question, ask_deepseek_question
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure repository root is on sys.path so `glai` can be imported for dataset management
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import GameLens modules for dataset management only
from glai.db import init_database, get_db_session
from glai.models import Dataset, ModelVersion, PredictionRun, PlatformEnum
from glai.ingest import ingest_file, get_datasets, get_dataset_by_id, load_dataset_data
from glai.naming import make_canonical_name

# Import PostgreSQL models for new integration
try:
    from glai.models_pg import get_distinct_source_files, get_data_by_source_file
    from glai.webhook_upload import upload_csv_to_n8n
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

def main():
    """Main page function"""
    st.set_page_config(
        page_title="GameLens AI - Train, Predict, Validate, FAQ",
        page_icon="🚀",
        layout="wide"
    )
    
    # Initialize database
    init_database()
    
    # Sidebar navigation
    st.sidebar.title("🚀 GameLens AI")
    page = st.sidebar.selectbox(
        "Navigate",
        ["📊 Dataset Management", "🧠 Predictions & AI", "📈 Model Training", "❓ FAQ"]
    )
    
    if page == "📊 Dataset Management":
        show_datasets_tab()
    elif page == "🧠 Predictions & AI":
        show_predictions_tab()
    elif page == "📈 Model Training":
        show_training_tab()
    elif page == "❓ FAQ":
        show_faq_tab()

def show_datasets_tab():
    """Show dataset management tab"""
    st.header("📊 Dataset Management")
    
    # Option 1: n8n Webhook Upload
    st.subheader("Option 1: Upload via n8n Webhook (Recommended)")
    if POSTGRESQL_AVAILABLE:
        uploaded_file = st.file_uploader(
            "Upload CSV file to PostgreSQL via n8n webhook",
            type=['csv'],
            key="webhook_upload"
        )
        
        if uploaded_file is not None:
            # Save to temporary file
            temp_path = f"/tmp/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Upload to n8n webhook
            with st.spinner(f"Uploading {uploaded_file.name} via n8n webhook..."):
                result = upload_csv_to_n8n(temp_path, uploaded_file.name)
            
            if result['success']:
                st.success(f"✅ Successfully uploaded via webhook: {uploaded_file.name}")
                st.write(f"Response: {result['response']}")
            else:
                st.error(f"❌ Failed to upload {uploaded_file.name}: {result['error']}")
            
            # Clean up temp file
            os.remove(temp_path)
    else:
        st.warning("PostgreSQL integration not available. Please install required dependencies.")
    
    # Option 2: Direct Upload (Legacy)
    st.subheader("Option 2: Direct Upload (Legacy)")
    uploaded_files = st.file_uploader(
        "Upload CSV or Excel files",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        key="direct_upload"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save to temporary file
            temp_path = f"/tmp/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process the file using robust gamlens parsing
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    # Use robust CSV parsing from gamlens
                    df = read_csv_strict(temp_path)
                    
                    # Save to gamlens data directory
                    gamlens_path = os.path.join(DATA_DIR, uploaded_file.name)
                    df.to_csv(gamlens_path, index=False)
                    
                    # Also ingest using existing system for compatibility
                    dataset = ingest_file(temp_path, notes=f"Uploaded via Streamlit")
                    
                    st.success(f"✅ Successfully processed: {uploaded_file.name}")
                    st.write(f"- Records: {len(df):,}")
                    st.write(f"- Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
                    st.write(f"- Date range: {df['date'].min()} to {df['date'].max()}")
                    st.write(f"- Games: {', '.join(df['game'].unique()[:3])}{'...' if len(df['game'].unique()) > 3 else ''}")
                    
                except Exception as parse_error:
                    st.error(f"❌ Error parsing {uploaded_file.name}: {str(parse_error)}")
                    st.info("💡 Try using the n8n webhook upload option above for better CSV handling")
                    continue
            
            # Clean up temp file
            os.remove(temp_path)
    
    # Existing datasets
    st.subheader("Existing Datasets")
    
    # Get datasets from database
    try:
        db = get_db_session()
        datasets = db.query(Dataset).filter(Dataset.ingest_completed_at.isnot(None)).order_by(Dataset.ingest_started_at.desc()).all()
        db.close()
        
        if not datasets:
            st.info("No datasets found. Upload a file above.")
            return
        
        # Dataset selection
        dataset_options = {f"{ds.canonical_name} ({ds.records:,} records)": ds for ds in datasets}
        selected_name = st.selectbox("Select a dataset:", list(dataset_options.keys()))
        
        if selected_name:
            selected_dataset = dataset_options[selected_name]
            
            # Display dataset info
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Platform:** {selected_dataset.source_platform}")
                st.write(f"**Channel:** {selected_dataset.channel}")
                st.write(f"**Game:** {selected_dataset.game}")
            with col2:
                st.write(f"**Records:** {selected_dataset.records:,}")
                st.write(f"**Date Range:** {selected_dataset.data_start_date} to {selected_dataset.data_end_date}")
                st.write(f"**Uploaded:** {selected_dataset.ingest_started_at}")
    
    except Exception as e:
        st.error(f"Error loading datasets: {str(e)}")

def show_predictions_tab():
    """Show predictions and AI tab"""
    st.header("🧠 Predictions & AI")
    
    # Dataset selection dropdown
    st.subheader("📁 Select Dataset")
    
    # Get available datasets from PostgreSQL csv_uploads table
    pg_datasets = []
    try:
        if POSTGRESQL_AVAILABLE:
            pg_datasets = get_distinct_source_files()
        else:
            st.warning("PostgreSQL not available. Using legacy dataset loading.")
    except Exception as e:
        st.error(f"Error connecting to PostgreSQL: {str(e)}")
        st.info("Please ensure the database credentials are correct in .env file")
    
    # Use PostgreSQL datasets as primary source
    all_datasets = pg_datasets if pg_datasets else []
    
    if not all_datasets:
        st.info("No datasets available. Please upload a dataset first.")
        return
    
    # Dataset selection dropdown
    selected_dataset_name = st.selectbox(
        "Select dataset:",
        all_datasets,
        key="dataset_selection"
    )
    
    if selected_dataset_name:
        # Load data from PostgreSQL
        try:
            df = get_data_by_source_file(selected_dataset_name)
            
            # Convert data types - PostgreSQL stores everything as TEXT, so we need to convert
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Convert numeric columns from TEXT to numeric types
            numeric_columns = [
                'cost', 'revenue', 'installs', 'clicks', 'impressions', 'ad_revenue',
                'roas_d0', 'roas_d1', 'roas_d3', 'roas_d7', 'roas_d14', 'roas_d30', 'roas_d60', 'roas_d90',
                'retention_rate_d1', 'retention_rate_d2', 'retention_rate_d3', 'retention_rate_d7', 'retention_rate_d14', 'retention_rate_d30',
                'level_1_events', 'level_5_events', 'level_10_events', 'level_15_events', 'level_20_events', 'level_25_events', 'level_30_events', 'level_40_events', 'level_50_events'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    # Convert from string to numeric, handling empty strings and nulls
                    df[col] = pd.to_numeric(df[col].astype(str).replace(['', 'None', 'null', 'NULL'], '0'), errors='coerce').fillna(0)
            
            # Add core KPIs
            df_kpi, summary_table = add_core_kpis(df)
            
            # Store in session state
            st.session_state.df_kpi = df_kpi
            st.session_state.selected_dataset = selected_dataset_name
            
            # Display summary metrics
            st.subheader("📊 Dataset Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Cost", f"${df_kpi['cost'].sum():,.2f}")
            with col2:
                st.metric("Total Revenue", f"${df_kpi['revenue'].sum():,.2f}")
            with col3:
                st.metric("Total Installs", f"{df_kpi['installs'].sum():,}")
            with col4:
                avg_roas = (df_kpi['revenue'].sum() / df_kpi['cost'].sum()) if df_kpi['cost'].sum() > 0 else 0
                st.metric("Average ROAS", f"{avg_roas:.2f}")
            
            # AI Recommendations Section
            st.subheader("🤖 AI Recommendations")
            
            # API key input
            env_key = os.getenv("OPENAI_API_KEY", "").strip()
            api_key = st.text_input(
                "Adaptive AI API Key (optional if OPENAI_API_KEY is set in .env)",
                type="password",
                value="",
                help="Leave empty to use OPENAI_API_KEY from .env file"
            )
            
            # Preset questions
            st.write("**Quick Questions:**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("When will ROI of 100% be achieved on this channel? D15? D30? D90?", use_container_width=True):
                    # Special prompt for ROAS curve projection
                    question = """You are a precise marketing data analyst who explains ROAS (Return On Ad Spend) performance and projections clearly and methodically.

You will receive structured JSON campaign data with fields like:
roas_d0, roas_d1, roas_d3, roas_d7, retention rates, cost, installs, and revenue.
These represent performance by day or cohort.

Your job is to evaluate ROAS performance and forecast when it will reach 100% (break-even), using clear reasoning and quantitative analysis.

---

⚠️ Critical Data Interpretation Rule:
- All roas_d* values are ALREADY normalized ratios between 0 and 1, where 1.0 = 100% ROI.
- Do NOT divide or rescale them again.
- When you see small decimals (e.g. 0.003, 0.004, 0.009), treat them as 0.3×, 0.4×, 0.9× cumulative ROAS (≈30%, 40%, 90%), not 0.3%.
- Multiply by 100 only when expressing percentages for humans.
- Never interpret them as 0.3% or 0.03%.
- Always assume roas_d* values represent *cumulative* ROAS growth toward 1.0 (100% ROI).

If a number looks "too small," assume under-reporting or early-stage data — do NOT rescale down further.

---

Your output must strictly follow this structure (in text form, not JSON):

---
OK: <true/false>
Insufficient data: <true/false>
Current average ROAS: <value or "N/A">
Projected final ROAS: <value or "N/A">
Break-even day: <number or "unknown">
Break-even date range: earliest <date>, latest <date>
Daily projection:
Day 0: <roas>
Day 1: <roas>
Day 3: <roas>
Day 7: <roas>
Day 10: <roas>
Day 14: <roas>
Day 21: <roas>
Day 30: <roas>
Assumptions:
- ...
- ...
Notes:
- ...
---

Follow this process strictly:

1️⃣ MODEL  
Fit a smooth, increasing cumulative ROAS curve:
ROAS(t) = Final_ROAS × (1 − exp(−k·t))

- Use observed roas_d* values (d0, d1, d3, d7, etc.) to estimate the curve.
- The curve must be monotonic (each later day ≥ previous).
- Choose Final_ROAS so that it aligns with retention decay — don't exceed plausible limits.
- Continue modeling until ROAS ≥ 1.0; that day = break-even.

2️⃣ DATES  
Use available cohort dates to derive real break-even range:
earliest = min(start_date) + break_even_day  
latest = max(start_date) + break_even_day  

3️⃣ RETENTION LINK  
- If retention drops sharply after day 3, slow ROAS growth after that point.
- If retention stabilizes, allow smoother growth toward 100%.

4️⃣ EXPLANATION STYLE  
- Write short, factual sentences.
- Quantify each observation.
- If data is missing or inconsistent, mark "Insufficient data: true".
- Keep tone analytical, neutral, and professional.
- Do not guess numbers — reason from provided data.

---

Do not output JSON.
Write only a human-readable report in the structure above."""
                    st.session_state.current_question = question
            
            with col2:
                if st.button("Should we continue running this campaign or pause it?", use_container_width=True):
                    st.session_state.current_question = "Should we continue running this campaign or pause it?"
            
            with col3:
                if st.button("What is the projected ROAS if we keep spending at the same pace?", use_container_width=True):
                    st.session_state.current_question = "What is the projected ROAS if we keep spending at the same pace?"
            
            with col4:
                if st.button("When will we reach 100% ROAS on each channel?", use_container_width=True):
                    # Special prompt for ROAS curve projection
                    question = """You are a precise marketing data analyst who explains ROAS (Return On Ad Spend) performance and projections clearly and methodically.

You will receive structured JSON campaign data with fields like:
roas_d0, roas_d1, roas_d3, roas_d7, retention rates, cost, installs, and revenue.
These represent performance by day or cohort.

Your job is to evaluate ROAS performance and forecast when it will reach 100% (break-even), using clear reasoning and quantitative analysis.

---

⚠️ Critical Data Interpretation Rule:
- All roas_d* values are ALREADY normalized ratios between 0 and 1, where 1.0 = 100% ROI.
- Do NOT divide or rescale them again.
- When you see small decimals (e.g. 0.003, 0.004, 0.009), treat them as 0.3×, 0.4×, 0.9× cumulative ROAS (≈30%, 40%, 90%), not 0.3%.
- Multiply by 100 only when expressing percentages for humans.
- Never interpret them as 0.3% or 0.03%.
- Always assume roas_d* values represent *cumulative* ROAS growth toward 1.0 (100% ROI).

If a number looks "too small," assume under-reporting or early-stage data — do NOT rescale down further.

---

Your output must strictly follow this structure (in text form, not JSON):

---
OK: <true/false>
Insufficient data: <true/false>
Current average ROAS: <value or "N/A">
Projected final ROAS: <value or "N/A">
Break-even day: <number or "unknown">
Break-even date range: earliest <date>, latest <date>
Daily projection:
Day 0: <roas>
Day 1: <roas>
Day 3: <roas>
Day 7: <roas>
Day 10: <roas>
Day 14: <roas>
Day 21: <roas>
Day 30: <roas>
Assumptions:
- ...
- ...
Notes:
- ...
---

Follow this process strictly:

1️⃣ MODEL  
Fit a smooth, increasing cumulative ROAS curve:
ROAS(t) = Final_ROAS × (1 − exp(−k·t))

- Use observed roas_d* values (d0, d1, d3, d7, etc.) to estimate the curve.
- The curve must be monotonic (each later day ≥ previous).
- Choose Final_ROAS so that it aligns with retention decay — don't exceed plausible limits.
- Continue modeling until ROAS ≥ 1.0; that day = break-even.

2️⃣ DATES  
Use available cohort dates to derive real break-even range:
earliest = min(start_date) + break_even_day  
latest = max(start_date) + break_even_day  

3️⃣ RETENTION LINK  
- If retention drops sharply after day 3, slow ROAS growth after that point.
- If retention stabilizes, allow smoother growth toward 100%.

4️⃣ EXPLANATION STYLE  
- Write short, factual sentences.
- Quantify each observation.
- If data is missing or inconsistent, mark "Insufficient data: true".
- Keep tone analytical, neutral, and professional.
- Do not guess numbers — reason from provided data.

---

Do not output JSON.
Write only a human-readable report in the structure above."""
                    st.session_state.current_question = question
            
            # Custom question input
            st.write("**Custom Question:**")
            custom_question = st.text_input("Ask anything about your data:")
            
            if st.button("Ask Custom Question") and custom_question:
                st.session_state.current_question = custom_question
            
            # Process current question
            if 'current_question' in st.session_state:
                question = st.session_state.current_question
                
                # Build payload for AI
                dataset_name = st.session_state.get('ds_name', selected_dataset_name)
                payload = build_payload_for_ai(df_kpi, dataset_name)
                
                # Get AI responses from both systems
                with st.spinner("🤖 Adaptive AI is analyzing your data..."):
                    answer1 = ask_one_question(api_key, question, payload)
                
                with st.spinner("🤖 Adaptive AI 2 is analyzing your data..."):
                    try:
                        answer2 = ask_deepseek_question(None, question, payload)  # Use DEEPSEEK_API_KEY from .env
                    except Exception as e:
                        answer2 = f"❌ Adaptive AI 2 Error: {str(e)}"
                
                # Display responses
                st.markdown("### 🤖 Adaptive AI Response")
                st.markdown(answer1)
                
                st.markdown("### 🤖 Adaptive AI 2 Response")
                st.markdown(answer2)
                
                # For ROAS question, also display the data sent to Adaptive AI
                st.write(f"🔍 Debug: Question = '{question[:100]}...'")  # Debug log
                
                # Check if this is the ROAS question (more flexible matching)
                is_roas_question = (
                    "When will we reach 100% ROAS on each channel?" in question or
                    "precise marketing data analyst" in question or
                    "ROAS(t) = Final_ROAS × (1 − exp(−k·t))" in question
                )
                
                if is_roas_question:
                    st.markdown("### 📊 Data Sent to Adaptive AI")
                    st.write("✅ ROAS question detected - showing data tables")
                    
                    # Display complete prompt sent to Adaptive AI
                    st.markdown("#### 🤖 Complete Prompt Sent to Adaptive AI")
                    complete_prompt = f"""System Prompt:
You are a precise marketing data analyst who explains ROAS (Return On Ad Spend) performance and projections clearly and methodically.

You will receive structured JSON campaign data with fields like:
roas_d0, roas_d1, roas_d3, roas_d7, retention rates, cost, installs, and revenue.
These represent performance by day or cohort.

Your job is to evaluate ROAS performance and forecast when it will reach 100% (break-even), using clear reasoning and quantitative analysis.

---

⚠️ Critical Data Interpretation Rule:
- All roas_d* values are ALREADY normalized ratios between 0 and 1, where 1.0 = 100% ROI.
- Do NOT divide or rescale them again.
- When you see small decimals (e.g. 0.003, 0.004, 0.009), treat them as 0.3×, 0.4×, 0.9× cumulative ROAS (≈30%, 40%, 90%), not 0.3%.
- Multiply by 100 only when expressing percentages for humans.
- Never interpret them as 0.3% or 0.03%.
- Always assume roas_d* values represent *cumulative* ROAS growth toward 1.0 (100% ROI).

If a number looks "too small," assume under-reporting or early-stage data — do NOT rescale down further.

---

Your output must strictly follow this structure (in text form, not JSON):

---
OK: <true/false>
Insufficient data: <true/false>
Current average ROAS: <value or "N/A">
Projected final ROAS: <value or "N/A">
Break-even day: <number or "unknown">
Break-even date range: earliest <date>, latest <date>
Daily projection:
Day 0: <roas>
Day 1: <roas>
Day 3: <roas>
Day 7: <roas>
Day 10: <roas>
Day 14: <roas>
Day 21: <roas>
Day 30: <roas>
Assumptions:
- ...
- ...
Notes:
- ...
---

Follow this process strictly:

1️⃣ MODEL  
Fit a smooth, increasing cumulative ROAS curve:
ROAS(t) = Final_ROAS × (1 − exp(−k·t))

- Use observed roas_d* values (d0, d1, d3, d7, etc.) to estimate the curve.
- The curve must be monotonic (each later day ≥ previous).
- Choose Final_ROAS so that it aligns with retention decay — don't exceed plausible limits.
- Continue modeling until ROAS ≥ 1.0; that day = break-even.

2️⃣ DATES  
Use available cohort dates to derive real break-even range:
earliest = min(start_date) + break_even_day  
latest = max(start_date) + break_even_day  

3️⃣ RETENTION LINK  
- If retention drops sharply after day 3, slow ROAS growth after that point.
- If retention stabilizes, allow smoother growth toward 100%.

4️⃣ EXPLANATION STYLE  
- Write short, factual sentences.
- Quantify each observation.
- If data is missing or inconsistent, mark "Insufficient data: true".
- Keep tone analytical, neutral, and professional.
- Do not guess numbers — reason from provided data.

---

Do not output JSON.
Write only a human-readable report in the structure above.

User Prompt:
Here is the campaign data in JSON format:
```json
{json.dumps(payload, indent=2)}
```

Question: {question}"""

                    with st.expander("📝 View Complete Prompt (Click to Expand)", expanded=False):
                        st.code(complete_prompt, language='text')
                    
                    # Display campaign data table
                    if 'all_campaigns' in payload:
                        campaigns_df = pd.DataFrame(payload['all_campaigns'])
                        st.markdown("#### 📈 Campaign Data Table")
                        st.dataframe(campaigns_df, use_container_width=True)
                        st.caption(f"📊 Campaign Data: {len(campaigns_df)} records sent to Adaptive AI")
                    
                    # Display aggregated data
                    if 'aggregates_channel_country' in payload:
                        agg_df = pd.DataFrame(payload['aggregates_channel_country'])
                        st.markdown("#### 📋 Channel-Country Aggregates")
                        st.dataframe(agg_df, use_container_width=True)
                        st.caption("📊 Aggregated metrics by channel and country")
                    
                    # Display data format info
                    if 'data_format' in payload:
                        st.markdown("#### ℹ️ Data Format Information")
                        for key, value in payload['data_format'].items():
                            st.text(f"{key}: {value}")
                    
                    # Display payload summary
                    st.markdown("#### 📊 Payload Summary")
                    payload_summary = {
                        "Dataset Name": payload.get('dataset_name', 'N/A'),
                        "Schema Version": payload.get('schema_version', 'N/A'),
                        "Granularity": payload.get('granularity', 'N/A'),
                        "Total Campaigns": len(payload.get('all_campaigns', [])),
                        "Channel-Country Combinations": len(payload.get('aggregates_channel_country', [])),
                        "Columns Included": len(payload.get('columns', [])),
                        "Rows Count": payload.get('rows_count', 'N/A')
                    }
                    for key, value in payload_summary.items():
                        st.text(f"{key}: {value}")
                
                # Add to chat history
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                st.session_state.chat_history.append(("User", question))
                st.session_state.chat_history.append(("Adaptive AI", answer1))
                st.session_state.chat_history.append(("Adaptive AI 2", answer2))
                
                # Clear current question
                del st.session_state.current_question
        
        except Exception as e:
            st.error(f"Error loading from PostgreSQL: {str(e)}")

def show_training_tab():
    """Show training tab"""
    st.header("🧠 Model Training")
    st.info("Model training functionality is being integrated with the new gamlens system for improved performance and accuracy.")

def show_faq_tab():
    """Show FAQ tab"""
    st.header("❓ FAQ")
    
    st.markdown("""
    ### 📊 Dataset Management
    **Q: What file formats are supported?**
    A: CSV, Excel (.xlsx, .xls) files are supported. For best results, use the n8n webhook upload option.

    **Q: What if my CSV has formatting issues?**
    A: The n8n webhook upload includes robust parsing that handles various CSV formats, encodings, and data quality issues.

    ### 🤖 AI Recommendations
    **Q: What questions can I ask?**
    A: You can ask about campaign performance, ROI analysis, scaling recommendations, retention analysis, and more. Use the preset buttons or type your own questions.

    **Q: Do I need an Adaptive AI API key?**
    A: Yes, for AI recommendations. Set OPENAI_API_KEY in your .env file or enter it in the UI.

    **Q: How accurate are the AI recommendations?**
    A: The AI analyzes your actual campaign data and provides data-driven insights. Always validate recommendations with your team and historical performance.

    ### 🧠 Model Training
    **Q: How do I train a model?**
    A: Model training functionality is being integrated with the new gamlens system for improved performance and accuracy.

    ### 🔧 Technical
    **Q: What if I get an error?**
    A: Check that your data matches the expected schema, ensure your Adaptive AI API key is valid, and try refreshing the page.

    **Q: Can I export results?**
    A: Yes, you can copy AI responses and download data tables from the interface.
    """)

if __name__ == "__main__":
    main()
