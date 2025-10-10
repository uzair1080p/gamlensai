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
from lib.ai import build_payload_for_ai, ask_one_question
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

# Page configuration
st.set_page_config(
    page_title="GameLens AI - Train, Predict, Validate, FAQ",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
@st.cache_resource
def init_db():
    """Initialize database connection"""
    return init_database()

# Initialize database
init_db()

# Ensure data directories exist
ensure_dirs()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
    .selection-banner {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

def show_selection_banner():
    """Show dataset selection banner"""
    if 'selected_dataset' in st.session_state and st.session_state.selected_dataset:
        dataset = st.session_state.selected_dataset
        st.markdown(f"""
        <div class="selection-banner">
            <h4>📊 Selected Dataset: {dataset.canonical_name}</h4>
            <p><strong>Platform:</strong> {dataset.source_platform} | 
            <strong>Channel:</strong> {dataset.channel} | 
            <strong>Game:</strong> {dataset.game} | 
            <strong>Records:</strong> {dataset.records:,}</p>
        </div>
        """, unsafe_allow_html=True)

def show_datasets_tab():
    """Show datasets tab with gamlens integration"""
    st.header("📦 Dataset Management")
    
    # File upload section
    st.subheader("Upload New Dataset")
    
    # n8n Webhook Upload Option
    if POSTGRESQL_AVAILABLE:
        st.write("**Option 1: Upload via n8n Webhook (Recommended)**")
        st.info("Use the n8n webhook to upload CSV files directly to the PostgreSQL database. This bypasses CSV parsing issues.")
        
        webhook_files = st.file_uploader(
            "Upload CSV files via n8n webhook",
            type=["csv"],
            accept_multiple_files=True,
            help="Upload CSV files that will be processed by the n8n workflow and stored in PostgreSQL",
            key="webhook_uploader"
        )
        
        if webhook_files:
            for uploaded_file in webhook_files:
                try:
                    # Save uploaded file temporarily
                    temp_path = f"temp_{uploaded_file.name}"
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
                    
                except Exception as e:
                    st.error(f"❌ Error uploading {uploaded_file.name}: {str(e)}")
    
    st.write("**Option 2: Direct Upload (Legacy)**")
    uploaded_files = st.file_uploader(
        "Upload CSV or Excel files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Upload files following the GameLens data template schema"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Also save to gamlens data directory
                gamlens_path = os.path.join(DATA_DIR, uploaded_file.name)
                with open(gamlens_path, "wb") as f:
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
                
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
    
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
            st.session_state.selected_dataset = selected_dataset
            
            # Show dataset details
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

def show_training_tab():
    """Show training tab"""
    st.header("🧠 Model Training")
    
    if 'selected_dataset' not in st.session_state or not st.session_state.selected_dataset:
        st.warning("Please select a dataset first.")
        return
    
    selected_dataset = st.session_state.selected_dataset
    
    st.info("🚧 **Training functionality is being integrated with the new gamlens system.**")
    st.write("This will include:")
    st.write("- LightGBM/XGBoost baseline models")
    st.write("- ROAS forecasting models")
    st.write("- Model validation and comparison")
    st.write("- Integration with the new AI recommendation system")

def show_predictions_tab():
    """Show predictions tab with PostgreSQL integration"""
    st.header("🔮 Predictions & AI Recommendations")
    
    # Dataset selection dropdown
    st.subheader("📁 Select Dataset")
    
    # Get available datasets from PostgreSQL csv_uploads table
    pg_datasets = []
    try:
        if POSTGRESQL_AVAILABLE:
            pg_datasets = get_distinct_source_files()
    except Exception as e:
        st.error(f"Error connecting to PostgreSQL: {str(e)}")
        st.info("Please ensure the database credentials are correct in .env file")
        return
    
    # Use PostgreSQL datasets as primary source
    all_datasets = pg_datasets
    
    if not all_datasets:
        st.warning("No datasets found. Please upload a CSV file using the n8n webhook or the Dataset Management tab.")
        return
    
    compatible_datasets = all_datasets
    
    # Dataset selection with session state to prevent reset
    session_key = f"selected_dataset_predictions"
    if session_key not in st.session_state:
        st.session_state[session_key] = compatible_datasets[0] if compatible_datasets else None
    
    selected_dataset_name = st.selectbox(
        "Choose a dataset:",
        compatible_datasets,
        index=compatible_datasets.index(st.session_state[session_key]) if st.session_state[session_key] in compatible_datasets else 0,
        help="Select the dataset you want to analyze",
        key=session_key
    )
    
    if not selected_dataset_name:
        st.warning("Please select a dataset.")
        return
    
    # Load dataset data using gamlens
    try:
        dataset_loaded = False
        df_kpi = None
        table = None
        
        # Load from PostgreSQL csv_uploads table
        if selected_dataset_name in pg_datasets:
            try:
                csv_uploads = get_data_by_source_file(selected_dataset_name)
                
                if not csv_uploads:
                    st.error(f"❌ No data found for dataset: {selected_dataset_name}")
                    return
                
                # Convert to DataFrame
                data_dicts = []
                for upload in csv_uploads:
                    row_dict = {
                        'game': upload.game,
                        'channel': upload.channel,
                        'platform': upload.platform,
                        'country': upload.country,
                        'date': upload.date,
                        'installs': upload.installs,
                        'cost': upload.cost,
                        'ad_revenue': upload.ad_revenue,
                        'revenue': upload.revenue,
                        'roas_d0': upload.roas_d0,
                        'roas_d1': upload.roas_d1,
                        'roas_d3': upload.roas_d3,
                        'roas_d7': upload.roas_d7,
                        'roas_d14': upload.roas_d14,
                        'roas_d30': upload.roas_d30,
                        'roas_d60': upload.roas_d60,
                        'roas_d90': upload.roas_d90,
                        'retention_rate_d1': upload.retention_rate_d1,
                        'retention_rate_d2': upload.retention_rate_d2,
                        'retention_rate_d3': upload.retention_rate_d3,
                        'retention_rate_d7': upload.retention_rate_d7,
                        'retention_rate_d14': upload.retention_rate_d14,
                        'retention_rate_d30': upload.retention_rate_d30,
                        'level_1_events': upload.level_1_events,
                        'level_5_events': upload.level_5_events,
                        'level_10_events': upload.level_10_events,
                        'level_15_events': upload.level_15_events,
                        'level_20_events': upload.level_20_events,
                        'level_25_events': upload.level_25_events,
                        'level_30_events': upload.level_30_events,
                        'level_40_events': upload.level_40_events,
                        'level_50_events': upload.level_50_events,
                    }
                    data_dicts.append(row_dict)
                
                df = pd.DataFrame(data_dicts)
                
                # Convert text columns to appropriate data types
                # Convert date column
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                
                # Convert integer columns
                for col in ['installs', 'cost']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')  # Nullable integer
                
                # Convert float columns
                for col in ['ad_revenue', 'revenue', 'roas_d0', 'roas_d1', 'roas_d3', 'roas_d7', 
                           'roas_d14', 'roas_d30', 'roas_d60', 'roas_d90',
                           'retention_rate_d1', 'retention_rate_d2', 'retention_rate_d3', 
                           'retention_rate_d7', 'retention_rate_d14', 'retention_rate_d30',
                           'level_1_events', 'level_5_events', 'level_10_events', 
                           'level_15_events', 'level_20_events', 'level_25_events',
                           'level_30_events', 'level_40_events', 'level_50_events']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Add KPIs
                df_kpi, table = add_core_kpis(df)
                st.success(f"✅ Loaded dataset from PostgreSQL: {selected_dataset_name}")
                dataset_loaded = True
                
            except Exception as e:
                st.error(f"Error loading from PostgreSQL: {str(e)}")
        
        # Fallback: If PostgreSQL fails, show error (no local CSV support in this version)
        if not dataset_loaded:
            st.error("❌ Failed to load dataset from PostgreSQL. Please ensure data is uploaded via n8n webhook.")
        
        if dataset_loaded:
            # Store in session state
            st.session_state.df_kpi = df_kpi
            st.session_state.table = table
            st.session_state.ds_name = selected_dataset_name
        else:
            st.error("❌ Could not load dataset data from any source")
            return
    
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return
    
    # Display summary metrics
    st.subheader("📊 Campaign Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_cost = df_kpi["cost"].sum()
        st.metric("Total Cost", f"${total_cost:,.2f}")
    with col2:
        total_revenue = df_kpi["total_revenue"].sum()
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    with col3:
        total_installs = df_kpi["installs"].sum()
        st.metric("Total Installs", f"{total_installs:,.0f}")
    with col4:
        avg_roas = df_kpi["ROAS"].mean()
        st.metric("Average ROAS", f"{avg_roas:.2f}")
    
    # Summary table
    st.dataframe(table, use_container_width=True)
    
    # AI Recommendations Section
    st.subheader("🤖 AI Recommendations")
    
    # API key input
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    api_key = st.text_input(
        "OpenAI API Key (optional if OPENAI_API_KEY is set in .env)",
        type="password",
        value="",
        help="Leave empty to use OPENAI_API_KEY from .env file"
    )
    
    # Preset questions
    st.write("**Quick Questions:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🎯 Which campaigns should we pause?", use_container_width=True):
            question = "Which campaigns should be paused due to poor ROI/retention? Provide bullets with thresholds and reasons."
            st.session_state.current_question = question
    
    with col2:
        if st.button("💰 Required CPI for D30 profitability?", use_container_width=True):
            question = "What CPI is needed to reach profitability by D30 per channel? Provide a table with assumptions."
            st.session_state.current_question = question
    
    with col3:
        if st.button("🚀 Which geo is ready to scale?", use_container_width=True):
            question = "Which geo is ready to scale aggressively? Provide criteria, data support, and risks."
            st.session_state.current_question = question
    
    with col4:
        if st.button("When will we reach 100% ROAS on each channel?", use_container_width=True):
            # Special prompt for ROAS curve projection
            question = """You are a precise marketing data analyst who explains ROAS (Return On Ad Spend) performance
and projections clearly and methodically. 
You will receive real campaign data points (ROAS, retention, cohorts) and must walk through
the analysis step by step — never skipping or guessing numbers.

Your answer must follow this exact structure (in text form, not JSON):

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

Follow this analysis process strictly:

1️⃣ MODEL  
Fit a smooth, increasing curve of cumulative ROAS:
ROAS(t) = Final_ROAS × (1 − exp(−k·t))  

- Choose *k* so that the curve roughly passes through the observed ROAS points.  
- Choose *Final_ROAS* so that the curve saturates consistently with the retention decay pattern.
- The curve must be monotonic (each day ≥ previous day).  
- Continue until ROAS ≥ 1.0; that day = break-even.

2️⃣ DATES  
Use the provided cohort start dates to compute actual break-even date range:  
earliest = min(cohort.start_date) + break_even_day  
latest = max(cohort.start_date) + break_even_day  

3️⃣ RETENTION LINK  
Adjust the saturation (Final_ROAS) downward if retention declines sharply between d3–d7.  
Retention tells you how sustainable later conversions are.

4️⃣ EXPLANATION STYLE  
- Explain in short, clear sentences.
- Quantify every step (don't invent new data).
- If some inputs are missing, mark them as unavailable and reason cautiously.
- Keep tone analytical, calm, confident.

---

Do not output JSON. Write a human-readable analysis in the structure above.
Return nothing outside this format."""
            st.session_state.current_question = question
    
    # Custom question
    st.write("**Custom Question:**")
    custom_question = st.text_input(
        "Ask anything about your campaigns:",
        placeholder="e.g., When will ROI reach 100% per channel?",
        key="custom_question_input"
    )
    
    if st.button("Ask AI", type="primary", use_container_width=True):
        if custom_question:
            st.session_state.current_question = custom_question
    
    # Process question
    if hasattr(st.session_state, 'current_question') and st.session_state.current_question:
        question = st.session_state.current_question
        
        try:
            # Build payload for AI
            dataset_name = st.session_state.get('ds_name', selected_dataset_name)
            payload = build_payload_for_ai(df_kpi, dataset_name)
            
            # Get AI response
            with st.spinner("🤖 AI is analyzing your data..."):
                answer = ask_one_question(api_key, question, payload)
            
            # Display response
            st.markdown("### 🤖 AI Response")
            st.markdown(answer)
            
            # For ROAS question, also display the data sent to GPT
            st.write(f"🔍 Debug: Question = '{question[:100]}...'")  # Debug log
            
            # Check if this is the ROAS question (more flexible matching)
            is_roas_question = (
                "When will we reach 100% ROAS on each channel?" in question or
                "precise marketing data analyst" in question or
                "ROAS(t) = Final_ROAS × (1 − exp(−k·t))" in question
            )
            
            if is_roas_question:
                st.markdown("### 📊 Data Sent to GPT")
                st.write("✅ ROAS question detected - showing data tables")
                
                # Display complete prompt sent to GPT
                st.markdown("#### 🤖 Complete Prompt Sent to GPT")
                complete_prompt = f"""System Prompt:
You are a precise marketing data analyst who explains ROAS (Return On Ad Spend) performance and projections clearly and methodically. 
You will receive real campaign data points (ROAS, retention, cohorts) and must walk through the analysis step by step — never skipping or guessing numbers.

Your answer must follow this exact structure (in text form, not JSON):

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

Follow this analysis process strictly:

1️⃣ MODEL  
Fit a smooth, increasing curve of cumulative ROAS:
ROAS(t) = Final_ROAS × (1 − exp(−k·t))  

- Choose *k* so that the curve roughly passes through the observed ROAS points.  
- Choose *Final_ROAS* so that the curve saturates consistently with the retention decay pattern.
- The curve must be monotonic (each day ≥ previous day).  
- Continue until ROAS ≥ 1.0; that day = break-even.

2️⃣ DATES  
Use the provided cohort start dates to compute actual break-even date range:  
earliest = min(cohort.start_date) + break_even_day  
latest = max(cohort.start_date) + break_even_day  

3️⃣ RETENTION LINK  
Adjust the saturation (Final_ROAS) downward if retention declines sharply between d3–d7.  
Retention tells you how sustainable later conversions are.

4️⃣ EXPLANATION STYLE  
- Explain in short, clear sentences.
- Quantify every step (don't invent new data).
- If some inputs are missing, mark them as unavailable and reason cautiously.
- Keep tone analytical, calm, confident.

---

Do not output JSON. Write a human-readable analysis in the structure above.
Return nothing outside this format.

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
                    st.caption(f"📊 Campaign Data: {len(campaigns_df)} records sent to GPT")
                
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
            st.session_state.chat_history.append(("AI", answer))
            
            # Clear current question
            del st.session_state.current_question
            
        except Exception as e:
            st.error(f"Error getting AI response: {str(e)}")
    
    # Chat history
    if 'chat_history' in st.session_state and st.session_state.chat_history:
        st.subheader("💬 Chat History")
        for role, message in st.session_state.chat_history[-10:]:  # Show last 10 messages
            with st.expander(f"{role}: {message[:100]}..." if len(message) > 100 else f"{role}: {message}"):
                st.markdown(message)
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

def show_faq_tab():
    """Show FAQ tab"""
    st.header("❓ FAQ")
    
    st.markdown("""
    ## GameLens AI - Frequently Asked Questions
    
    ### 📊 Data & Upload
    **Q: What data format should I use?**
    A: Upload CSV files that match the data template schema with columns: game, channel, platform, country, date, installs, cost, ad_revenue, revenue, roas_d0-d90, retention_rate_d1-d30, level_1_events-level_50_events.
    
    **Q: How do I get started?**
    A: 1. Upload your CSV data on the Dataset Management tab, 2. Select it for analysis, 3. Ask questions using the AI assistant on the Predictions tab.
    
    ### 🤖 AI Recommendations
    **Q: What questions can I ask?**
    A: You can ask about campaign performance, ROI analysis, scaling recommendations, retention analysis, and more. Use the preset buttons or type your own questions.
    
    **Q: Do I need an OpenAI API key?**
    A: Yes, for AI recommendations. Set OPENAI_API_KEY in your .env file or enter it in the UI.
    
    **Q: How accurate are the AI recommendations?**
    A: The AI analyzes your actual campaign data and provides data-driven insights. Always validate recommendations with your team and historical performance.
    
    ### 🧠 Model Training
    **Q: What models are available?**
    A: The system supports LightGBM and XGBoost for baseline ROAS forecasting, with AI-powered recommendations for advanced analysis.
    
    **Q: How do I train a model?**
    A: Model training functionality is being integrated with the new gamlens system for improved performance and accuracy.
    
    ### 🔧 Technical
    **Q: What if I get an error?**
    A: Check that your data matches the expected schema, ensure your OpenAI API key is valid, and try refreshing the page.
    
    **Q: Can I export results?**
    A: Yes, you can copy AI responses and download data tables from the interface.
    """)

# Main app
def main():
    st.markdown('<h1 class="main-header">🚀 GameLens AI - Train, Predict, Validate, FAQ</h1>', unsafe_allow_html=True)
    
    # Show selection banner if dataset is selected
    show_selection_banner()
    
    # Tab navigation
    tab1, tab2, tab3, tab4 = st.tabs(["📦 Dataset Management", "🧠 Model Training", "🔮 Predictions & AI", "❓ FAQ"])
    
    with tab1:
        show_datasets_tab()
    
    with tab2:
        show_training_tab()
    
    with tab3:
        show_predictions_tab()
    
    with tab4:
        show_faq_tab()

if __name__ == "__main__":
    main()
