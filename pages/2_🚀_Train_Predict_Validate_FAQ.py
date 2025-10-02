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
                
                # Ingest the file using existing system
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    dataset = ingest_file(temp_path, notes=f"Uploaded via Streamlit")
                
                st.success(f"✅ Successfully ingested: {dataset.canonical_name}")
                st.write(f"- Platform: {dataset.source_platform}")
                st.write(f"- Channel: {dataset.channel}")
                st.write(f"- Game: {dataset.game}")
                st.write(f"- Records: {dataset.records}")
                st.write(f"- Date range start: {dataset.data_start_date}")
                st.write(f"- Upload (End) date: {dataset.data_end_date}")
                
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
    """Show predictions tab with gamlens integration"""
    st.header("🔮 Predictions & AI Recommendations")
    
    # Dataset selection dropdown
    st.subheader("📁 Select Dataset")
    
    # Get available datasets from both gamlens data directory and database
    gamlens_files = []
    if os.path.exists(DATA_DIR):
        gamlens_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    
    # Get datasets from database
    db_datasets = []
    try:
        from glai.db import get_db_session
        from glai.models import Dataset
        db = get_db_session()
        db_datasets = [ds.canonical_name for ds in db.query(Dataset).filter(Dataset.ingest_completed_at.isnot(None)).all()]
        db.close()
    except:
        pass
    
    # Combine and deduplicate
    all_datasets = list(set(gamlens_files + db_datasets))
    
    # Filter out datasets that are known to be incompatible
    compatible_datasets = []
    for dataset_name in all_datasets:
        # Skip datasets with "unknown" in the name as they're likely from old ingestion
        if "unknown" in dataset_name.lower() and dataset_name not in gamlens_files:
            continue
        compatible_datasets.append(dataset_name)
    
    # If no compatible datasets, show all with a warning
    if not compatible_datasets:
        compatible_datasets = all_datasets
        if all_datasets:
            st.warning("⚠️ Some datasets may have compatibility issues. Try selecting a CSV file from the gamlens directory.")
    
    if not compatible_datasets:
        st.warning("No datasets found. Please upload a dataset first in the Dataset Management tab.")
        return
    
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
        
        # Try to load from gamlens data directory first
        if selected_dataset_name in gamlens_files:
            try:
                path = os.path.join(DATA_DIR, selected_dataset_name)
                df = read_csv_strict(path)
                df_kpi, table = add_core_kpis(df)
                st.success(f"✅ Loaded dataset: {selected_dataset_name}")
                dataset_loaded = True
            except Exception as e:
                st.error(f"Error loading from gamlens directory: {str(e)}")
        
        if not dataset_loaded:
            # Fallback to existing dataset loading from database
            try:
                # Find the dataset in the database by name
                from glai.db import get_db_session
                from glai.models import Dataset
                db = get_db_session()
                db_dataset = db.query(Dataset).filter(
                    Dataset.canonical_name == selected_dataset_name,
                    Dataset.ingest_completed_at.isnot(None)
                ).first()
                
                if db_dataset and db_dataset.storage_path and os.path.exists(db_dataset.storage_path):
                    # Load from parquet file
                    import pandas as pd
                    df = pd.read_parquet(db_dataset.storage_path)
                    
                    # Check if the dataset has the required columns for add_core_kpis
                    required_columns = ['game', 'channel', 'platform', 'country', 'date', 'installs', 'cost', 'revenue']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    
                    if missing_columns:
                        st.warning(f"⚠️ Dataset '{selected_dataset_name}' is missing required columns: {missing_columns}")
                        st.info("This dataset was likely ingested with an older schema. Please re-upload it with the correct template.")
                        db.close()
                        return
                    
                    # Try to add KPIs
                    try:
                        df_kpi, table = add_core_kpis(df)
                        st.success(f"✅ Loaded dataset from database: {selected_dataset_name}")
                        dataset_loaded = True
                    except Exception as kpi_error:
                        st.error(f"❌ Error processing dataset KPIs: {str(kpi_error)}")
                        st.info("The dataset structure may be incompatible with the current schema.")
                        db.close()
                        return
                else:
                    st.error(f"❌ Dataset '{selected_dataset_name}' not found in database or file missing")
                    return
                db.close()
            except Exception as e:
                st.error(f"Error loading from database: {str(e)}")
                return
        
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
    col1, col2, col3 = st.columns(3)
    
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
