"""
Unified GameLens AI page: Train, Predict, Validate, FAQ
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import io
import uuid
from datetime import datetime, date
from typing import List, Dict, Any, Optional

# Ensure repository root is on sys.path so `glai` can be imported reliably
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import GameLens modules
from glai.db import init_database, get_db_session
from glai.models import Dataset, ModelVersion, PredictionRun, PlatformEnum
from glai.ingest import ingest_file, get_datasets, get_dataset_by_id, load_dataset_data
from glai.train import train_lgbm_quantile, get_model_versions, get_model_version_by_id, load_model_artifacts
from glai.predict import run_predictions, get_prediction_runs, load_predictions, generate_recommendations
from glai.naming import make_canonical_name
from glai.faq_gpt import get_faq_gpt
from glai.recommend_gpt import get_gpt_recommendations
from glai.raw_source_loader import load_raw_source_dataframe, load_cleaned_dataframe


def load_raw_csv_data(dataset):
    """Load raw CSV/Excel data when normalized data has zeros."""
    try:
        # Prefer cleaned data from n8n; fallback to local raw loader
        cleaned = load_cleaned_dataframe(dataset)
        if cleaned is not None and not cleaned.empty:
            return cleaned
        return load_raw_source_dataframe(dataset)
    except Exception as e:
        print(f"Error loading raw data: {e}")
        return None

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
    """Show current selection banner"""
    st.markdown('<div class="selection-banner">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_dataset = st.session_state.get('selected_dataset')
        if selected_dataset:
            st.write(f"**Selected Dataset:** {selected_dataset.canonical_name}")
        else:
            st.write("**Selected Dataset:** None")
    
    with col2:
        selected_model = st.session_state.get('selected_model')
        if selected_model:
            st.write(f"**Selected Model:** {selected_model.model_name} v{selected_model.version}")
        else:
            st.write("**Selected Model:** None")
    
    with col3:
        if selected_dataset and selected_model:
            st.success("✅ Ready for predictions")
        else:
            st.warning("⚠️ Select dataset and model")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_datasets_tab():
    """Show datasets tab with PostgreSQL integration"""
    st.header("📦 Dataset Management")
    
    # Template download + guide (restored)
    st.subheader("📋 Data Template")
    # Prefer v2 template if present (cleaned rows/line-endings); fallback to v1
    tpl_csv_v2 = "Data_Template_GameLens_AI_v2.csv"
    tpl_csv_v1 = "Data_Template_GameLens_AI.csv"
    tpl_csv = tpl_csv_v2 if os.path.exists(tpl_csv_v2) else tpl_csv_v1
    tpl_md = "DATA_TEMPLATE_GUIDE.md"
    col_tpl1, col_tpl2, col_tpl3 = st.columns([1,1,2])
    with col_tpl1:
        if os.path.exists(tpl_csv):
            with open(tpl_csv, "r") as f:
                st.download_button(
                    label="📥 Download Template CSV",
                    data=f.read(),
                    file_name=os.path.basename(tpl_csv),
                    mime="text/csv"
                )
        else:
            st.info("Template CSV not found in project root.")
    with col_tpl2:
        if os.path.exists(tpl_md):
            with open(tpl_md, "r") as f:
                st.download_button(
                    label="📖 Download Template Guide",
                    data=f.read(),
                    file_name="DATA_TEMPLATE_GUIDE.md",
                    mime="text/markdown"
                )
        else:
            st.info("Guide not found.")
    with col_tpl3:
        if st.checkbox("Show template preview") and os.path.exists(tpl_csv):
            try:
                prev = pd.read_csv(tpl_csv).head(10)
                st.dataframe(prev, use_container_width=True)
            except Exception:
                st.warning("Could not preview template.")
    
    # File upload section
    st.subheader("Upload New Dataset")
    
    uploaded_files = st.file_uploader(
        "Upload CSV or Excel files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Upload files following the Game > Platform > Channel > Countries hierarchy"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Ingest the file
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
    
    # Get datasets from PostgreSQL csv_uploads table
    try:
        from glai.models_pg import get_distinct_source_files, get_data_by_source_file
        
        source_files = get_distinct_source_files()
        
        if not source_files:
            st.info("No datasets found in PostgreSQL. Upload a CSV file using the n8n webhook or upload above.")
            return
        
        # Dataset selection
        selected_source_file = st.selectbox("Select a dataset:", source_files)
        
        if selected_source_file:
            # Get data for selected source file
            csv_uploads = get_data_by_source_file(selected_source_file)
            
            if csv_uploads:
                # Show dataset summary
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Source File:** {selected_source_file}")
                    st.write(f"**Total Records:** {len(csv_uploads):,}")
                    
                    # Get unique values
                    games = set(upload.game for upload in csv_uploads if upload.game)
                    channels = set(upload.channel for upload in csv_uploads if upload.channel)
                    platforms = set(upload.platform for upload in csv_uploads if upload.platform)
                    countries = set(upload.country for upload in csv_uploads if upload.country)
                    
                    st.write(f"**Games:** {', '.join(games) if games else 'N/A'}")
                    st.write(f"**Channels:** {', '.join(channels) if channels else 'N/A'}")
                
                with col2:
                    st.write(f"**Platforms:** {', '.join(platforms) if platforms else 'N/A'}")
                    st.write(f"**Countries:** {', '.join(countries) if countries else 'N/A'}")
                    
                    # Date range
                    dates = [upload.date for upload in csv_uploads if upload.date]
                    if dates:
                        min_date = min(dates)
                        max_date = max(dates)
                        st.write(f"**Date Range:** {min_date} to {max_date}")
                
                # Show sample data
                st.subheader("Sample Data")
                import pandas as pd
                
                # Convert to DataFrame for display
                sample_data = []
                for upload in csv_uploads[:10]:  # Show first 10 rows
                    sample_data.append({
                        'Game': upload.game,
                        'Channel': upload.channel,
                        'Platform': upload.platform,
                        'Country': upload.country,
                        'Date': upload.date,
                        'Installs': upload.installs,
                        'Cost': upload.cost,
                        'Revenue': upload.revenue,
                        'ROAS D7': upload.roas_d7,
                    })
                
                if sample_data:
                    df_sample = pd.DataFrame(sample_data)
                    st.dataframe(df_sample, use_container_width=True)
                    
                    if len(csv_uploads) > 10:
                        st.info(f"Showing first 10 of {len(csv_uploads):,} records")
    
    except Exception as e:
        st.error(f"Error loading datasets from PostgreSQL: {str(e)}")
        st.info("Please ensure the database credentials are correct in .env file")
def show_model_training_tab():
    """Show model training tab"""
    st.header("🤖 Model Training")
    
    # Training controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Configuration")

        # Dataset selection (only completed datasets with valid files)
        datasets = get_datasets()
        valid_datasets = [
            d for d in datasets
            if getattr(d, 'ingest_completed_at', None) is not None and getattr(d, 'storage_path', None)
               and os.path.exists(d.storage_path)
        ]
        if valid_datasets:
            dataset_options = {f"{d.canonical_name} ({d.records} records)": d.id for d in valid_datasets}
            selected_dataset_names = st.multiselect(
                "Select Datasets for Training",
                list(dataset_options.keys()),
                default=[list(dataset_options.keys())[0]] if dataset_options else []
            )
            selected_dataset_ids = [dataset_options[name] for name in selected_dataset_names]
        else:
            st.warning("No completed datasets with available data files. Upload data on the Datasets tab and wait until Status shows Complete.")
            selected_dataset_ids = []

        # Discover available ROAS targets from selected (or all valid) datasets
        def discover_target_days(ds_ids):
            sample_ids = ds_ids if ds_ids else [d.id for d in valid_datasets[:3]]
            days = set()
            debug_info = []
            for did in sample_ids:
                try:
                    ds = get_dataset_by_id(str(did))
                    if not (ds and ds.storage_path and os.path.exists(ds.storage_path)):
                        debug_info.append(f"skip: missing file for {did}")
                        continue
                    # Try to read schema with pyarrow, fallback to pandas; if still empty, read a small DF
                    cols = []
                    err = None
                    try:
                        import pyarrow.parquet as pq  # type: ignore
                        cols = [str(c) for c in pq.ParquetFile(ds.storage_path).schema.names]
                    except Exception as e:
                        err = str(e)
                        try:
                            cols = [str(c) for c in pd.read_parquet(ds.storage_path).columns]
                        except Exception as e2:
                            err = f"{err} | pandas:{e2}"
                    if not cols:
                        try:
                            df_head = pd.read_parquet(ds.storage_path).head(2)
                            cols = [str(c) for c in df_head.columns]
                        except Exception:
                            pass
                    if not cols:
                        debug_info.append(f"no-cols: {ds.storage_path} ({err})")
                        continue
                    import re as _re
                    rcols = []
                    for col in cols:
                        c = col.strip()
                        lc = c.lower()
                        if lc.startswith("roas_d"):
                            rcols.append(col)
                            m = _re.search(r"roas_d\s*(\d+)", lc)
                            if m:
                                days.add(int(m.group(1)))
                    debug_info.append(f"{os.path.basename(ds.storage_path)} → roas cols: {', '.join(rcols[:20])}")
                except Exception as e:
                    debug_info.append(f"error: {e}")
                    continue
            return sorted(days) if days else [], debug_info

        available_days, debug_info = discover_target_days(selected_dataset_ids)
        # Prefer D30 if present; else first available
        default_idx = available_days.index(30) if 30 in available_days else (available_days.index(7) if 7 in available_days else 0)
        target_day = st.selectbox(
            "Target Day",
            available_days if available_days else [30],
            index=default_idx if available_days else 0,
            help="Targets are discovered from ROAS columns in your dataset(s)"
        )
        # Show what was discovered for clarity
        if available_days:
            st.caption(f"Available targets detected: {', '.join(['D'+str(d) for d in available_days])}")
        else:
            st.caption("No ROAS buckets detected from dataset schema; defaulting to D30. Use the checkbox below to show debug info.")
            with st.expander("Debug: detected ROAS columns info"):
                for line in debug_info:
                    st.write(line)
        
        # Model parameters
        st.subheader("Model Parameters")
        learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.05, 0.01)
        max_depth = st.slider("Max Depth", 4, 12, 6)
        n_estimators = st.slider("Number of Estimators", 50, 500, 100)
        
        notes = st.text_area("Notes (optional)", placeholder="Add notes about this model...")
    
    with col2:
        st.subheader("Mode Selection")
        
        mode = st.radio(
            "Send data to model:",
            ["Train", "Predict"],
            help="Train a new model or use existing model for predictions"
        )
        
        if mode == "Train":
            if st.button("🚀 Train Model", type="primary", disabled=not selected_dataset_ids):
                if not selected_dataset_ids:
                    st.error("Please select at least one dataset for training.")
                else:
                    with st.spinner("Training model..."):
                        try:
                            params = {
                                'learning_rate': learning_rate,
                                'max_depth': max_depth,
                                'n_estimators': n_estimators
                            }
                            
                            model_version = train_lgbm_quantile(
                                selected_dataset_ids,
                                target_day,
                                params,
                                notes
                            )
                            
                            st.success(f"✅ Model trained successfully!")
                            st.write(f"Model: {model_version.model_name}")
                            st.write(f"Version: {model_version.version}")
                            st.write(f"Target Day: D{model_version.target_day}")
                            # Keep selection and navigate to Predictions automatically
                            st.session_state['selected_model'] = model_version
                            st.session_state['active_tab'] = "Predictions"
                            st.session_state['nav_message'] = "Model trained. Switched to Predictions."
                            st.rerun()
                            
                            # Show metrics
                            if model_version.metrics_json:
                                metrics = model_version.metrics_json.get('p50', {})
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("R²", f"{metrics.get('r2', 0):.4f}")
                                with col2:
                                    st.metric("MAPE", f"{metrics.get('mape', 0):.4f}")
                                with col3:
                                    st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
                                with col4:
                                    st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
                            
                            # Auto-select the new model
                            st.session_state['selected_model'] = model_version
                            
                        except Exception as e:
                            st.error(f"❌ Training failed: {str(e)}")
        
        elif mode == "Predict":
            # Model selection for predictions
            models = get_model_versions()
            if models:
                model_options = {f"{m.model_name} v{m.version} (D{m.target_day})": m.id for m in models}
                # Add a pseudo-model option for AI recommendations (neutral naming)
                GPT_OPTION_LABEL = "Adaptive AI Recommendations"
                model_options[GPT_OPTION_LABEL] = "__gpt__"
                selected_model_name = st.selectbox(
                    "Select Model for Predictions",
                    list(model_options.keys())
                )
                selected_model_id = model_options[selected_model_name]
                
                # Dataset selection for predictions
                if valid_datasets:
                    pred_dataset_options = {f"{d.canonical_name}": d.id for d in valid_datasets}
                    selected_pred_dataset_name = st.selectbox(
                        "Select Dataset for Predictions",
                        list(pred_dataset_options.keys())
                    )
                    selected_pred_dataset_id = pred_dataset_options[selected_pred_dataset_name]
                    
                    if st.button("🎯 Run Predictions", type="primary"):
                        with st.spinner("Running predictions..."):
                            try:
                                if selected_model_id == "__gpt__":
                                    # Route to Predictions tab with GPT augmentation enabled
                                    st.session_state['selected_model'] = None
                                    st.session_state['selected_dataset'] = get_dataset_by_id(selected_pred_dataset_id)
                                    st.session_state['force_gpt'] = True
                                    st.session_state['active_tab'] = "Predictions"
                                    st.success("✅ Switched to Predictions with GPT recommendations enabled.")
                                    st.rerun()
                                else:
                                    prediction_run = run_predictions(
                                        selected_model_id,
                                        selected_pred_dataset_id,
                                        targets=[target_day]
                                    )
                                    
                                    st.success("✅ Predictions completed!")
                                    st.write(f"Prediction Run ID: {str(prediction_run.id)[:8]}...")
                                    st.write(f"Rows processed: {prediction_run.n_rows}")
                                    
                                    # Auto-select for predictions tab
                                    st.session_state['selected_model'] = get_model_version_by_id(selected_model_id)
                                    st.session_state['selected_dataset'] = get_dataset_by_id(selected_pred_dataset_id)
                                    st.session_state['active_tab'] = "Predictions"
                                    st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Prediction failed: {str(e)}")
            else:
                st.warning("No trained models available. Train a model first.")
    
    # Model history
    st.subheader("Model History")
    
    models = get_model_versions()
    if models:
        model_data = []
        for model in models:
            metrics = model.metrics_json.get('p50', {}) if model.metrics_json else {}
            model_data.append({
                'Model Name': model.model_name,
                'Version': model.version,
                'Target Day': f"D{model.target_day}",
                'Created': model.created_at.strftime("%Y-%m-%d %H:%M"),
                'R²': f"{metrics.get('r2', 0):.4f}",
                'MAPE': f"{metrics.get('mape', 0):.4f}",
                'RMSE': f"{metrics.get('rmse', 0):.4f}",
                'MAE': f"{metrics.get('mae', 0):.4f}",
                'Status': "✅ Ready"
            })
        
        df_models = pd.DataFrame(model_data)
        
        # Display with expandable details
        for i, model in enumerate(models):
            with st.expander(f"{model.model_name} v{model.version} - D{model.target_day}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Created:** {model.created_at.strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"**Target Day:** D{model.target_day}")
                    st.write(f"**Training Datasets:** {len(model.train_dataset_ids)}")
                    
                    if model.notes:
                        st.write(f"**Notes:** {model.notes}")
                
                with col2:
                    if model.metrics_json:
                        metrics = model.metrics_json.get('p50', {})
                        st.write("**Performance Metrics:**")
                        st.write(f"R²: {metrics.get('r2', 0):.4f}")
                        st.write(f"MAPE: {metrics.get('mape', 0):.4f}")
                        st.write(f"RMSE: {metrics.get('rmse', 0):.4f}")
                        st.write(f"MAE: {metrics.get('mae', 0):.4f}")
                
                # Feature importance
                try:
                    artifacts = load_model_artifacts(model)
                    if artifacts.get('feature_importance') is not None:
                        st.write("**Top 10 Features:**")
                        top_features = artifacts['feature_importance'].head(10)
                        st.dataframe(top_features, use_container_width=True)
                except Exception as e:
                    st.write(f"Could not load feature importance: {e}")
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Select Model", key=f"select_model_{i}"):
                        st.session_state['selected_model'] = model
                        st.rerun()
                
                with col2:
                    if st.button("View Details", key=f"details_{i}"):
                        st.session_state['selected_model'] = model
                        st.session_state['active_tab'] = "Predictions"
                        st.rerun()
                
                with col3:
                    if st.button("Delete", key=f"delete_{i}"):
                        st.warning("Delete functionality not implemented yet")
    else:
        st.info("No trained models found. Train a model to see it here.")

def show_predictions_tab():
    """Show predictions tab with PostgreSQL integration"""
    st.header("🔮 Predictions & AI Recommendations")
    
    # Dataset selection dropdown
    st.subheader("📁 Select Dataset")
    
    # Get available datasets from PostgreSQL csv_uploads table
    pg_datasets = []
    try:
        from glai.models_pg import get_distinct_source_files
        pg_datasets = get_distinct_source_files()
    except Exception as e:
        st.error(f"Error connecting to PostgreSQL: {str(e)}")
        st.info("Please ensure the database credentials are correct in .env file")
        return
    
    # Fallback: also check gamlens files if no PostgreSQL data
    gamlens_files = []
    if not pg_datasets and os.path.exists(DATA_DIR):
        gamlens_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    
    # Use PostgreSQL datasets as primary source
    all_datasets = pg_datasets if pg_datasets else gamlens_files
    
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
    
    # Load dataset data from PostgreSQL
    try:
        dataset_loaded = False
        df_kpi = None
        table = None
        
        # Load from PostgreSQL csv_uploads table
        if selected_dataset_name in pg_datasets:
            try:
                from glai.models_pg import get_data_by_source_file
                import pandas as pd
                
                # Get data from PostgreSQL
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
                
                # Convert text columns to numeric where needed
                for col in ['roas_d14', 'roas_d30', 'roas_d60', 'roas_d90', 
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
        
        # Fallback to gamlens files if PostgreSQL fails
        if not dataset_loaded and selected_dataset_name in gamlens_files:
            try:
                path = os.path.join(DATA_DIR, selected_dataset_name)
                df = read_csv_strict(path)
                df_kpi, table = add_core_kpis(df)
                st.success(f"✅ Loaded dataset from gamlens directory: {selected_dataset_name}")
                dataset_loaded = True
            except Exception as e:
                st.error(f"Error loading from gamlens directory: {str(e)}")
        
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
        
