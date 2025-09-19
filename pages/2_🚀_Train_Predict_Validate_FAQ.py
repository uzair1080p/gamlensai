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
from glai.raw_source_loader import load_raw_source_dataframe


def load_raw_csv_data(dataset):
    """Load raw CSV/Excel data when normalized data has zeros."""
    try:
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
    """Show datasets tab"""
    st.header("📊 Datasets")
    
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
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        platform_filter = st.selectbox(
            "Filter by Platform",
            ["All"] + [p.value for p in PlatformEnum]
        )
    
    with col2:
        game_filter = st.text_input("Filter by Game", placeholder="Enter game name...")
    
    with col3:
        # Always show actions (was toggle before)
        show_actions = True
        
        # Delete all datasets button
        if st.button("🗑️ Delete All Datasets", type="secondary", help="Delete all datasets permanently"):
            st.session_state['deleting_all_datasets'] = True
            st.rerun()
    
    # Get datasets
    datasets = get_datasets(
        platform=platform_filter if platform_filter != "All" else None,
        game=game_filter if game_filter else None
    )
    
    if datasets:
        # Create dataset dataframe for display
        dataset_data = []
        for dataset in datasets:
            dataset_data.append({
                'ID': str(dataset.id)[:8] + "...",
                'Canonical Name': dataset.canonical_name,
                'Platform': dataset.source_platform,
                'Channel': dataset.channel,
                'Game': dataset.game,
                'Records': dataset.records,
                'Start Date': f"{dataset.data_start_date}",
                'Upload (End) Date': f"{dataset.data_end_date}",
                'Ingested': dataset.ingest_started_at.strftime("%Y-%m-%d %H:%M"),
                'Status': "✅ Complete" if dataset.ingest_completed_at else "⏳ Processing"
            })
        
        df_datasets = pd.DataFrame(dataset_data)
        
        # Display with selection
        # Cached Excel exporter to keep downloads snappy
        @st.cache_data(show_spinner=False)
        def _dataset_to_excel_bytes(dataset_id: str) -> bytes:
            df = load_dataset_data(get_dataset_by_id(dataset_id))
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="data")
            return output.getvalue()

        if show_actions:
            # Add selection column
            selected_indices = []
            for i, row in df_datasets.iterrows():
                col1, col2, col3, col4 = st.columns([1, 3, 2, 2])
                
                with col1:
                    if st.button("Select", key=f"select_{i}"):
                        st.session_state['selected_dataset'] = datasets[i]
                        st.rerun()
                
                with col2:
                    st.write(f"**{row['Canonical Name']}**")
                    st.write(f"Platform: {row['Platform']} | Channel: {row['Channel']}")
                
                with col3:
                    st.write(f"Records: {row['Records']}")
                    st.write(f"Status: {row['Status']}")
                
                with col4:
                    if st.button("Use for Training", key=f"train_{i}"):
                        st.session_state['selected_dataset'] = datasets[i]
                        st.session_state['active_tab'] = "Model Training"
                        st.session_state['nav_message'] = "Dataset selected. Open the Model Training tab to continue."
                        st.rerun()
                    
                    if st.button("Use for Predictions", key=f"predict_{i}"):
                        st.session_state['selected_dataset'] = datasets[i]
                        st.session_state['active_tab'] = "Predictions"
                        st.session_state['nav_message'] = "Dataset selected. Open the Predictions tab to continue."
                        st.rerun()
                    
                    # Rename dataset functionality
                    if st.button("Rename", key=f"rename_{i}"):
                        st.session_state['renaming_dataset_id'] = str(datasets[i].id)
                        st.session_state['renaming_dataset_name'] = datasets[i].canonical_name
                        st.rerun()
                    
                    # Delete dataset functionality
                    if st.button("🗑️ Delete", key=f"delete_{i}"):
                        st.session_state['deleting_dataset_id'] = str(datasets[i].id)
                        st.session_state['deleting_dataset_name'] = datasets[i].canonical_name
                        st.rerun()

                    # Row-level one-click download
                    try:
                        xbytes = _dataset_to_excel_bytes(str(datasets[i].id))
                        st.download_button(
                            label="Download",
                            data=xbytes,
                            file_name=f"{datasets[i].canonical_name}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"dl_btn_{i}"
                        )
                    except Exception as _e:
                        st.caption("Download unavailable for this dataset.")
                
                st.markdown("---")
        else:
            st.dataframe(df_datasets, use_container_width=True)

        # Rename dataset dialog
        if 'renaming_dataset_id' in st.session_state:
            st.subheader("📝 Rename Dataset")
            current_name = st.session_state.get('renaming_dataset_name', '')
            new_name = st.text_input(
                "New Dataset Name", 
                value=current_name,
                key="rename_input"
            )
            
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("✅ Save", key="save_rename"):
                    if new_name and new_name != current_name:
                        try:
                            # Update dataset name in database
                            from glai.db import get_db_session
                            from glai.models import Dataset
                            from sqlalchemy.orm import Session
                            
                            db = get_db_session()
                            try:
                                # Convert string ID to UUID
                                import uuid
                                dataset_id = uuid.UUID(st.session_state['renaming_dataset_id'])
                                dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
                                if dataset:
                                    dataset.canonical_name = new_name
                                    db.commit()
                                    st.success(f"✅ Dataset renamed to: {new_name}")
                                    # Clear rename state
                                    del st.session_state['renaming_dataset_id']
                                    del st.session_state['renaming_dataset_name']
                                    st.rerun()
                                else:
                                    st.error("Dataset not found")
                            finally:
                                db.close()
                        except Exception as e:
                            st.error(f"Error renaming dataset: {e}")
                    else:
                        st.warning("Please enter a different name")
            
            with col2:
                if st.button("❌ Cancel", key="cancel_rename"):
                    del st.session_state['renaming_dataset_id']
                    if 'renaming_dataset_name' in st.session_state:
                        del st.session_state['renaming_dataset_name']
                    st.rerun()

        # Delete dataset confirmation dialog
        if 'deleting_dataset_id' in st.session_state:
            st.subheader("🗑️ Delete Dataset")
            dataset_name = st.session_state.get('deleting_dataset_name', 'Unknown')
            st.warning(f"⚠️ **Are you sure you want to delete dataset '{dataset_name}'?**")
            st.error("🚨 **This action cannot be undone!** The dataset and all associated data will be permanently removed.")
            
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("✅ Yes, Delete", key="confirm_delete", type="primary"):
                    try:
                        # Delete dataset from database and filesystem
                        from glai.db import get_db_session
                        from glai.models import Dataset
                        from pathlib import Path
                        
                        db = get_db_session()
                        try:
                            # Convert string ID to UUID
                            import uuid
                            dataset_id = uuid.UUID(st.session_state['deleting_dataset_id'])
                            dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
                            if dataset:
                                # First, delete associated prediction runs
                                from glai.models import PredictionRun
                                prediction_runs = db.query(PredictionRun).filter(PredictionRun.dataset_id == dataset.id).all()
                                for pred_run in prediction_runs:
                                    db.delete(pred_run)
                                
                                # Delete the data file if it exists
                                if dataset.storage_path and Path(dataset.storage_path).exists():
                                    try:
                                        os.remove(dataset.storage_path)
                                        # Also try to remove the parent directory if it's empty
                                        parent_dir = Path(dataset.storage_path).parent
                                        if parent_dir.exists() and not any(parent_dir.iterdir()):
                                            parent_dir.rmdir()
                                    except Exception as e:
                                        st.warning(f"Could not delete data file: {e}")
                                
                                # Delete from database
                                db.delete(dataset)
                                db.commit()
                                st.success(f"✅ Dataset '{dataset_name}' deleted successfully")
                                
                                # Clear delete state
                                del st.session_state['deleting_dataset_id']
                                del st.session_state['deleting_dataset_name']
                                st.rerun()
                            else:
                                st.error("Dataset not found")
                        finally:
                            db.close()
                    except Exception as e:
                        st.error(f"Error deleting dataset: {e}")
            
            with col2:
                if st.button("❌ Cancel", key="cancel_delete"):
                    del st.session_state['deleting_dataset_id']
                    if 'deleting_dataset_name' in st.session_state:
                        del st.session_state['deleting_dataset_name']
                    st.rerun()

        # Delete all datasets confirmation dialog
        if 'deleting_all_datasets' in st.session_state:
            st.subheader("🗑️ Delete All Datasets")
            dataset_count = len(datasets) if datasets else 0
            st.error(f"🚨 **WARNING: This will permanently delete ALL {dataset_count} datasets!**")
            st.warning("⚠️ **This action cannot be undone!** All datasets and their associated data files will be permanently removed from the system.")
            
            if dataset_count > 0:
                st.write("**Datasets to be deleted:**")
                for i, dataset in enumerate(datasets[:10]):  # Show first 10
                    st.write(f"- {dataset.canonical_name}")
                if len(datasets) > 10:
                    st.write(f"... and {len(datasets) - 10} more datasets")
            
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("✅ Yes, Delete All", key="confirm_delete_all", type="primary"):
                    try:
                        # Delete all datasets from database and filesystem
                        from glai.db import get_db_session
                        from glai.models import Dataset
                        from pathlib import Path
                        import uuid
                        
                        db = get_db_session()
                        deleted_count = 0
                        try:
                            all_datasets = db.query(Dataset).all()
                            for dataset in all_datasets:
                                # First, delete associated prediction runs
                                from glai.models import PredictionRun
                                prediction_runs = db.query(PredictionRun).filter(PredictionRun.dataset_id == dataset.id).all()
                                for pred_run in prediction_runs:
                                    db.delete(pred_run)
                                
                                # Delete the data file if it exists
                                if dataset.storage_path and Path(dataset.storage_path).exists():
                                    try:
                                        os.remove(dataset.storage_path)
                                        # Also try to remove the parent directory if it's empty
                                        parent_dir = Path(dataset.storage_path).parent
                                        if parent_dir.exists() and not any(parent_dir.iterdir()):
                                            parent_dir.rmdir()
                                    except Exception as e:
                                        st.warning(f"Could not delete data file for {dataset.canonical_name}: {e}")
                                
                                # Delete from database
                                db.delete(dataset)
                                deleted_count += 1
                            
                            db.commit()
                            st.success(f"✅ Successfully deleted {deleted_count} datasets")
                            
                            # Clear delete state
                            del st.session_state['deleting_all_datasets']
                            st.rerun()
                        finally:
                            db.close()
                    except Exception as e:
                        st.error(f"Error deleting datasets: {e}")
            
            with col2:
                if st.button("❌ Cancel", key="cancel_delete_all"):
                    del st.session_state['deleting_all_datasets']
                    st.rerun()

        # Download selected dataset as Excel (always rendered below list)
        ds_to_download = st.session_state.get('selected_dataset')
        if ds_to_download:
            try:
                xls_bytes = _dataset_to_excel_bytes(str(ds_to_download.id))
                st.download_button(
                    "📥 Download Selected Dataset (Excel)",
                    data=xls_bytes,
                    file_name=f"{ds_to_download.canonical_name}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_selected_dataset"
                )
            except Exception as e:
                st.caption(f"Could not prepare Excel download: {e}")
    else:
        st.info("No datasets found. Upload some files to get started!")

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
    """Show predictions tab"""
    st.header("📈 Predictions")
    
    selected_model = st.session_state.get('selected_model')
    selected_dataset = st.session_state.get('selected_dataset')
    force_gpt = st.session_state.pop('force_gpt', False)
    
    if not selected_dataset:
        st.warning("Please select a dataset from the Model Training tab.")
        return
    
    st.subheader("Current Selection")
    col1, col2 = st.columns(2)
    with col1:
        if selected_model:
            st.write(f"**Model:** {selected_model.model_name} v{selected_model.version}")
            st.write(f"**Target Day:** D{selected_model.target_day}")
        elif force_gpt:
            st.write("**Model:** Adaptive AI Recommendations")
            st.write("**Target Day:** N/A (augmentation only)")
    with col2:
        st.write(f"**Dataset:** {selected_dataset.canonical_name}")
        st.write(f"**Records:** {selected_dataset.records}")
    
    # Run predictions button
    if st.button("🎯 Run New Predictions", type="primary"):
        with st.spinner("Running predictions..."):
            try:
                prediction_run = run_predictions(
                    str(selected_model.id),
                    str(selected_dataset.id),
                    targets=[selected_model.target_day]
                )
                
                st.success("✅ Predictions completed!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
    
    # Show existing predictions
    st.subheader("Prediction Results")
    
    prediction_runs = []
    if selected_model:
        prediction_runs = get_prediction_runs(
            model_version_id=str(selected_model.id),
            dataset_id=str(selected_dataset.id)
        )
    
    if prediction_runs:
        latest_run = prediction_runs[0]  # Most recent
        
        # Load predictions
        try:
            pred_df = load_predictions(latest_run)
            
            # Summary statistics
            st.subheader("Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Campaigns", len(pred_df))
            
            with col2:
                mean_roas = pred_df['predicted_roas_p50'].mean()
                st.metric("Mean Predicted ROAS", f"{mean_roas:.4f}")
            
            with col3:
                above_1_0 = (pred_df['predicted_roas_p50'] >= 1.0).sum()
                percentage = (above_1_0 / len(pred_df)) * 100
                st.metric("Campaigns ≥ 1.0 ROAS", f"{above_1_0} ({percentage:.1f}%)")
            
            with col4:
                mean_confidence = pred_df['confidence_interval'].mean()
                st.metric("Mean Confidence Width", f"{mean_confidence:.4f}")
            
            # ROAS projections table
            st.subheader("ROAS Projections")
            
            # Create summary dataframe for client-facing table
            base_df = load_dataset_data(selected_dataset)
            summary_df = pred_df.copy()
            summary_df['Campaign'] = summary_df.index + 1
            # CPI and Installs from dataset
            if 'installs' in base_df.columns:
                summary_df['Installs'] = base_df['installs'].values
            else:
                summary_df['Installs'] = 0
            if all(col in base_df.columns for col in ['cost','installs']):
                with pd.option_context('mode.use_inf_as_na', True):
                    summary_df['CPI ($)'] = (base_df['cost'] / base_df['installs']).replace([pd.NA, pd.NaT], 0).fillna(0).round(2)
            else:
                summary_df['CPI ($)'] = 0.0
            # ROAS D7/D14 from dataset if present
            summary_df['ROAS D7 (%)'] = (base_df['roas_d7']*100).round(1) if 'roas_d7' in base_df.columns else pd.Series([None]*len(base_df))
            summary_df['ROAS D14 (%)'] = (base_df['roas_d14']*100).round(1) if 'roas_d14' in base_df.columns else pd.Series([None]*len(base_df))
            # ROI 100% By (Day): simple check against target day
            summary_df['ROI 100% By (Day)'] = summary_df['predicted_roas_p50'].apply(lambda x: selected_model.target_day if pd.notna(x) and x>=1.0 else None)
            # Retention D7
            if 'retention_d7' in base_df.columns:
                summary_df['Retention D7 (%)'] = (base_df['retention_d7']*100).round(1)
            elif 'retention_d7_rate' in base_df.columns:
                summary_df['Retention D7 (%)'] = (base_df['retention_d7_rate']*100).round(1)
            else:
                summary_df['Retention D7 (%)'] = None
            # ARPU needed for break-even approximated as CPI
            summary_df['ARPU Needed for Break-even ($)'] = summary_df['CPI ($)']
            # Add recommendations
            recommendations = generate_recommendations(pred_df)
            action_mapping = {}
            for action, data in recommendations.items():
                for campaign in data['campaigns']:
                    action_mapping[campaign['row_index']] = action.title()
            summary_df['Recommended Action'] = summary_df['row_index'].map(action_mapping)
            # Select and order columns for display
            client_columns = ['Campaign','CPI ($)','Installs','ROAS D7 (%)','ROAS D14 (%)','ROI 100% By (Day)','Retention D7 (%)','ARPU Needed for Break-even ($)','Recommended Action']
            st.dataframe(summary_df[client_columns], use_container_width=True)

            # Optional: Augment with GPT-powered recommendations
            default_gpt = True if force_gpt else False
            use_gpt = st.checkbox("Augment with GPT recommendations", value=default_gpt,
                                   help="Calls GPT to provide an additional 'GPT Action' and rationale per campaign.")
            gpt_map = {}
            if use_gpt:
                with st.spinner("Calling GPT for campaign-level recommendations..."):
                    gpt_map = get_gpt_recommendations(pred_df)
                if gpt_map:
                    display_df['GPT Action'] = display_df['row_index'].map(lambda i: gpt_map.get(int(i), {}).get('action'))
                    display_df['GPT Rationale'] = display_df['row_index'].map(lambda i: gpt_map.get(int(i), {}).get('rationale'))
                    if 'budget_change_pct' in next(iter(gpt_map.values()), {}):
                        display_df['GPT Budget %'] = display_df['row_index'].map(lambda i: gpt_map.get(int(i), {}).get('budget_change_pct'))
            
            # Build compact dataset for FAQ context using raw CSV data
            try:
                # Try to get raw CSV data for FAQ context
                raw_df = load_raw_csv_data(selected_dataset)
                if raw_df is not None and not raw_df.empty:
                    # Build FAQ data from raw CSV
                    compact_for_faq = []
                    for i, row in raw_df.iterrows():
                        # Parse cost and revenue from raw strings (using normalized column names)
                        cost_str = str(row.get("cost", "0")).replace("$", "").replace(",", "").strip()
                        revenue_str = str(row.get("revenue", "0")).replace("$", "").replace(",", "").strip()
                        
                        try:
                            cost_val = float(cost_str) if cost_str and cost_str != "0" and cost_str != "-" else 0
                            revenue_val = float(revenue_str) if revenue_str and revenue_str != "0" and revenue_str != "-" else 0
                        except (ValueError, TypeError):
                            cost_val = 0
                            revenue_val = 0
                        
                        installs = row.get("installs", 0)
                        cpi = cost_val / installs if installs > 0 else 0
                        
                        # Calculate ROAS from cost/revenue if ROAS columns are missing
                        roas_d7 = row.get("roas_d7", 0)
                        roas_d14 = row.get("roas_d14", 0)
                        if pd.isna(roas_d7) or roas_d7 == 0:
                            roas_d7 = (revenue_val / cost_val) if cost_val > 0 else 0
                        if pd.isna(roas_d14) or roas_d14 == 0:
                            roas_d14 = (revenue_val / cost_val) if cost_val > 0 else 0
                        
                        compact_row = {
                            "Campaign": i + 1,
                            "CPI ($)": round(cpi, 2),
                            "Installs": installs,
                            "ROAS D7 (%)": round(roas_d7 * 100, 2) if pd.notna(roas_d7) else 0,
                            "ROAS D14 (%)": round(roas_d14 * 100, 2) if pd.notna(roas_d14) else 0,
                            "ROI 100% By (Day)": None,
                            "Retention D7 (%)": row.get("retention_rate_d7", None),
                            "ARPU Needed for Break-even ($)": round(cost_val / installs, 2) if installs > 0 else 0,
                            "Recommended Action": "Unknown"  # No GPT actions for model-based predictions
                        }
                        compact_for_faq.append(compact_row)
                else:
                    # Fallback to processed data
                    compact_for_faq = summary_df[client_columns].head(100).to_dict(orient='records')
            except Exception as e:
                st.warning(f"Could not load raw CSV for FAQ context: {e}")
                # Fallback to processed data
                compact_for_faq = summary_df[client_columns].head(100).to_dict(orient='records')
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # ROAS distribution
                fig = px.histogram(
                    pred_df, 
                    x='predicted_roas_p50',
                    nbins=30,
                    title="Predicted ROAS Distribution",
                    labels={'predicted_roas_p50': 'Predicted ROAS (p50)', 'count': 'Number of Campaigns'}
                )
                fig.add_vline(x=1.0, line_dash="dash", line_color="red", annotation_text="ROAS = 1.0")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence intervals
                sample_size = min(20, len(pred_df))
                sample_df = pred_df.head(sample_size)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(sample_size)),
                    y=sample_df['predicted_roas_p50'],
                    mode='markers',
                    name='Predicted ROAS (p50)',
                    marker=dict(color='blue', size=8)
                ))
                fig.add_trace(go.Scatter(
                    x=list(range(sample_size)),
                    y=sample_df['predicted_roas_p90'],
                    mode='lines',
                    name='Upper Bound (p90)',
                    line=dict(color='lightblue', width=0)
                ))
                fig.add_trace(go.Scatter(
                    x=list(range(sample_size)),
                    y=sample_df['predicted_roas_p10'],
                    mode='lines',
                    fill='tonexty',
                    name='Lower Bound (p10)',
                    line=dict(color='lightblue', width=0)
                ))
                fig.update_layout(
                    title=f"Sample Predictions with Confidence Intervals (D{selected_model.target_day})",
                    xaxis_title="Campaign Index",
                    yaxis_title="ROAS"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations summary
            st.subheader("Recommendations Summary")
            
            rec_col1, rec_col2, rec_col3, rec_col4 = st.columns(4)
            
            with rec_col1:
                scale_count = recommendations['scale']['count']
                scale_share = recommendations['scale']['spend_share']
                st.metric("Scale", f"{scale_count} campaigns", f"{scale_share:.1f}% spend")
            
            with rec_col2:
                maintain_count = recommendations['maintain']['count']
                maintain_share = recommendations['maintain']['spend_share']
                st.metric("Maintain", f"{maintain_count} campaigns", f"{maintain_share:.1f}% spend")
            
            with rec_col3:
                reduce_count = recommendations['reduce']['count']
                reduce_share = recommendations['reduce']['spend_share']
                st.metric("Reduce", f"{reduce_count} campaigns", f"{reduce_share:.1f}% spend")
            
            with rec_col4:
                cut_count = recommendations['cut']['count']
                cut_share = recommendations['cut']['spend_share']
                st.metric("Cut", f"{cut_count} campaigns", f"{cut_share:.1f}% spend")
            
            # Download predictions
            csv = pred_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name=f"predictions_{selected_model.model_name}_v{selected_model.version}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Error loading predictions: {str(e)}")
    else:
        # GPT-only path: no local predictions available but GPT was requested
        if force_gpt:
            try:
                st.info("Using AI recommendations without local predictions.")
                base_df = load_dataset_data(selected_dataset)
                # Build a payload aligned with recommender expectations (include more columns)
                gpt_df = pd.DataFrame({'row_index': range(len(base_df))})
                for col in base_df.columns:
                    if col not in gpt_df.columns:
                        try:
                            gpt_df[col] = base_df[col].values
                        except Exception:
                            pass
                # Data is already normalized in the dataset loader, no need to re-parse

                # Send raw CSV data directly to GPT without any preprocessing
                try:
                    # Try to load raw CSV data first
                    raw_df = load_raw_csv_data(selected_dataset)
                    if raw_df is not None:
                        # Use raw CSV data directly
                        gpt_df = raw_df.copy()
                        st.info("📄 Using raw CSV data for GPT analysis")
                    else:
                        st.warning("⚠️ Raw CSV not found, using normalized data")
                except Exception as e:
                    st.warning(f"⚠️ Could not load raw CSV: {e}, using normalized data")

                # Call GPT recommender with raw data
                with st.spinner("Calling GPT for campaign-level recommendations..."):
                    gpt_map = get_gpt_recommendations(gpt_df)
                # Display table - create from base_df to ensure data integrity
                gpt_display = base_df.copy()
                gpt_display['Campaign'] = gpt_display.index + 1
                # Add row_index for GPT mapping
                gpt_display['row_index'] = gpt_display.index
                
                # Check if cost/revenue are all zeros (old normalized data) and try to load raw CSV
                if 'cost' in gpt_display.columns and gpt_display['cost'].sum() == 0:
                    st.warning("⚠️ Detected old normalized data with zeros. Loading raw CSV data...")
                    try:
                        # Try to load raw CSV data
                        raw_df = load_raw_csv_data(selected_dataset)
                        if raw_df is not None and 'cost' in raw_df.columns:
                            gpt_display['Cost'] = raw_df['cost'].astype(str)
                            gpt_display['Revenue'] = raw_df['revenue'].astype(str) if 'revenue' in raw_df.columns else gpt_display['revenue'].astype(str)
                            st.success("✅ Loaded raw CSV data with currency strings")
                        else:
                            gpt_display['Cost'] = gpt_display['cost'].astype(str)
                            gpt_display['Revenue'] = gpt_display['revenue'].astype(str)
                    except Exception as e:
                        st.error(f"Could not load raw data: {e}")
                        gpt_display['Cost'] = gpt_display['cost'].astype(str)
                        gpt_display['Revenue'] = gpt_display['revenue'].astype(str)
                else:
                    # As requested, do not show 'Cost' and 'Revenue' columns for now
                    columns_to_drop = []
                    if 'cost' in gpt_display.columns:
                        columns_to_drop.append('cost')
                    if 'revenue' in gpt_display.columns:
                        columns_to_drop.append('revenue')
                    # Also check for 'Cost' and 'Revenue' if they were created by other logic
                    if 'Cost' in gpt_display.columns:
                        columns_to_drop.append('Cost')
                    if 'Revenue' in gpt_display.columns:
                        columns_to_drop.append('Revenue')
                    
                    if columns_to_drop:
                        gpt_display = gpt_display.drop(columns=list(set(columns_to_drop)), errors='ignore')
                # Build client-facing summary table for GPT-only path using raw CSV data
                try:
                    # Try to get raw CSV data for summary table
                    raw_df = load_raw_csv_data(selected_dataset)
                    if raw_df is not None and not raw_df.empty:
                        # Debug: Show which CSV file was loaded
                        st.info(f"📄 Loaded raw CSV data: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
                        st.info(f"📄 CSV columns: {list(raw_df.columns)}")
                        if 'cost' in raw_df.columns:
                            st.info(f"📄 Cost sample: {raw_df['cost'].head().tolist()}")
                        if 'revenue' in raw_df.columns:
                            st.info(f"📄 Revenue sample: {raw_df['revenue'].head().tolist()}")
                        # Use raw CSV data for summary table
                        summary_df = pd.DataFrame({'Campaign': range(1, len(raw_df) + 1)})
                        summary_df['row_index'] = range(len(raw_df))
                        
                        # Parse cost and revenue from raw strings
                        cost_values = []
                        revenue_values = []
                        for i, row in raw_df.iterrows():
                            # Parse cost
                            cost_str = str(row.get("cost", "0")).replace("$", "").replace(",", "").strip()
                            try:
                                cost_val = float(cost_str) if cost_str and cost_str != "0" and cost_str != "-" else 0
                            except (ValueError, TypeError):
                                cost_val = 0
                            cost_values.append(cost_val)
                            
                            # Parse revenue
                            revenue_str = str(row.get("revenue", "0")).replace("$", "").replace(",", "").strip()
                            try:
                                revenue_val = float(revenue_str) if revenue_str and revenue_str != "0" and revenue_str != "-" else 0
                            except (ValueError, TypeError):
                                revenue_val = 0
                            revenue_values.append(revenue_val)
                        
                        # Calculate metrics from raw data
                        installs = raw_df.get("installs", 0).fillna(0)
                        summary_df['Installs'] = installs
                        summary_df['CPI ($)'] = [cost_val / install_val if install_val > 0 else 0 for cost_val, install_val in zip(cost_values, installs)]
                        
                        # ROAS calculations
                        roas_d7_values = []
                        roas_d14_values = []
                        for i, row in raw_df.iterrows():
                            # Check if ROAS columns exist in raw data
                            roas_d7 = row.get("roas_d7", None)
                            roas_d14 = row.get("roas_d14", None)
                            
                            if pd.notna(roas_d7) and roas_d7 != 0:
                                roas_d7_values.append(roas_d7 * 100)
                            elif cost_values[i] > 0:
                                roas_d7_values.append((revenue_values[i] / cost_values[i]) * 100)
                            else:
                                roas_d7_values.append(0)
                                
                            if pd.notna(roas_d14) and roas_d14 != 0:
                                roas_d14_values.append(roas_d14 * 100)
                            elif cost_values[i] > 0:
                                roas_d14_values.append((revenue_values[i] / cost_values[i]) * 100)
                            else:
                                roas_d14_values.append(0)
                        
                        summary_df['ROAS D7 (%)'] = roas_d7_values
                        summary_df['ROAS D14 (%)'] = roas_d14_values
                        
                        # Retention D7
                        if 'retention_d7' in raw_df.columns:
                            summary_df['Retention D7 (%)'] = (raw_df['retention_d7'] * 100).round(1)
                        elif 'retention_d7_rate' in raw_df.columns:
                            summary_df['Retention D7 (%)'] = (raw_df['retention_d7_rate'] * 100).round(1)
                        else:
                            summary_df['Retention D7 (%)'] = None
                        
                        # ROI 100% By (Day) not available without local model
                        summary_df['ROI 100% By (Day)'] = None
                        
                        # ARPU needed for break-even approximated as CPI
                        summary_df['ARPU Needed for Break-even ($)'] = summary_df['CPI ($)']
                        
                        # Map GPT actions
                        summary_df['Recommended Action'] = summary_df['row_index'].map(lambda i: gpt_map.get(int(i), {}).get('action'))
                        
                        st.success("✅ Using raw CSV data for summary table")
                    else:
                        # Fallback to normalized data
                        summary_df = gpt_display.copy()
                        summary_df['Installs'] = gpt_display['installs'] if 'installs' in gpt_display.columns else 0
                        if 'cost' in gpt_display.columns and 'installs' in gpt_display.columns:
                            with pd.option_context('mode.use_inf_as_na', True):
                                summary_df['CPI ($)'] = (gpt_display.get('cost', 0) / gpt_display.get('installs', 1)).replace([pd.NA, pd.NaT], 0).fillna(0).round(2)
                        else:
                            summary_df['CPI ($)'] = 0.0
                        summary_df['ROAS D7 (%)'] = (gpt_display['roas_d7']*100).round(1) if 'roas_d7' in gpt_display.columns else None
                        summary_df['ROAS D14 (%)'] = (gpt_display['roas_d14']*100).round(1) if 'roas_d14' in gpt_display.columns else None
                        summary_df['ROI 100% By (Day)'] = None
                        if 'retention_d7' in gpt_display.columns:
                            summary_df['Retention D7 (%)'] = (gpt_display['retention_d7']*100).round(1)
                        elif 'retention_d7_rate' in gpt_display.columns:
                            summary_df['Retention D7 (%)'] = (gpt_display['retention_d7_rate']*100).round(1)
                        else:
                            summary_df['Retention D7 (%)'] = None
                        summary_df['ARPU Needed for Break-even ($)'] = summary_df['CPI ($)']
                        summary_df['Recommended Action'] = summary_df['row_index'].map(lambda i: gpt_map.get(int(i), {}).get('action'))
                        st.warning("⚠️ Using normalized data for summary table")
                except Exception as e:
                    st.error(f"Error building summary table: {e}")
                    # Fallback to basic structure
                    summary_df = pd.DataFrame({'Campaign': range(1, len(gpt_display) + 1)})
                    summary_df['CPI ($)'] = 0.0
                    summary_df['Installs'] = 0
                    summary_df['ROAS D7 (%)'] = None
                    summary_df['ROAS D14 (%)'] = None
                    summary_df['ROI 100% By (Day)'] = None
                    summary_df['Retention D7 (%)'] = None
                    summary_df['ARPU Needed for Break-even ($)'] = 0.0
                    summary_df['Recommended Action'] = None
                client_columns = ['Campaign','CPI ($)','Installs','ROAS D7 (%)','ROAS D14 (%)','ROI 100% By (Day)','Retention D7 (%)','ARPU Needed for Break-even ($)','Recommended Action']
                st.dataframe(summary_df[client_columns], use_container_width=True)
                # Build compact dataset for FAQ prompt using raw CSV data
                try:
                    # Try to get raw CSV data for FAQ context
                    raw_df = load_raw_csv_data(selected_dataset)
                    if raw_df is not None and not raw_df.empty:
                        # Build FAQ data from raw CSV
                        compact_for_faq = []
                        for i, row in raw_df.iterrows():
                            # Parse cost and revenue from raw strings (using normalized column names)
                            cost_str = str(row.get("cost", "0")).replace("$", "").replace(",", "").strip()
                            revenue_str = str(row.get("revenue", "0")).replace("$", "").replace(",", "").strip()
                            
                            try:
                                cost_val = float(cost_str) if cost_str and cost_str != "0" and cost_str != "-" else 0
                                revenue_val = float(revenue_str) if revenue_str and revenue_str != "0" and revenue_str != "-" else 0
                            except (ValueError, TypeError):
                                cost_val = 0
                                revenue_val = 0
                            
                            installs = row.get("installs", 0)
                            cpi = cost_val / installs if installs > 0 else 0
                            
                            # Calculate ROAS from cost/revenue if ROAS columns are missing
                            roas_d7 = row.get("roas_d7", 0)
                            roas_d14 = row.get("roas_d14", 0)
                            if pd.isna(roas_d7) or roas_d7 == 0:
                                roas_d7 = (revenue_val / cost_val) if cost_val > 0 else 0
                            if pd.isna(roas_d14) or roas_d14 == 0:
                                roas_d14 = (revenue_val / cost_val) if cost_val > 0 else 0
                            
                            compact_row = {
                                "Campaign": i + 1,
                                "CPI ($)": round(cpi, 2),
                                "Installs": installs,
                                "ROAS D7 (%)": round(roas_d7 * 100, 2) if pd.notna(roas_d7) else 0,
                                "ROAS D14 (%)": round(roas_d14 * 100, 2) if pd.notna(roas_d14) else 0,
                                "ROI 100% By (Day)": None,
                                "Retention D7 (%)": row.get("retention_rate_d7", None),
                                "ARPU Needed for Break-even ($)": round(cost_val / installs, 2) if installs > 0 else 0,
                                "Recommended Action": gpt_map.get(i, {}).get("action", "Unknown")
                            }
                            compact_for_faq.append(compact_row)
                    else:
                        # Fallback to processed data
                        compact_for_faq = summary_df[client_columns].head(100).to_dict(orient='records')
                except Exception as e:
                    st.warning(f"Could not load raw CSV for FAQ context: {e}")
                    # Fallback to processed data
                    compact_for_faq = summary_df[client_columns].head(100).to_dict(orient='records')

                with st.expander("Show AI payload preview"):
                    st.write("GPT DataFrame shape:", gpt_df.shape)
                    st.write("GPT DataFrame columns:", gpt_df.columns.tolist())
                    st.write("Cost column sample:", gpt_df['cost'].head().tolist() if 'cost' in gpt_df.columns else "No cost column")
                    st.write("Revenue column sample:", gpt_df['revenue'].head().tolist() if 'revenue' in gpt_df.columns else "No revenue column")
                    if 'cost_raw' in gpt_df.columns:
                        st.write("Cost raw sample:", gpt_df['cost_raw'].head().tolist())
                    if 'revenue_raw' in gpt_df.columns:
                        st.write("Revenue raw sample:", gpt_df['revenue_raw'].head().tolist())
                    st.write(gpt_df.head(10))

                # Auto-answer client FAQs using current context + GPT outputs
                st.subheader("AI Answers to Client FAQs")
                faq_gpt = get_faq_gpt()
                context = faq_gpt.generate_context_summary(
                    selected_model=None,
                    selected_dataset=selected_dataset,
                    filters=None
                )
                # Inject a lightweight summary of recommendations into context
                context["ai_recommendations"] = {
                    "count": int(len(gpt_map)) if gpt_map else 0
                    # Removed actions_breakdown to prevent "Cut" bias in FAQ responses
                }
                # Include compact dataset for the prompt
                context["dataset_compact"] = compact_for_faq
                # Mark that we have AI predictions available
                context["has_predictions"] = True
                context["model_type"] = "Adaptive AI Recommendations (GPT-powered)"
                
                # Debug: Show what's being sent to FAQ
                st.info(f"🔍 FAQ Context Debug:")
                st.info(f"  - GPT recommendations count: {len(gpt_map) if gpt_map else 0}")
                st.info(f"  - Dataset compact rows: {len(compact_for_faq) if compact_for_faq else 0}")
                if compact_for_faq and len(compact_for_faq) > 0:
                    st.info(f"  - First row CPI: {compact_for_faq[0].get('CPI ($)', 'N/A')}")
                    st.info(f"  - First row ROAS D7: {compact_for_faq[0].get('ROAS D7 (%)', 'N/A')}")
                    st.info(f"  - First row Action: {compact_for_faq[0].get('Recommended Action', 'N/A')}")
                client_questions = [
                    "When will ROI of 100% be achieved on this channel? D15? D30? D90?",
                    "Should we continue running this campaign or pause it?",
                    "What is the projected ROAS if we keep spending at the same pace?",
                    "Which channels are driving the highest quality players long-term?",
                    "How does actual performance compare to AI forecast from Day 3?"
                ]
                for q in client_questions:
                    with st.expander(q):
                        with st.spinner("Generating..."):
                            ans = faq_gpt.generate_faq_answer(q, context)
                        st.write(ans)
                # Deep debug dropdown for GPT payload/response
                with st.expander("🔎 GPT debug (payload, prompts, errors)"):
                    # Be defensive in case the running server has an older class without this helper
                    try:
                        get_debug = getattr(faq_gpt, "get_last_debug", None)
                        debug = get_debug() if callable(get_debug) else {}
                        if not debug and not callable(get_debug):
                            st.caption("Debug API not available in this session. Restart the app to load the latest code if needed.")
                    except Exception as _dbg_e:
                        debug = {}
                        st.error(f"Debug retrieval failed: {_dbg_e}")
                    st.write("Client available:", debug.get('client_available', True))
                    st.write("Model type:", context.get('model_type'))
                    st.write("Has predictions:", context.get('has_predictions'))
                    st.write("Context keys:", debug.get('context_keys'))
                    st.write("Context lengths:", debug.get('context_lengths'))
                    st.write("System prompt:")
                    st.code(debug.get('system_prompt', '')[:4000])
                    st.write("User prompt:")
                    st.code(debug.get('user_prompt', '')[:4000])
                    if 'error' in debug:
                        st.error(f"Error: {debug['error'].get('message')}")
                    if 'response_preview' in debug:
                        st.write("Response preview:")
                        st.code(debug.get('response_preview', '')[:4000])
            except Exception as e:
                st.error(f"❌ GPT-only recommendations failed: {e}")
        else:
            st.info("No predictions found. Run predictions to see results here.")

def show_validation_tab():
    """Show validation tab"""
    st.header("✅ Validation")
    
    selected_model = st.session_state.get('selected_model')
    selected_dataset = st.session_state.get('selected_dataset')
    
    if not selected_model or not selected_dataset:
        st.warning("Please select a model and dataset from the Model Training tab.")
        return
    
    # Filters
    st.subheader("Filters")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        game_filter = st.selectbox("Game", ["All"])
    
    with col2:
        platform_filter = st.selectbox("Platform", ["All"])
    
    with col3:
        channel_filter = st.selectbox("Channel", ["All"])
    
    with col4:
        country_filter = st.selectbox("Country", ["All"])
    
    # Load data for validation
    try:
        # Load dataset data
        df = load_dataset_data(selected_dataset)
        
        # Load predictions
        prediction_runs = get_prediction_runs(
            model_version_id=str(selected_model.id),
            dataset_id=str(selected_dataset.id)
        )
        
        if not prediction_runs:
            st.warning("No predictions found. Run predictions first.")
            return
        
        pred_df = load_predictions(prediction_runs[0])
        
        # Merge with actual data if available
        target_col = f'roas_d{selected_model.target_day}'
        if target_col in df.columns:
            # Create validation dataframe
            val_df = pred_df.copy()
            val_df['actual_roas'] = df[target_col].iloc[val_df['row_index']].values
            val_df['abs_error'] = (val_df['actual_roas'] - val_df['predicted_roas_p50']).abs()
            val_df['ape'] = (val_df['abs_error'] / (val_df['actual_roas'] + 1e-8)) * 100
            
            # Validation metrics
            st.subheader("Validation Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                mape = val_df['ape'].mean()
                st.metric("MAPE", f"{mape:.4f}")
            
            with col2:
                mae = val_df['abs_error'].mean()
                st.metric("MAE", f"{mae:.4f}")
            
            with col3:
                rmse = np.sqrt((val_df['abs_error'] ** 2).mean())
                st.metric("RMSE", f"{rmse:.4f}")
            
            with col4:
                r2 = 1 - (val_df['abs_error'] ** 2).sum() / ((val_df['actual_roas'] - val_df['actual_roas'].mean()) ** 2).sum()
                st.metric("R²", f"{r2:.4f}")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Actual vs Predicted
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=val_df['actual_roas'],
                    y=val_df['predicted_roas_p50'],
                    mode='markers',
                    name='Predictions',
                    marker=dict(color='blue', opacity=0.6)
                ))
                
                # Perfect prediction line
                min_val = min(val_df['actual_roas'].min(), val_df['predicted_roas_p50'].min())
                max_val = max(val_df['actual_roas'].max(), val_df['predicted_roas_p50'].max())
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title="Actual vs Predicted ROAS",
                    xaxis_title="Actual ROAS",
                    yaxis_title="Predicted ROAS"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Error distribution
                fig = px.histogram(
                    val_df,
                    x='abs_error',
                    nbins=30,
                    title="Absolute Error Distribution",
                    labels={'abs_error': 'Absolute Error', 'count': 'Number of Campaigns'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed validation table
            st.subheader("Detailed Validation Results")
            
            display_cols = ['row_index', 'actual_roas', 'predicted_roas_p50', 'abs_error', 'ape']
            if 'cost' in val_df.columns:
                display_cols.append('cost')
            if 'revenue' in val_df.columns:
                display_cols.append('revenue')
            
            st.dataframe(val_df[display_cols], use_container_width=True)
            
            # Download validation results
            csv = val_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Validation CSV",
                data=csv,
                file_name=f"validation_{selected_model.model_name}_v{selected_model.version}.csv",
                mime="text/csv"
            )
            
        else:
            st.warning(f"Actual ROAS data not available for D{selected_model.target_day}")
            st.info("Validation requires actual ROAS data in the dataset.")
            
    except Exception as e:
        st.error(f"❌ Error loading validation data: {str(e)}")

def show_faq_tab():
    """Show FAQ tab"""
    st.header("❓ FAQ")
    
    selected_model = st.session_state.get('selected_model')
    selected_dataset = st.session_state.get('selected_dataset')
    
    # Filters
    st.subheader("Context Filters")
    col1, col2, col3, col4 = st.columns(4)
    
    # Get actual data for filters
    datasets = get_datasets()
    models = get_model_versions()
    
    # Extract unique values for filters
    games = ["All"] + sorted(list(set([d.game for d in datasets if d.game])))
    platforms = ["All"] + sorted(list(set([d.source_platform.value if hasattr(d.source_platform, 'value') else str(d.source_platform) for d in datasets if d.source_platform])))
    channels = ["All"] + sorted(list(set([d.channel for d in datasets if d.channel])))
    countries = ["All"] + sorted(list(set([country for d in datasets if d.countries for country in d.countries])))
    
    with col1:
        game_filter = st.selectbox("Game", games, key="faq_game")
    
    with col2:
        platform_filter = st.selectbox("Platform", platforms, key="faq_platform")
    
    with col3:
        channel_filter = st.selectbox("Channel", channels, key="faq_channel")
    
    with col4:
        country_filter = st.selectbox("Country", countries, key="faq_country")
    
    # Custom question input
    st.subheader("Ask a Question")
    custom_question = st.text_input(
        "Ask anything about your ROAS forecasting:",
        placeholder="e.g., What insights can you provide about our campaign performance?"
    )
    
    if custom_question:
        # Generate comprehensive context for GPT
        faq_gpt = get_faq_gpt()
        context = faq_gpt.generate_context_summary(
            selected_model=selected_model,
            selected_dataset=selected_dataset,
            filters={
                'game': game_filter,
                'platform': platform_filter,
                'channel': channel_filter,
                'country': country_filter
            }
        )
        
        # Generate GPT-powered answer
        with st.spinner("🤖 Generating intelligent answer..."):
            answer = faq_gpt.generate_faq_answer(custom_question, context)
        
        st.write("**Answer:**")
        st.write(answer)
    
    # Predefined questions from client FAQ
    st.subheader("Common Questions")
    
    # Client's actual FAQ questions organized by category
    faq_categories = {
        "Performance & ROI": [
            "When will ROI of 100% be achieved on this channel? D15? D30? D90?",
            "Should we continue running this campaign or pause it?",
            "What is the projected ROAS if we keep spending at the same pace?",
            "Which channels are driving the highest quality players long-term?",
            "How does actual performance compare to AI forecast from Day 3?"
        ],
        "User Acquisition & Cost": [
            "What CPI do we need to hit to achieve profitability by D30/D90?",
            "Which campaign is overspending without delivering value?",
            "What's the optimal budget allocation across channels to maximize ROAS?",
            "Which geo is giving us the lowest CPI with strong retention?",
            "Which ad network/creative is driving the highest LTV users?"
        ],
        "Retention & Engagement": [
            "What retention rate do we need at D7/D15 to hit D30 ROAS goals?",
            "Which levels are causing the biggest drop-offs in player progression?",
            "What's the predicted churn rate of players from this cohort?",
            "If retention improves by X%, what impact will it have on revenue?",
            "Which player segments are most likely to become high-value users?"
        ],
        "Revenue & Monetization": [
            "What ARPU do we need by D15/D30 to achieve break-even?",
            "Which country is monetizing best (highest ARPPU)?",
            "Are players engaging more with IAP or ad monetization?",
            "What changes in ad placement or IAP pricing can boost revenue?",
            "If ARPU increases by $0.10, how will it affect D90 ROAS?"
        ],
        "Scaling & Strategy": [
            "Which geo is ready to scale aggressively?",
            "Which campaigns should we cut immediately?",
            "If we double UA spend, what's the projected long-term ROI?",
            "What's the safest budget increase to avoid overspending on bad cohorts?",
            "Which channels are future-proof vs. short-term wins?"
        ]
    }
    
    # Display questions by category
    for category, questions in faq_categories.items():
        st.markdown(f"**{category}**")
        for question in questions:
            with st.expander(question):
                # Generate comprehensive context for GPT
                faq_gpt = get_faq_gpt()
                context = faq_gpt.generate_context_summary(
                    selected_model=selected_model,
                    selected_dataset=selected_dataset,
                    filters={
                        'game': game_filter,
                        'platform': platform_filter,
                        'channel': channel_filter,
                        'country': country_filter
                    }
                )
                
                # Generate GPT-powered answer
                with st.spinner("🤖 Generating intelligent answer..."):
                    answer = faq_gpt.generate_faq_answer(question, context)
                
                st.write(answer)
        st.markdown("---")

def main():
    """Generate FAQ answer based on context"""
    question_lower = question.lower()
    
    # Performance & ROI questions
    if "roi" in question_lower and ("100%" in question or "d15" in question_lower or "d30" in question_lower or "d90" in question_lower):
        if context['model'] and context['dataset']:
            return f"To determine when ROI reaches 100% on this channel, run predictions using model '{context['model']}' on dataset '{context['dataset']}' in the Predictions tab. The system will show projected ROAS over time with confidence intervals to identify when break-even is achieved."
        else:
            return "Please select a model and dataset to get ROI projections. Go to the Model Training tab to train or select a model, then run predictions to see when 100% ROI will be achieved."
    
    elif "continue" in question_lower and ("campaign" in question_lower or "pause" in question_lower):
        if context['model'] and context['dataset']:
            return f"Campaign continuation recommendations are available in the Predictions tab using model '{context['model']}'. The system categorizes campaigns as Scale, Maintain, Reduce, or Cut based on predicted ROAS and confidence intervals."
        else:
            return "Please select a model and dataset to get campaign recommendations. Use the Model Training tab to make your selections, then check the Predictions tab for specific guidance."
    
    elif "projected roas" in question_lower or "spending at the same pace" in question_lower:
        if context['model'] and context['dataset']:
            return f"Projected ROAS at current spending pace is available in the Predictions tab using model '{context['model']}' on dataset '{context['dataset']}'. The system provides p10, p50, and p90 predictions to show expected performance ranges."
        else:
            return "Please select a model and dataset to get ROAS projections. Use the Model Training tab to make your selections, then run predictions to see expected performance."
    
    # User Acquisition & Cost questions
    elif "cpi" in question_lower and ("profitability" in question_lower or "d30" in question_lower or "d90" in question_lower):
        if context['model'] and context['dataset']:
            return f"Target CPI for profitability can be calculated using model '{context['model']}' on dataset '{context['dataset']}'. Run predictions in the Predictions tab to see the relationship between CPI and projected ROAS, helping identify the maximum CPI for profitable campaigns."
        else:
            return "Please select a model and dataset to analyze CPI requirements. Use the Model Training tab to make your selections, then run predictions to see CPI vs ROAS relationships."
    
    elif "overspending" in question_lower or "without delivering value" in question_lower:
        if context['model'] and context['dataset']:
            return f"Campaigns that are overspending without delivering value will be identified in the Predictions tab using model '{context['model']}'. Look for campaigns with high spend but low predicted ROAS and wide confidence intervals."
        else:
            return "Please select a model and dataset to identify overspending campaigns. Use the Model Training tab to make your selections, then check the Predictions tab for campaign performance analysis."
    
    # Retention & Engagement questions
    elif "retention rate" in question_lower and ("d7" in question_lower or "d15" in question_lower or "d30" in question_lower):
        if context['model'] and context['dataset']:
            return f"Required retention rates for D30 ROAS goals can be analyzed using model '{context['model']}' on dataset '{context['dataset']}'. Check the Model Training tab for feature importance to see how retention impacts ROAS predictions."
        else:
            return "Please select a model and dataset to analyze retention requirements. Use the Model Training tab to train a model and view feature importance, showing how retention rates affect ROAS predictions."
    
    elif "level" in question_lower and ("drop-off" in question_lower or "progression" in question_lower):
        if context['model'] and context['dataset']:
            return f"Level progression analysis is available using model '{context['model']}' on dataset '{context['dataset']}'. Check the Model Training tab for feature importance to see which level events most impact ROAS predictions."
        else:
            return "Please select a model and dataset to analyze level progression. Use the Model Training tab to train a model and view feature importance, showing which levels are most critical for ROAS."
    
    # Revenue & Monetization questions
    elif "arpu" in question_lower and ("break-even" in question_lower or "d15" in question_lower or "d30" in question_lower):
        if context['model'] and context['dataset']:
            return f"Target ARPU for break-even can be calculated using model '{context['model']}' on dataset '{context['dataset']}'. Run predictions in the Predictions tab to see the relationship between revenue metrics and projected ROAS."
        else:
            return "Please select a model and dataset to analyze ARPU requirements. Use the Model Training tab to make your selections, then run predictions to see revenue vs ROAS relationships."
    
    elif "monetizing best" in question_lower or "highest arppu" in question_lower:
        if context['model'] and context['dataset']:
            return f"Country monetization analysis is available using model '{context['model']}' on dataset '{context['dataset']}'. Use the context filters above to select specific countries, then run predictions to compare monetization performance."
        else:
            return "Please select a model and dataset to analyze country monetization. Use the context filters above to select countries, then run predictions to compare performance."
    
    # Scaling & Strategy questions
    elif "scale aggressively" in question_lower or "ready to scale" in question_lower:
        if context['model'] and context['dataset']:
            return f"Scaling recommendations are available in the Predictions tab using model '{context['model']}' on dataset '{context['dataset']}'. Look for campaigns with high predicted ROAS and narrow confidence intervals as candidates for aggressive scaling."
        else:
            return "Please select a model and dataset to get scaling recommendations. Use the Model Training tab to make your selections, then check the Predictions tab for specific guidance."
    
    elif "cut immediately" in question_lower or "campaigns should we cut" in question_lower:
        if context['model'] and context['dataset']:
            return f"Campaign cutting recommendations are available in the Predictions tab using model '{context['model']}' on dataset '{context['dataset']}'. Look for campaigns with low predicted ROAS and wide confidence intervals as candidates for immediate cuts."
        else:
            return "Please select a model and dataset to get cutting recommendations. Use the Model Training tab to make your selections, then check the Predictions tab for specific guidance."
    
    # General fallbacks
    elif "performance" in question_lower or "accuracy" in question_lower:
        if context['model']:
            return f"Based on the selected model '{context['model']}', you can view detailed performance metrics in the Predictions tab. The model provides R², MAPE, RMSE, and MAE scores to evaluate prediction accuracy."
        else:
            return "Please select a model to view performance metrics. Go to the Model Training tab to train or select a model."
    
    elif "scale" in question_lower or "campaigns" in question_lower:
        if context['model'] and context['dataset']:
            return f"Based on your selected model '{context['model']}' and dataset '{context['dataset']}', check the Predictions tab for specific campaign recommendations. The system categorizes campaigns as Scale, Maintain, Reduce, or Cut based on predicted ROAS and confidence intervals."
        else:
            return "Please select both a model and dataset to get campaign recommendations. Use the Model Training tab to make your selections."
    
    elif "drivers" in question_lower or "insights" in question_lower:
        if context['model']:
            return f"For insights about ROAS drivers, check the Model Training tab where you can view feature importance for model '{context['model']}'. This shows which factors most influence ROAS predictions."
        else:
            return "Feature importance and ROAS drivers are available after training a model. Go to the Model Training tab to train a model and view insights."
    
    elif "expect" in question_lower or "roas" in question_lower:
        if context['model'] and context['dataset']:
            return f"Expected ROAS predictions are available in the Predictions tab using model '{context['model']}' on dataset '{context['dataset']}'. The system provides p10, p50, and p90 predictions with confidence intervals."
        else:
            return "ROAS predictions require both a trained model and dataset. Use the Model Training tab to select or train a model, then run predictions."
    
    elif "confident" in question_lower:
        return "Confidence in predictions is measured through confidence intervals (p10 to p90 range). Narrower intervals indicate higher confidence. Check the Predictions tab to see confidence metrics for your specific model and dataset."
    
    else:
        return "I can help you with questions about model performance, campaign recommendations, ROAS predictions, and data insights. Please select a model and dataset for more specific answers, or ask a more specific question."

def main():
    """Main application"""
    # Header
    st.markdown('<h1 class="main-header">🚀 GameLens AI - Train, Predict, Validate, FAQ</h1>', unsafe_allow_html=True)
    if 'nav_message' in st.session_state:
        st.info(st.session_state.pop('nav_message'))
    
    # Show selection banner
    show_selection_banner()
    
    # Sidebar-controlled navigation (programmatic)
    tabs = ["Datasets", "Model Training", "Predictions", "Validation", "FAQ"]
    if 'active_tab' not in st.session_state or st.session_state['active_tab'] not in tabs:
        st.session_state['active_tab'] = "Datasets"
    selected_idx = tabs.index(st.session_state['active_tab'])
    chosen = st.sidebar.radio("Navigation", tabs, index=selected_idx)
    st.session_state['active_tab'] = chosen
    
    if chosen == "Datasets":
        show_datasets_tab()
    elif chosen == "Model Training":
        show_model_training_tab()
    elif chosen == "Predictions":
        show_predictions_tab()
    elif chosen == "Validation":
        show_validation_tab()
    else:
        show_faq_tab()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**GameLens AI v2.0**")
    st.sidebar.markdown("Database + Model Registry + GPT Naming")

if __name__ == "__main__":
    main()
