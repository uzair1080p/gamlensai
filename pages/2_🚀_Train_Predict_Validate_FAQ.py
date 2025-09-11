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
    tpl_csv = "Data_Template_GameLens_AI.csv"
    tpl_md = "DATA_TEMPLATE_GUIDE.md"
    col_tpl1, col_tpl2, col_tpl3 = st.columns([1,1,2])
    with col_tpl1:
        if os.path.exists(tpl_csv):
            with open(tpl_csv, "r") as f:
                st.download_button(
                    label="📥 Download Template CSV",
                    data=f.read(),
                    file_name="Data_Template_GameLens_AI.csv",
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
                import pandas as pd
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
        show_actions = st.checkbox("Show Actions", value=True)
    
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
                        st.rerun()
                    
                    if st.button("Use for Predictions", key=f"predict_{i}"):
                        st.session_state['selected_dataset'] = datasets[i]
                        st.session_state['active_tab'] = "Predictions"
                        st.rerun()
                
                st.markdown("---")
        else:
            st.dataframe(df_datasets, use_container_width=True)
    else:
        st.info("No datasets found. Upload some files to get started!")

def show_model_training_tab():
    """Show model training tab"""
    st.header("🤖 Model Training")
    
    # Training controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Configuration")
        
        target_day = st.selectbox(
            "Target Day",
            [15, 30, 45, 90],
            help="Predict ROAS for this day"
        )
        
        # Dataset selection
        datasets = get_datasets()
        if datasets:
            dataset_options = {f"{d.canonical_name} ({d.records} records)": d.id for d in datasets}
            selected_dataset_names = st.multiselect(
                "Select Datasets for Training",
                list(dataset_options.keys()),
                default=[list(dataset_options.keys())[0]] if dataset_options else []
            )
            selected_dataset_ids = [dataset_options[name] for name in selected_dataset_names]
        else:
            st.warning("No datasets available. Upload some data first.")
            selected_dataset_ids = []
        
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
                selected_model_name = st.selectbox(
                    "Select Model for Predictions",
                    list(model_options.keys())
                )
                selected_model_id = model_options[selected_model_name]
                
                # Dataset selection for predictions
                if datasets:
                    pred_dataset_options = {f"{d.canonical_name}": d.id for d in datasets}
                    selected_pred_dataset_name = st.selectbox(
                        "Select Dataset for Predictions",
                        list(pred_dataset_options.keys())
                    )
                    selected_pred_dataset_id = pred_dataset_options[selected_pred_dataset_name]
                    
                    if st.button("🎯 Run Predictions", type="primary"):
                        with st.spinner("Running predictions..."):
                            try:
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
    
    if not selected_model or not selected_dataset:
        st.warning("Please select a model and dataset from the Model Training tab.")
        return
    
    st.subheader("Current Selection")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Model:** {selected_model.model_name} v{selected_model.version}")
        st.write(f"**Target Day:** D{selected_model.target_day}")
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
            
            # Create display dataframe
            display_df = pred_df.copy()
            display_df['Campaign'] = display_df.index + 1
            display_df['Predicted ROAS (p50)'] = display_df['predicted_roas_p50'].round(4)
            display_df['Confidence Interval'] = display_df['confidence_interval'].round(4)
            display_df['p10'] = display_df['predicted_roas_p10'].round(4)
            display_df['p90'] = display_df['predicted_roas_p90'].round(4)
            
            # Add recommendations
            recommendations = generate_recommendations(pred_df)
            action_mapping = {}
            for action, data in recommendations.items():
                for campaign in data['campaigns']:
                    action_mapping[campaign['row_index']] = action.title()
            
            display_df['Action'] = display_df['row_index'].map(action_mapping)
            
            # Select columns for display
            display_columns = ['Campaign', 'Predicted ROAS (p50)', 'p10', 'p90', 'Confidence Interval', 'Action']
            if 'cost' in display_df.columns:
                display_columns.insert(-1, 'cost')
            if 'revenue' in display_df.columns:
                display_columns.insert(-1, 'revenue')
            
            st.dataframe(display_df[display_columns], use_container_width=True)
            
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
    
    with col1:
        game_filter = st.selectbox("Game", ["All"], key="faq_game")
    
    with col2:
        platform_filter = st.selectbox("Platform", ["All"], key="faq_platform")
    
    with col3:
        channel_filter = st.selectbox("Channel", ["All"], key="faq_channel")
    
    with col4:
        country_filter = st.selectbox("Country", ["All"], key="faq_country")
    
    # Custom question input
    st.subheader("Ask a Question")
    custom_question = st.text_input(
        "Ask anything about your ROAS forecasting:",
        placeholder="e.g., What insights can you provide about our campaign performance?"
    )
    
    if custom_question:
        # Generate answer based on current context
        context_info = {
            'model': selected_model.model_name if selected_model else None,
            'dataset': selected_dataset.canonical_name if selected_dataset else None,
            'filters': {
                'game': game_filter,
                'platform': platform_filter,
                'channel': channel_filter,
                'country': country_filter
            }
        }
        
        # Simple fallback answer system
        answer = generate_faq_answer(custom_question, context_info)
        st.write("**Answer:**")
        st.write(answer)
    
    # Predefined questions
    st.subheader("Common Questions")
    
    questions = [
        "What is the current model performance?",
        "Which campaigns should we scale based on predictions?",
        "How accurate are our ROAS predictions?",
        "What are the top drivers of ROAS in our data?",
        "What ROAS should we expect from our campaigns?",
        "How confident should we be in our predictions?",
        "What insights can you provide about our advertising data?"
    ]
    
    for question in questions:
        with st.expander(question):
            context_info = {
                'model': selected_model.model_name if selected_model else None,
                'dataset': selected_dataset.canonical_name if selected_dataset else None,
                'filters': {
                    'game': game_filter,
                    'platform': platform_filter,
                    'channel': channel_filter,
                    'country': country_filter
                }
            }
            
            answer = generate_faq_answer(question, context_info)
            st.write(answer)

def generate_faq_answer(question: str, context: Dict[str, Any]) -> str:
    """Generate FAQ answer based on context"""
    question_lower = question.lower()
    
    if "performance" in question_lower or "accuracy" in question_lower:
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
    
    # Show selection banner
    show_selection_banner()
    
    # Tab navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Datasets", 
        "🤖 Model Training", 
        "📈 Predictions", 
        "✅ Validation", 
        "❓ FAQ"
    ])
    
    with tab1:
        show_datasets_tab()
    
    with tab2:
        show_model_training_tab()
    
    with tab3:
        show_predictions_tab()
    
    with tab4:
        show_validation_tab()
    
    with tab5:
        show_faq_tab()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**GameLens AI v2.0**")
    st.sidebar.markdown("Database + Model Registry + GPT Naming")

if __name__ == "__main__":
    main()
