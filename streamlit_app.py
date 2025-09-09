import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import re
import gc
from datetime import datetime
from typing import Optional

# Try to import psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Add src to path for imports
sys.path.append('src')

# Import GameLens modules
from utils.data_loader import GameLensDataLoader
from utils.memory_efficient_feature_engineering import MemoryEfficientFeatureEngineer
from utils.memory_efficient_roas_forecaster import MemoryEfficientROASForecaster

# LLM service with server compatibility check
import os
import sys

LLM_AVAILABLE = False
GameLensLLMService = None

# Check if we're on a server that might have Bus error issues
def is_server_environment():
    """Detect if we're running on a server that might have Bus error issues"""
    try:
        # Check for common server indicators
        if os.path.exists('/proc/version'):
            with open('/proc/version', 'r') as f:
                version_info = f.read().lower()
                if 'ubuntu' in version_info or 'linux' in version_info:
                    return True
        
        # Check for cloud server indicators
        if os.path.exists('/etc/cloud') or os.path.exists('/sys/class/dmi/id/product_name'):
            return True
            
        # Check if running as root (common on servers)
        if os.geteuid() == 0:
            return True
            
        return False
    except:
        return False

# Try to import LLM service with enhanced error handling
try:
    from utils.llm_service import GameLensLLMService
    LLM_AVAILABLE = True
    print("✅ LLM service loaded successfully")
except Exception as e:
    print(f"⚠️ LLM service not available: {e}")
    LLM_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="GameLens AI - ROAS Forecasting Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple authentication
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# Demo credentials
DEMO_USERNAME = "demo"
DEMO_PASSWORD = "demo123"

# Memory monitoring function
def get_memory_usage():
    """Get current memory usage"""
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            return f"{memory_mb:.1f} MB"
        except:
            return "Unknown"
    return "psutil not available"

def log_memory_usage(stage: str):
    """Log memory usage at different stages"""
    memory_usage = get_memory_usage()
    print(f"Memory usage at {stage}: {memory_usage}")
    gc.collect()  # Force garbage collection

# Authentication UI
if not st.session_state['authenticated']:
    st.markdown('<h1 class="main-header">🎮 GameLens AI - ROAS Forecasting Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Phase 1: Multi-Platform ROAS Forecasting with Unity Ads & Mistplay")
    
    st.markdown("---")
    st.subheader("🔐 Login Required")
    
    col1, col2 = st.columns(2)
    with col1:
        username = st.text_input("Username", value="demo")
    with col2:
        password = st.text_input("Password", type="password", value="demo123")
    
    if st.button("Login", type="primary"):
        if username == DEMO_USERNAME and password == DEMO_PASSWORD:
            st.session_state['authenticated'] = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")
    
    st.info("Demo credentials: Username: demo, Password: demo123")
    st.stop()

# Main application (only runs if authenticated)
else:
    # Logout button in sidebar
    if st.sidebar.button("Logout"):
        st.session_state['authenticated'] = False
        st.rerun()

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
</style>
""", unsafe_allow_html=True)

    @st.cache_data
    def load_data():
        """Load and cache data"""
        try:
            log_memory_usage("before data loading")
            data_loader = GameLensDataLoader()
            all_data = data_loader.load_all_data()
            log_memory_usage("after loading all data")
            combined_data = data_loader.combine_platform_data(all_data)
            log_memory_usage("after combining data")
            return combined_data, data_loader
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None

    @st.cache_data
    def create_features(combined_data):
        """Create and cache features"""
        try:
            log_memory_usage("before feature engineering")
            feature_engineer = MemoryEfficientFeatureEngineer()
            features_df = feature_engineer.create_features(combined_data)
            log_memory_usage("after feature engineering")
            return features_df, feature_engineer
        except Exception as e:
            st.error(f"Error creating features: {e}")
            return None, None

    def show_data_overview(combined_data):
        """Display data overview"""
        st.header("📊 Data Overview")
        
        # Data summary
        col1, col2, col3, col4 = st.columns(4)
        
        total_files = sum(len(df) for df in combined_data.values() if not df.empty)
        platforms = set()
        for df in combined_data.values():
            if not df.empty and 'platform' in df.columns:
                platforms.update(df['platform'].unique())
        
        with col1:
            st.metric("Total Data Files", len([df for df in combined_data.values() if not df.empty]))
        with col2:
            st.metric("Total Records", total_files)
        with col3:
            st.metric("Platforms", len(platforms))
        with col4:
            st.metric("Data Types", len([k for k, v in combined_data.items() if not v.empty]))
        
        # Platform distribution
        st.subheader("Platform Distribution")
        platform_data = []
        for data_type, df in combined_data.items():
            if not df.empty and 'platform' in df.columns:
                for platform in df['platform'].unique():
                    platform_data.append({
                        'Data Type': data_type.replace('_', ' ').title(),
                        'Platform': platform,
                        'Records': len(df[df['platform'] == platform])
                    })
        
        if platform_data:
            platform_df = pd.DataFrame(platform_data)
            fig = px.bar(platform_df, x='Platform', y='Records', color='Data Type',
                        title="Data Distribution by Platform and Type")
            st.plotly_chart(fig, use_container_width=True)
        
        # Data quality metrics
        st.subheader("Data Quality Metrics")
        quality_metrics = []
        for data_type, df in combined_data.items():
            if not df.empty:
                quality_metrics.append({
                    'Data Type': data_type.replace('_', ' ').title(),
                    'Records': len(df),
                    'Missing Values (%)': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                    'Columns': len(df.columns)
                })
        
        if quality_metrics:
            quality_df = pd.DataFrame(quality_metrics)
            st.dataframe(quality_df, use_container_width=True)

        # Level progression quick analytics
        if 'level_progression' in combined_data and not combined_data['level_progression'].empty:
            st.subheader("Level Progression Snapshot")
            lvl_df = combined_data['level_progression']
            # Heuristics: find level columns
            level_cols = [c for c in lvl_df.columns if str(c).lower().startswith('level')]
            if level_cols:
                # Compute simple summary
                summary = {
                    'rows': len(lvl_df),
                    'level_columns': len(level_cols)
                }
                st.write(summary)

    def show_feature_engineering(features_df, feature_engineer):
        """Display feature engineering results"""
        st.header("🔧 Feature Engineering")
        
        if features_df is None:
            st.error("No features available. Please check data loading.")
            return
        
        # Feature summary
        feature_summary = feature_engineer.get_feature_summary(features_df)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Features", feature_summary['total_features'])
        with col2:
            st.metric("Numeric Features", feature_summary['numeric_features'])
        with col3:
            st.metric("Categorical Features", feature_summary['categorical_features'])
        with col4:
            st.metric("Total Samples", feature_summary['total_samples'])
        
        # Feature types breakdown
        st.subheader("Feature Types Breakdown")
        feature_types = feature_summary['feature_types']
        
        fig = px.pie(
            values=list(feature_types.values()),
            names=list(feature_types.keys()),
            title="Feature Distribution by Type"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlation heatmap
        st.subheader("Feature Correlation Heatmap")
        numeric_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_features) > 1:
            correlation_matrix = features_df[numeric_features].corr()
            
            # Select top correlated features
            top_features = correlation_matrix.abs().sum().sort_values(ascending=False).head(20).index
            top_corr = correlation_matrix.loc[top_features, top_features]
            
            fig = px.imshow(
                top_corr,
                title="Top 20 Feature Correlations",
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            st.plotly_chart(fig, use_container_width=True)

    def show_model_training(features_df):
        """Display model training interface"""
        st.header("🤖 Model Training")
        
        if features_df is None:
            st.error("No features available for training.")
            return
        
        # Model configuration
        col1, col2 = st.columns(2)
        
        with col1:
            # Target selection
            roas_cols = [col for col in features_df.columns if 'roas_d' in col]
            if roas_cols:
                target_col = st.selectbox("Select Target ROAS Day", roas_cols)
                target_day = int(re.search(r'd(\d+)', target_col).group(1))
            else:
                st.error("No ROAS columns found in data.")
                return
            
            # Model parameters (optimized for 32GB RAM)
            st.subheader("Model Parameters (Optimized for 32GB RAM)")
            n_estimators = st.slider("Number of Estimators", 50, 500, 100)  # Higher default for better quality
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.05, 0.01)  # Lower for better learning
            max_depth = st.slider("Max Depth", 4, 12, 6)  # Higher default for complex patterns
        
        with col2:
            # Training options
            st.subheader("Training Options")
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
            cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
            random_state = st.number_input("Random State", value=42, min_value=0)
        
        # Prepare training data
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model..."):
                try:
                    # Remove rows with missing target
                    valid_data = features_df.dropna(subset=[target_col])
                    
                    if len(valid_data) == 0:
                        st.error("No valid data for training after removing missing targets.")
                        return
                    
                    # Prepare features and target (prevent data leakage)
                    exclude_cols = ['platform', 'subdirectory', 'data_type', target_col]
                    candidate_cols = [col for col in valid_data.columns if col not in exclude_cols]

                    def is_future_signal(col: str) -> bool:
                        try:
                            # ROAS features: only allow strictly earlier than target
                            if 'roas_d' in col:
                                m = re.search(r'roas_d(\d+)', col)
                                if m:
                                    return int(m.group(1)) >= target_day
                            # Retention features: only allow < target_day
                            if 'retention_' in col:
                                m = re.search(r'retention.*?(\d+)', col)
                                if m:
                                    return int(m.group(1)) >= target_day
                            # Level features often end with numbers or have early_ prefix
                            if 'level' in col:
                                m = re.search(r'(?:level[^\d]*)(\d+)$', col)
                                if m:
                                    return int(m.group(1)) >= target_day
                            return False
                        except Exception:
                            return False

                    feature_cols = [c for c in candidate_cols if not is_future_signal(c)]
                    
                    X = valid_data[feature_cols].fillna(0)
                    y = valid_data[target_col]
                    
                    # Initialize and train model
                    log_memory_usage("before model training")
                    forecaster = MemoryEfficientROASForecaster()
                    
                    # Train model with memory monitoring
                    models = forecaster.train_model(
                        X, y, 
                        quantiles=[0.1, 0.5, 0.9],
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        random_state=random_state
                    )
                    log_memory_usage("after model training")
                    
                    # Store model in session state
                    st.session_state['forecaster'] = forecaster
                    st.session_state['X'] = X
                    st.session_state['y'] = y
                    st.session_state['feature_cols'] = feature_cols
                    st.session_state['target_col'] = target_col
                    st.session_state['target_day'] = target_day
                    st.session_state['model_params'] = {
                        'n_estimators': n_estimators,
                        'learning_rate': learning_rate,
                        'max_depth': max_depth,
                        'cv_folds': cv_folds,
                        'test_size': test_size,
                    }
                    
                    st.success(f"✅ Model trained successfully! Target: D{target_day} ROAS")
                    
                    # Show model info
                    feature_importance_df = forecaster.get_feature_importance(top_n=10)
                    if not feature_importance_df.empty:
                        st.subheader("Top 10 Feature Importance")
                        st.dataframe(feature_importance_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error training model: {e}")

    def show_predictions():
        """Display predictions and analysis"""
        st.header("📈 Predictions & Analysis")
        
        if 'forecaster' not in st.session_state:
            st.warning("Please train a model first.")
            return
        
        forecaster = st.session_state['forecaster']
        X = st.session_state['X']
        y = st.session_state['y']
        target_day = st.session_state['target_day']
        
        # Check dataset size and warn if using subset
        if len(X) > 5000:
            st.info(f"📊 Large dataset detected ({len(X):,} samples). Using subset of 5,000 samples for predictions to ensure responsive performance.")
        
        # Make predictions with memory monitoring and timeout protection
        log_memory_usage("before predictions")
        try:
            with st.spinner("Making predictions on dataset (this may take a moment)..."):
                import time
                start_time = time.time()
                predictions = forecaster.predict_with_confidence(X)
                end_time = time.time()
                st.success(f"✅ Predictions completed in {end_time - start_time:.2f} seconds")
            log_memory_usage("after predictions")
        except Exception as e:
            st.error(f"Error making predictions: {e}")
            st.warning("Please try training the model again or check your data.")
            return
        
        # Model performance metrics
        try:
            with st.spinner("Evaluating model performance..."):
                metrics = forecaster.evaluate_model(X, y)
        except Exception as e:
            st.error(f"Error evaluating model: {e}")
            st.warning("Using default metrics.")
            metrics = {'r2': 0, 'rmse': 0, 'mape': 0, 'mae': 0, 'confidence_coverage': 0}
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² Score", f"{metrics['r2']:.4f}")
        with col2:
            st.metric("MAPE", f"{metrics['mape']:.4f}")
        with col3:
            st.metric("RMSE", f"{metrics['rmse']:.4f}")
        with col4:
            st.metric("MAE", f"{metrics['mae']:.4f}")
        
        # Show confidence interval coverage if available
        if 'confidence_coverage' in metrics:
            st.metric("Confidence Coverage", f"{metrics['confidence_coverage']:.4f}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Actual vs Predicted
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y, y=predictions['roas_prediction'],
                mode='markers',
                name='Predictions',
                marker=dict(color='blue', opacity=0.6)
            ))
            fig.add_trace(go.Scatter(
                x=[y.min(), y.max()], y=[y.min(), y.max()],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                title=f"Actual vs Predicted ROAS (D{target_day})",
                xaxis_title="Actual ROAS",
                yaxis_title="Predicted ROAS"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Residuals
            residuals = y - predictions['roas_prediction']
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=predictions['roas_prediction'], y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(color='green', opacity=0.6)
            ))
            fig.update_layout(
                title=f"Residuals Plot",
                xaxis_title="Predicted ROAS",
                yaxis_title="Residuals"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Confidence intervals
        if 'roas_pred_q0.1' in predictions.columns and 'roas_pred_q0.9' in predictions.columns:
            st.subheader("Prediction Confidence Intervals")
            
            # Sample of predictions with confidence intervals
            sample_size = min(20, len(predictions['roas_prediction']))
            sample_indices = np.random.choice(len(predictions['roas_prediction']), sample_size, replace=False)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(sample_size)),
                y=predictions['roas_prediction'].iloc[sample_indices],
                mode='markers',
                name='Predicted ROAS',
                marker=dict(color='blue', size=8)
            ))
            fig.add_trace(go.Scatter(
                x=list(range(sample_size)),
                y=predictions['roas_pred_q0.9'].iloc[sample_indices],
                mode='lines',
                name='Upper Bound (90%)',
                line=dict(color='lightblue', width=0)
            ))
            fig.add_trace(go.Scatter(
                x=list(range(sample_size)),
                y=predictions['roas_pred_q0.1'].iloc[sample_indices],
                mode='lines',
                fill='tonexty',
                name='Lower Bound (10%)',
                line=dict(color='lightblue', width=0)
            ))
            fig.update_layout(
                title=f"Sample Predictions with Confidence Intervals (D{target_day})",
                xaxis_title="Sample Index",
                yaxis_title="ROAS"
            )
            st.plotly_chart(fig, use_container_width=True)

    def show_recommendations():
        """Display recommendations"""
        st.header("💡 Recommendations")
        
        if 'forecaster' not in st.session_state:
            st.warning("Please train a model first.")
            return
        
        forecaster = st.session_state['forecaster']
        X = st.session_state['X']
        y = st.session_state.get('y')
        feature_cols = st.session_state['feature_cols']
        target_day = st.session_state['target_day']
        
        # Recommendation parameters
        col1, col2 = st.columns(2)
        
        with col1:
            target_roas = st.number_input("Target ROAS", value=0.5, min_value=0.0, step=0.1)
            confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.8, 0.05)
        
        with col2:
            num_recommendations = st.number_input("Number of Recommendations", value=10, min_value=1, max_value=50)
        
        if st.button("🎯 Generate Recommendations", type="primary"):
            with st.spinner("Generating recommendations..."):
                try:
                    # Make predictions
                    predictions = forecaster.predict_with_confidence(X)
                    
                    # Generate recommendations
                    recommendations = forecaster.generate_recommendations(
                        X, y, target_roas
                    )
                    
                    # Display recommendations
                    st.subheader(f"Campaign Recommendations (Target ROAS: {target_roas})")
                    
                    # Convert to list of dictionaries for easier iteration
                    rec_list = recommendations.head(num_recommendations).to_dict('records')
                    
                    for i, rec in enumerate(rec_list):
                        with st.expander(f"Campaign {i+1} - {rec['recommendation']}"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Predicted ROAS", f"{rec['predicted_roas']:.4f}")
                            with col2:
                                st.metric("Confidence Level", rec['confidence_level'])
                            with col3:
                                # Color code the recommendation
                                if 'Scale' in rec['recommendation']:
                                    st.markdown('<div class="metric-card success-metric">Scale 📈</div>', unsafe_allow_html=True)
                                elif 'Maintain' in rec['recommendation']:
                                    st.markdown('<div class="metric-card warning-metric">Maintain ⚖️</div>', unsafe_allow_html=True)
                                elif 'Reduce' in rec['recommendation']:
                                    st.markdown('<div class="metric-card warning-metric">Reduce 📉</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown('<div class="metric-card danger-metric">Cut ❌</div>', unsafe_allow_html=True)
                            
                            # Show confidence interval if available
                            if 'confidence_lower' in rec and 'confidence_upper' in rec:
                                st.write(f"**Confidence Interval:** [{rec['confidence_lower']:.4f}, {rec['confidence_upper']:.4f}]")
                            
                            # Show ROAS gap
                            if 'roas_gap' in rec:
                                st.write(f"**ROAS Gap:** {rec['roas_gap']:.4f}")
                    
                    # Export recommendations
                    if st.button("📥 Export Recommendations"):
                        rec_df = pd.DataFrame(recommendations)
                        csv = rec_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"roas_recommendations_d{target_day}.csv",
                            mime="text/csv"
                        )
                    
                except Exception as e:
                    st.error(f"Error generating recommendations: {e}")

    def show_level_progression(combined_data):
        """Analytics focused on level progression quality by channel/platform/game."""
        st.header("🧩 Level Progression Analytics")
        lvl_df = combined_data.get('level_progression') if isinstance(combined_data, dict) else None
        if lvl_df is None or lvl_df.empty:
            st.info("No level progression data loaded.")
            return

        # Identify columns
        meta_cols = ['platform', 'channel', 'network', 'source', 'game', 'title', 'app', 'country', 'geo']
        meta_cols = [c for c in meta_cols if c in lvl_df.columns]
        level_cols = [c for c in lvl_df.columns if str(c).lower().startswith('level')]

        if not level_cols:
            st.warning("No level columns detected (columns starting with 'level').")
            st.dataframe(lvl_df.head(50), use_container_width=True)
            return

        # Filters
        c1, c2, c3 = st.columns(3)
        game_col = next((c for c in ['game', 'title', 'app'] if c in lvl_df.columns), None)
        plat_col = next((c for c in ['platform', 'source', 'network'] if c in lvl_df.columns), None)
        chan_col = next((c for c in ['channel', 'network'] if c in lvl_df.columns), None)
        sel_game = c1.selectbox("Game", ["All"] + sorted(lvl_df[game_col].dropna().unique().tolist()) if game_col else ["All"])
        sel_plat = c2.selectbox("Platform", ["All"] + sorted(lvl_df[plat_col].dropna().unique().tolist()) if plat_col else ["All"])
        sel_chan = c3.selectbox("Channel", ["All"] + sorted(lvl_df[chan_col].dropna().unique().tolist()) if chan_col else ["All"])

        scoped = lvl_df.copy()
        if game_col and sel_game != "All":
            scoped = scoped[scoped[game_col] == sel_game]
        if plat_col and sel_plat != "All":
            scoped = scoped[scoped[plat_col] == sel_plat]
        if chan_col and sel_chan != "All":
            scoped = scoped[scoped[chan_col] == sel_chan]

        # Compute max level achieved row-wise and drop-off curve
        import numpy as np
        level_array = scoped[level_cols].replace([np.inf, -np.inf], np.nan)
        max_level = level_array.apply(lambda r: r.last_valid_index(), axis=1)
        # last_valid_index returns column name; transform to numeric order index
        lvl_order = {name: idx for idx, name in enumerate(level_cols, start=1)}
        max_level_num = max_level.map(lambda c: lvl_order.get(c, 0))

        st.subheader("Quality KPIs")
        k1, k2 = st.columns(2)
        k1.metric("Median Max Level", f"{int(np.nanmedian(max_level_num))}")
        k2.metric("95th Percentile Max Level", f"{int(np.nanpercentile(max_level_num.dropna(), 95)) if max_level_num.notna().any() else 0}")

        # Drop-off curve: percentage of users reaching each level
        reach_pct = {}
        total = len(scoped)
        for i, col in enumerate(level_cols, start=1):
            reach_pct[col] = float(scoped[col].notna().sum()) / total if total > 0 else 0.0
        drop_df = pd.DataFrame({"level": list(reach_pct.keys()), "reach_pct": list(reach_pct.values())})
        fig = px.line(drop_df, x="level", y="reach_pct", title="Level Reach Percentage (Drop-off)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Sample Records")
        st.dataframe(scoped.head(200), use_container_width=True)

    def show_data_ingestion():
        """Upload Excel/CSV, preview, store on server, and optionally ask GPT to interpret schema."""
        st.header("📥 Data Ingestion")
        st.info("Upload .xlsx or .csv files following the Game > Platform > Channel > Countries hierarchy. We'll store them under data/raw/ and make them available for training.")
        
        # Show template download
        st.subheader("📋 Data Template")
        st.markdown("""
        **Download the data template** to ensure proper data structure:
        
        **Required Hierarchy**: Game > Platform > Channel > Countries
        """)
        
        # Create download button for the template
        if os.path.exists("Data_Template_GameLens_AI.csv"):
            with open("Data_Template_GameLens_AI.csv", "r") as file:
                csv_data = file.read()
            
            st.download_button(
                label="📥 Download Data Template CSV",
                data=csv_data,
                file_name="Data_Template_GameLens_AI.csv",
                mime="text/csv",
                help="Download the data template with Game > Platform > Channel > Countries hierarchy"
            )
        else:
            st.error("Template file not found. Please ensure Data_Template_GameLens_AI.csv exists in the project root.")
        
        # Show template guide
        if os.path.exists("DATA_TEMPLATE_GUIDE.md"):
            with open("DATA_TEMPLATE_GUIDE.md", "r") as file:
                guide_content = file.read()
            
            st.download_button(
                label="📖 Download Template Guide",
                data=guide_content,
                file_name="DATA_TEMPLATE_GUIDE.md",
                mime="text/markdown",
                help="Download the comprehensive data template guide"
            )
        
        # Show template preview
        if st.checkbox("Show template preview"):
            template_data = {
                'game': ['Your Game Name', 'Your Game Name'],
                'platform': ['Unity Ads', 'Mistplay'],
                'channel': ['Android', 'iOS'],
                'country': ['United States', 'Canada'],
                'date': ['2025-01-01', '2025-01-01'],
                'installs': [100, 150],
                'cost': [50.0, 75.0],
                'roas_d30': [1.2, 1.4]
            }
            template_df = pd.DataFrame(template_data)
            st.dataframe(template_df, use_container_width=True)
        
        # Show direct file access information
        st.info("""
        **Alternative Access Methods:**
        - Use the download buttons above to get the template files
        - Or run the template server: `python serve_templates.py` (port 8505)
        - Then access directly via: `http://localhost:8505/Data_Template_GameLens_AI.csv`
        - Template guide: `http://localhost:8505/DATA_TEMPLATE_GUIDE.md`
        """)

        # Destination directories
        raw_dir = os.path.join("data", "raw")
        os.makedirs(raw_dir, exist_ok=True)

        uploaded_files = st.file_uploader(
            "Upload one or more files",
            type=["xlsx", "xls", "csv"],
            accept_multiple_files=True
        )

        if uploaded_files:
            for up in uploaded_files:
                try:
                    # Save to disk
                    save_path = os.path.join(raw_dir, up.name)
                    with open(save_path, "wb") as f:
                        f.write(up.getbuffer())

                    st.success(f"Saved: {save_path}")

                    # Read a small preview
                    preview_df = None
                    if up.name.lower().endswith((".xlsx", ".xls")):
                        try:
                            # Read first sheet for preview
                            preview_df = pd.read_excel(save_path, nrows=200)
                        except Exception:
                            preview_df = None
                    elif up.name.lower().endswith(".csv"):
                        try:
                            preview_df = pd.read_csv(save_path, nrows=200)
                        except Exception:
                            preview_df = None

                    if preview_df is not None and not preview_df.empty:
                        st.subheader(f"Preview: {up.name}")
                        st.dataframe(preview_df.head(20), use_container_width=True)

                        # Quick heuristics for column categories
                        col_names = [str(c) for c in preview_df.columns]
                        st.caption("Detected columns: " + ", ".join(col_names))

                        # Validate hierarchy structure
                        st.subheader(f"Data Validation: {up.name}")
                        hierarchy_cols = ['game', 'platform', 'channel', 'country']
                        missing_hierarchy = [col for col in hierarchy_cols if col not in preview_df.columns]
                        
                        if missing_hierarchy:
                            st.error(f"❌ Missing required hierarchy columns: {missing_hierarchy}")
                            st.warning("Please ensure your data includes the Game > Platform > Channel > Countries hierarchy")
                        else:
                            st.success("✅ All required hierarchy columns present")
                        
                        # Check for key data columns
                        key_cols = ['installs', 'cost', 'roas_d30']
                        missing_key = [col for col in key_cols if col not in preview_df.columns]
                        if missing_key:
                            st.warning(f"⚠️ Missing key data columns: {missing_key}")
                        else:
                            st.success("✅ Key data columns present")

                        # Optional: Ask GPT to interpret schema and suggest mappings
                        if LLM_AVAILABLE and GameLensLLMService:
                            if st.checkbox(f"Ask GPT to interpret data schema for {up.name}"):
                                try:
                                    llm = GameLensLLMService()
                                    if llm and llm.is_available():
                                        sample_json = preview_df.head(10).to_json(orient="records")
                                        question = (
                                            "Given this tabular data preview, identify likely columns for: "
                                            "date, game, platform, channel, country, spend, revenue, installs, "
                                            "ROAS day columns (e.g., roas_d1, roas_d7, roas_d30), retention. "
                                            "Also suggest which of these four filters we can support: Game, Platform, Channel, Countries."
                                        )
                                        context = {"columns": col_names, "sample": sample_json}
                                        answer = llm.answer_faq_question(question, context, faq_content="")
                                        st.write(answer)
                                    else:
                                        st.warning("LLM not available. Configure OPENAI_API_KEY to enable GPT interpretation.")
                                except Exception as e:
                                    st.warning(f"LLM interpretation failed: {e}")

                    else:
                        st.warning(f"Could not preview {up.name}. File saved; please ensure it is a valid spreadsheet.")
                except Exception as e:
                    st.error(f"Upload/save failed for {up.name}: {e}")

            st.success("Files uploaded. They will be picked up on next data load.")

            if st.button("Reload data now", type="primary"):
                # Clear caches and rerun to include new data in the pipeline
                load_data.clear()
                create_features.clear()
                st.success("Data caches cleared. Reloading...")
                st.rerun()

    def show_validation():
        """Validate predictions against historical actuals with group filters."""
        st.header("✅ Predictions Validation")

        if 'forecaster' not in st.session_state:
            st.warning("Train a model first in the Model Training tab.")
            return

        forecaster = st.session_state['forecaster']
        X = st.session_state.get('X')
        y = st.session_state.get('y')

        if X is None or y is None:
            st.warning("No training data found in session. Please re-train the model.")
            return

        # Get predictions
        try:
            preds = forecaster.predict_with_confidence(X)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        # Merge predictions and actuals
        df_val = pd.DataFrame(index=X.index).copy()
        df_val['predicted_roas'] = preds['roas_prediction']
        df_val['actual_roas'] = y
        df_val['abs_error'] = (df_val['actual_roas'] - df_val['predicted_roas']).abs()
        df_val['ape'] = ((df_val['actual_roas'] - df_val['predicted_roas']).abs() / df_val['actual_roas'].replace(0, np.nan)).clip(upper=10)

        # Attach available metadata columns from X if present
        possible_meta = [
            'game', 'title', 'app',
            'platform', 'source', 'network', 'channel',
            'country', 'geo', 'country_code', 'country_name', 'region'
        ]
        present_meta = [c for c in possible_meta if c in X.columns]
        meta_df = X[present_meta] if present_meta else pd.DataFrame(index=X.index)
        df_val = df_val.join(meta_df, how='left')

        # Filters: Game, Platform, Channel, Countries
        st.subheader("Filters")
        c1, c2, c3, c4 = st.columns(4)
        game_col = next((c for c in ['game', 'title', 'app'] if c in df_val.columns), None)
        plat_col = next((c for c in ['platform', 'source', 'network'] if c in df_val.columns), None)
        chan_col = next((c for c in ['channel', 'network'] if c in df_val.columns), None)
        country_col = next((c for c in ['country', 'geo', 'country_code', 'country_name', 'region'] if c in df_val.columns), None)

        selected_game = c1.selectbox("Game", ["All"] + sorted([str(x) for x in df_val[game_col].dropna().unique()]) if game_col else ["All"])
        selected_platform = c2.selectbox("Platform", ["All"] + sorted([str(x) for x in df_val[plat_col].dropna().unique()]) if plat_col else ["All"])
        selected_channel = c3.selectbox("Channel", ["All"] + sorted([str(x) for x in df_val[chan_col].dropna().unique()]) if chan_col else ["All"])
        selected_country = c4.selectbox("Countries", ["All"] + sorted([str(x) for x in df_val[country_col].dropna().unique()]) if country_col else ["All"])

        scoped = df_val.copy()
        if game_col and selected_game != "All":
            scoped = scoped[scoped[game_col] == selected_game]
        if plat_col and selected_platform != "All":
            scoped = scoped[scoped[plat_col] == selected_platform]
        if chan_col and selected_channel != "All":
            scoped = scoped[scoped[chan_col] == selected_channel]
        if country_col and selected_country != "All":
            scoped = scoped[scoped[country_col] == selected_country]

        # KPIs
        st.subheader("Validation Metrics")
        if len(scoped) == 0:
            st.warning("No data after filters.")
            return

        r2 = forecaster.performance_metrics.get('r2') if hasattr(forecaster, 'performance_metrics') else None
        mape = np.nanmean(scoped['ape']) if 'ape' in scoped else np.nan
        mae = np.nanmean(scoped['abs_error']) if 'abs_error' in scoped else np.nan

        m1, m2, m3 = st.columns(3)
        m1.metric("MAPE", f"{mape:.4f}")
        m2.metric("MAE", f"{mae:.4f}")
        m3.metric("R² (global)", f"{(r2 if r2 is not None else 0):.4f}")

        # Plots
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=scoped['actual_roas'], y=scoped['predicted_roas'], mode='markers', name='Points'))
            if scoped['actual_roas'].notna().any() and scoped['predicted_roas'].notna().any():
                mn = float(min(scoped['actual_roas'].min(), scoped['predicted_roas'].min()))
                mx = float(max(scoped['actual_roas'].max(), scoped['predicted_roas'].max()))
                fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode='lines', name='Ideal', line=dict(dash='dash', color='red')))
            fig.update_layout(title='Actual vs Predicted ROAS', xaxis_title='Actual', yaxis_title='Predicted')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.histogram(scoped, x='abs_error', nbins=30, title='Absolute Error Distribution')
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Detailed Table")
        st.dataframe(scoped.head(500), use_container_width=True)
        csv = scoped.to_csv(index=False)
        st.download_button("Download validation CSV", data=csv, file_name="validation_scoped.csv", mime="text/csv")

    def _read_faq_content() -> Optional[str]:
        """Read FAQ content from common locations and formats.

        Supports:
        - FAQ.docx (preferred)
        - faq.md
        - faq.txt
        Returns markdown string or None if not found.
        """
        # Try DOCX first
        possible_paths = [
            "FAQ.docx", "faq.docx", "docs/FAQ.docx", "docs/faq.docx",
            "faq.md", "FAQ.md", "docs/faq.md", "docs/FAQ.md",
            "faq.txt", "FAQ.txt", "docs/faq.txt", "docs/FAQ.txt",
        ]

        # Resolve path that exists
        selected_path = None
        for p in possible_paths:
            if os.path.exists(p):
                selected_path = p
                break

        if not selected_path:
            return None

        # If markdown or text -> simple read
        if selected_path.lower().endswith((".md", ".txt")):
            try:
                with open(selected_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                return None

        # If DOCX -> convert to markdown using python-docx
        if selected_path.lower().endswith(".docx"):
            try:
                from docx import Document
                doc = Document(selected_path)
                
                # Extract text from all paragraphs
                content_parts = []
                for paragraph in doc.paragraphs:
                    text = paragraph.text.strip()
                    if text:  # Only add non-empty paragraphs
                        content_parts.append(text)
                
                # Join all content
                full_content = "\n\n".join(content_parts)
                
                if full_content.strip():
                    return full_content
                else:
                    return None
                    
            except ImportError:
                return "❌ python-docx library not available. Please install with: pip install python-docx"
            except Exception as e:
                return f"❌ Error reading DOCX file: {str(e)}"
        
        return None

    def show_faq():
        """Render FAQ page with LLM-powered answers"""
        st.header("❓ Frequently Asked Questions")
        
        # Initialize LLM service if available
        llm_service = None
        if LLM_AVAILABLE and GameLensLLMService:
            try:
                llm_service = GameLensLLMService()
            except Exception as e:
                st.warning(f"⚠️ Error initializing LLM service: {e}")
                llm_service = None
        
        # Check if LLM is available
        if not llm_service or not llm_service.is_available():
            st.warning("⚠️ LLM service not available. Using intelligent fallback FAQ system.")
            st.info("💡 To enable LLM-powered FAQ answers:\n1. Copy `env.example` to `.env`\n2. Add your OpenAI API key: `OPENAI_API_KEY=your_key_here`\n3. Restart the application")
            st.success("✅ The fallback system provides comprehensive answers based on your data and model performance.")
        
        # Pull objects from session if available
        combined_data = st.session_state.get('combined_data')
        features_df = st.session_state.get('features_df')
        forecaster = st.session_state.get('forecaster')
        X = st.session_state.get('X')
        y = st.session_state.get('y')
        target_col = st.session_state.get('target_col')
        target_day = st.session_state.get('target_day')

        # --- Campaign-aware filtering controls ---
        # Build a unified dataframe (if available) to drive filter options
        import pandas as pd
        all_data_df = None
        if combined_data and isinstance(combined_data, dict):
            try:
                dfs = [df for df in combined_data.values() if df is not None and not df.empty]
                if dfs:
                    all_data_df = pd.concat(dfs, ignore_index=True, sort=False)
            except Exception:
                all_data_df = None
        # Fallbacks when the raw combined_data is not available
        if all_data_df is None or (hasattr(all_data_df, "empty") and all_data_df.empty):
            if features_df is not None and not features_df.empty:
                all_data_df = features_df.copy()
            elif 'predictions' in st.session_state:
                preds_df = st.session_state.get('predictions')
                if preds_df is not None and not preds_df.empty:
                    all_data_df = preds_df.copy()

        # Helper to get the first existing column from a list
        def first_existing_column(df: pd.DataFrame, candidates):
            if df is None:
                return None
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        platform_col = first_existing_column(all_data_df, [
            "platform", "source", "network", "channel"
        ])
        campaign_col = first_existing_column(all_data_df, [
            "campaign", "campaign_id", "campaign_name",
            "adset", "adset_id", "adset_name",
            "line_item", "adgroup", "adgroup_id", "adgroup_name"
        ])
        geo_col = first_existing_column(all_data_df, [
            "geo", "country", "country_code", "country_iso", "country_name", "region"
        ])

        with st.expander("Filter scope for FAQ answers (optional)", expanded=True):
            col_f1, col_f2, col_f3, col_f4 = st.columns(4)
            selected_game = None
            selected_platform = None
            selected_channel = None
            selected_country = None

            # Map columns
            game_col = first_existing_column(all_data_df, ["game", "title", "app"])
            platform_col = first_existing_column(all_data_df, ["platform", "source", "network", "channel"]) or platform_col
            channel_col = first_existing_column(all_data_df, ["channel", "network"]) or None
            country_col = first_existing_column(all_data_df, ["country", "geo", "country_code", "country_name", "region"]) or geo_col

            if game_col and all_data_df is not None and game_col in all_data_df.columns:
                options = ["All"] + sorted([str(x) for x in all_data_df[game_col].dropna().unique()])
                selected_game = col_f1.selectbox("Game", options, index=0)
                if selected_game == "All":
                    selected_game = None

            if platform_col and all_data_df is not None and platform_col in all_data_df.columns:
                options = ["All"] + sorted([str(x) for x in all_data_df[platform_col].dropna().unique()])
                selected_platform = col_f2.selectbox("Platform", options, index=0)
                if selected_platform == "All":
                    selected_platform = None

            if channel_col and all_data_df is not None and channel_col in all_data_df.columns:
                options = ["All"] + sorted([str(x) for x in all_data_df[channel_col].dropna().unique()])
                selected_channel = col_f3.selectbox("Channel", options, index=0)
                if selected_channel == "All":
                    selected_channel = None

            if country_col and all_data_df is not None and country_col in all_data_df.columns:
                options = ["All"] + sorted([str(x) for x in all_data_df[country_col].dropna().unique()])
                selected_country = col_f4.selectbox("Countries", options, index=0)
                if selected_country == "All":
                    selected_country = None

            if not any([game_col, platform_col, channel_col, country_col]):
                st.info("No filterable metadata (game/platform/channel/country) found in loaded data.")

        # Helper to apply the selected scope to any dataframe
        def apply_scope(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df is None or (hasattr(df, "empty") and df.empty):
                return df
            scoped = df
            try:
                if selected_game is not None:
                    col_name = first_existing_column(scoped, [game_col, "game", "title", "app"]) or ""
                    if col_name in scoped.columns:
                        scoped = scoped[scoped[col_name] == selected_game]
                if selected_platform is not None:
                    col_name = first_existing_column(scoped, [platform_col, "platform", "source", "network"]) or ""
                    if col_name in scoped.columns:
                        scoped = scoped[scoped[col_name] == selected_platform]
                if selected_channel is not None:
                    col_name = first_existing_column(scoped, [channel_col, "channel", "network"]) or ""
                    if col_name in scoped.columns:
                        scoped = scoped[scoped[col_name] == selected_channel]
                if selected_country is not None:
                    col_name = first_existing_column(scoped, [country_col, geo_col, "geo", "country", "country_code", "region"]) or ""
                    if col_name in scoped.columns:
                        scoped = scoped[scoped[col_name] == selected_country]
                return scoped
            except Exception:
                return df

        # Helper: compute KPIs used across answers
        def compute_kpis():
            kpis = {}
            try:
                if combined_data and isinstance(combined_data, dict):
                    # Records per platform
                    platform_counts = {}
                    for df in combined_data.values():
                        if df is not None and not df.empty and 'platform' in df.columns:
                            counts = df['platform'].value_counts().to_dict()
                            for k, v in counts.items():
                                platform_counts[k] = platform_counts.get(k, 0) + int(v)
                    kpis['platform_counts'] = platform_counts

                # Model metrics (scoped by selected filters when possible)
                X_scoped = apply_scope(X) if X is not None else None
                y_scoped = y.loc[X_scoped.index] if (X_scoped is not None and y is not None and hasattr(y, 'loc')) else y
                if forecaster is not None and X_scoped is not None and y_scoped is not None and len(X_scoped) > 0:
                    try:
                        metrics = forecaster.evaluate_model(X_scoped, y_scoped)
                    except Exception:
                        metrics = getattr(forecaster, 'performance_metrics', {}) or {}
                    kpis['metrics'] = metrics

                    # Predictions summary
                    try:
                        preds = forecaster.predict_with_confidence(X_scoped)
                        kpis['predictions'] = preds
                        kpis['pred_summary'] = {
                            'mean': float(preds['roas_prediction'].mean()),
                            'p10': float(preds['roas_prediction'].quantile(0.1)),
                            'p50': float(preds['roas_prediction'].quantile(0.5)),
                            'p90': float(preds['roas_prediction'].quantile(0.9)),
                        }
                    except Exception:
                        kpis['predictions'] = None
                        kpis['pred_summary'] = {}

                    # Feature importance
                    try:
                        fi = forecaster.get_feature_importance(top_n=10)
                        kpis['top_features'] = list(zip(fi['feature'], fi['importance']))
                    except Exception:
                        kpis['top_features'] = []

                    # Best platform by actual target if available
                    features_scoped = apply_scope(features_df) if features_df is not None else None
                    if features_scoped is not None and target_col in features_scoped.columns:
                        if 'platform' in features_scoped.columns:
                            by_plat = features_scoped[[target_col, 'platform']].dropna()
                            if not by_plat.empty:
                                agg = by_plat.groupby('platform')[target_col].mean().sort_values(ascending=False)
                                kpis['best_platform'] = str(agg.index[0])

                # Generate recommendations
                if forecaster is not None and X_scoped is not None and len(X_scoped) > 0:
                    try:
                        recs = forecaster.generate_recommendations(X_scoped, target_roas=1.0)
                        kpis['recommendations'] = recs
                    except Exception:
                        kpis['recommendations'] = None
                        
                return kpis
            except Exception:
                return {}

        kpis = compute_kpis()

        # Parse questions from FAQ content (docx/md/txt). Default questions if none found.
        content = _read_faq_content()
        
        # Show content loading status
        if content and not content.startswith("❌"):
            st.success(f"✅ FAQ content loaded successfully ({len(content)} characters)")
            if "FAQ.docx" in str(content) or content.count('\n') > 5:
                st.info("📄 DOCX content detected - questions will be extracted from your FAQ document")
        elif content and content.startswith("❌"):
            st.error(content)
        else:
            st.info("📝 No FAQ document found - using default questions")
        
        questions = []
        if content:
            # Enhanced parsing for DOCX content
            lines = content.splitlines()
            for i, raw in enumerate(lines):
                line = raw.strip(" -\t")
                if not line:
                    continue
                
                # More flexible question detection
                is_question = (
                    line.lower().startswith(('q', 'question')) or 
                    line.endswith('?') or
                    line.lower().startswith('what') or
                    line.lower().startswith('how') or
                    line.lower().startswith('why') or
                    line.lower().startswith('when') or
                    line.lower().startswith('where') or
                    line.lower().startswith('which') or
                    line.lower().startswith('who') or
                    line.lower().startswith('can') or
                    line.lower().startswith('should') or
                    line.lower().startswith('will') or
                    line.lower().startswith('does') or
                    line.lower().startswith('is') or
                    line.lower().startswith('are')
                )
                
                if is_question:
                    # Clean up the question
                    line = re.sub(r'^q\s*[:\-\.]\s*', '', line, flags=re.I)
                    line = re.sub(r'^question\s*[:\-\.]\s*', '', line, flags=re.I)
                    line = line.strip()
                    if line and len(line) > 10:  # Only add substantial questions
                        questions.append(line)
            
            # If no questions found, try to extract from paragraph structure
            if not questions and content:
                # Split by double newlines and look for question patterns
                paragraphs = content.split('\n\n')
                for para in paragraphs:
                    para = para.strip()
                    if para and ('?' in para or para.lower().startswith(('what', 'how', 'why', 'when', 'where', 'which', 'who', 'can', 'should', 'will', 'does', 'is', 'are'))):
                        # Take the first sentence if it looks like a question
                        first_sentence = para.split('.')[0].strip()
                        if first_sentence and len(first_sentence) > 10:
                            questions.append(first_sentence)

        if not questions:
            questions = [
                "What is the current model performance?",
                "Which platform performs best for ROAS?",
                "What are the top drivers of ROAS in our data?",
                "What ROAS should we expect from our campaigns?",
                "Which campaigns should we scale or cut based on predictions?",
                "How accurate are our ROAS predictions?",
                "What insights can you provide about our advertising data?",
            ]
            st.warning("⚠️ No questions found in FAQ document - using default questions")
        else:
            st.success(f"📋 Found {len(questions)} questions in your FAQ document")

        # LLM-powered answer function with enhanced context and scope
        def answer_question_with_llm(q: str) -> str:
            if llm_service and llm_service.is_available():
                try:
                    # Add additional context for better LLM responses
                    enhanced_kpis = kpis.copy()
                    
                    # Add current session state information
                    if 'target_day' in st.session_state:
                        enhanced_kpis['target_day'] = st.session_state['target_day']
                    
                    if 'model_params' in st.session_state:
                        enhanced_kpis['model_params'] = st.session_state['model_params']
                    
                    # Add scope information for campaign-aware answers
                    enhanced_kpis['scope'] = {
                        'game': selected_game or 'All',
                        'platform': selected_platform or 'All',
                        'channel': selected_channel or 'All',
                        'country': selected_country or 'All',
                    }

                    # Add data summary (scoped)
                    if all_data_df is not None:
                        scoped_df = apply_scope(all_data_df)
                        enhanced_kpis['data_summary'] = {
                            'total_records': int(len(scoped_df)) if scoped_df is not None else 0,
                            'platforms': list(sorted(scoped_df[platform_col].dropna().unique())) if (scoped_df is not None and platform_col in scoped_df.columns) else [],
                            'date_range': f"{scoped_df['date'].min()} to {scoped_df['date'].max()}" if (scoped_df is not None and 'date' in scoped_df.columns and not scoped_df.empty) else 'Unknown'
                        }
                        
                        # Add ROAS progression data for ROI timeline analysis
                        roas_columns = [col for col in scoped_df.columns if col.startswith('roas_d')]
                        if roas_columns:
                            enhanced_kpis['data_summary']['roas_columns'] = roas_columns
                            
                            # Calculate average ROAS progression by day
                            roas_progression = {}
                            for col in roas_columns:
                                if col in scoped_df.columns:
                                    avg_roas = scoped_df[col].mean()
                                    if not pd.isna(avg_roas):
                                        roas_progression[col] = float(avg_roas)
                            
                            if roas_progression:
                                enhanced_kpis['roas_progression'] = roas_progression
                    
                    return llm_service.answer_faq_question(q, enhanced_kpis, content or "")
                except Exception as e:
                    st.error(f"Error generating LLM answer: {e}")
                    return answer_question_simple(q, kpis)
            else:
                # Fallback to simple keyword-based answers
                return answer_question_simple(q, kpis)

        # Simple fallback answer function
        def answer_question_simple(q: str, kpis: dict) -> str:
            ql = q.lower()
            
            # Handle ROI timeline questions
            if any(k in ql for k in ["100% roi", "when will roi", "break-even", "payback", "d15", "d30", "d90", "roi of 100%"]):
                roas_prog = kpis.get('roas_progression', {})
                if roas_prog:
                    # Find when ROAS reaches 1.0
                    roas_1_0_day = None
                    for day, roas_value in roas_prog.items():
                        if isinstance(roas_value, (int, float)) and roas_value >= 1.0:
                            roas_1_0_day = day
                            break
                    
                    if roas_1_0_day:
                        return f"Based on the data, ROI of 100% (ROAS = 1.0) is achieved at {roas_1_0_day}. The ROAS progression shows: " + ", ".join([f"{day}: {value:.3f}" for day, value in roas_prog.items()])
                    else:
                        max_roas = max(roas_prog.values()) if roas_prog else 0
                        max_day = max(roas_prog.keys(), key=lambda x: roas_prog[x]) if roas_prog else "Unknown"
                        return f"ROI of 100% is not achieved in the available data. Maximum ROAS is {max_roas:.3f} at {max_day}. ROAS progression: " + ", ".join([f"{day}: {value:.3f}" for day, value in roas_prog.items()])
                else:
                    return "ROAS progression data not available. Please ensure your data includes ROAS columns (roas_d0, roas_d1, roas_d3, roas_d7, roas_d14, roas_d30, roas_d60, roas_d90) and train a model."
            
            if any(k in ql for k in ["perform", "metric", "accuracy", "r2", "mape", "rmse", "mae"]):
                m = kpis.get('metrics', {})
                if not m:
                    return "Model is not trained yet. Train a model in the Model Training tab."
                parts = []
                if 'r2' in m:
                    parts.append(f"R²: {m['r2']:.3f}")
                if 'mape' in m:
                    parts.append(f"MAPE: {m['mape']:.3f}")
                if 'rmse' in m:
                    parts.append(f"RMSE: {m['rmse']:.3f}")
                if 'mae' in m:
                    parts.append(f"MAE: {m['mae']:.3f}")
                if 'confidence_coverage' in m:
                    parts.append(f"CI coverage: {m['confidence_coverage']:.3f}")
                label = f" for D{target_day}" if target_day else ""
                return "Model performance" + label + ": " + ", ".join(parts)

            if any(k in ql for k in ["best platform", "which platform", "platform perform"]):
                bp = kpis.get('best_platform')
                if not bp:
                    return "Insufficient data to identify the best platform."
                return f"Best platform is {bp}"

            if any(k in ql for k in ["driver", "feature", "important"]):
                tops = kpis.get('top_features', [])
                if not tops:
                    return "Feature importance not available. Train a model first."
                top_list = ", ".join([t[0] for t in tops[:5]])
                return f"Top drivers of ROAS include: {top_list}."

            if any(k in ql for k in ["expect roas", "forecast", "prediction", "expected roas"]):
                ps = kpis.get('pred_summary', {})
                if not ps:
                    return "No predictions available yet. Train a model first."
                label = f"D{target_day} " if target_day else ""
                return (f"Expected {label}ROAS — mean: {ps.get('mean', float('nan')):.3f}, "
                        f"p50: {ps.get('p50', float('nan')):.3f}, "
                        f"range ~ p10 {ps.get('p10', float('nan')):.3f} to p90 {ps.get('p90', float('nan')):.3f}.")

            if any(k in ql for k in ["scale", "cut", "reduce", "maintain", "budget"]):
                recs = kpis.get('recommendations')
                if recs is None or recs.empty:
                    return "Recommendations unavailable. Train a model to generate recommendations."
                lines = []
                for _, r in recs.head(5).iterrows():
                    lines.append(f"- {r['recommendation']} (pred ROAS {r['predicted_roas']:.3f})")
                return "\n".join(lines)

            if any(k in ql for k in ["data", "coverage", "how much", "volume"]):
                pc = kpis.get('platform_counts', {})
                if not pc:
                    return "No platform distribution available."
                return "Data records by platform: " + ", ".join([f"{k}: {v}" for k, v in pc.items()])

            return "This question is currently not supported by the automated Q&A."

        # Custom question input
        st.subheader("💬 Ask a Custom Question")
        custom_question = st.text_input("Ask anything about your ROAS forecasting data:", 
                                      placeholder="e.g., What insights can you provide about our campaign performance?")
        
        if custom_question:
            with st.spinner("🤖 Generating answer..."):
                answer = answer_question_with_llm(custom_question)
                st.write(answer)

        # Render predefined Q&A
        st.subheader("📋 Common Questions")
        for q in questions:
            with st.expander(q):
                with st.spinner("🤖 Generating answer..."):
                    answer = answer_question_with_llm(q)
                    st.write(answer)

    def main():
        """Main application"""
        # Header
        st.markdown('<h1 class="main-header">🎮 GameLens AI - ROAS Forecasting Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("### Phase 1: Multi-Platform ROAS Forecasting with Unity Ads & Mistplay")
        
        # Sidebar
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Choose a page",
            ["Data Overview", "Feature Engineering", "Model Training", "Predictions", "Validation", "Level Progression", "Recommendations", "Data Ingestion", "FAQ"]
        )
        
        # Load data
        combined_data, data_loader = load_data()
        # Persist for cross-page access (e.g., FAQ filters)
        st.session_state['combined_data'] = combined_data
        
        if combined_data is None:
            st.error("Failed to load data. Please check your data directory.")
            return
        
        # Create features
        features_df, feature_engineer = create_features(combined_data)
        # Persist for cross-page access (e.g., FAQ filters)
        st.session_state['features_df'] = features_df
        
        # Page routing
        if page == "Data Overview":
            show_data_overview(combined_data)
        elif page == "Feature Engineering":
            show_feature_engineering(features_df, feature_engineer)
        elif page == "Model Training":
            show_model_training(features_df)
        elif page == "Predictions":
            show_predictions()
        elif page == "Validation":
            show_validation()
        elif page == "Recommendations":
            show_recommendations()
        elif page == "Level Progression":
            show_level_progression(combined_data)
        elif page == "Data Ingestion":
            show_data_ingestion()
        elif page == "FAQ":
            show_faq()
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.markdown("**GameLens AI Phase 1**")
        st.sidebar.markdown("Multi-platform ROAS forecasting")
        st.sidebar.markdown("Unity Ads + Mistplay support")

    if __name__ == "__main__":
        main()
