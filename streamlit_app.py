import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import re
from datetime import datetime

# Add src to path for imports
sys.path.append('src')

# Import GameLens modules
from utils.data_loader import GameLensDataLoader
from utils.feature_engineering import GameLensFeatureEngineer
from utils.roas_forecaster import GameLensROASForecaster

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
            data_loader = GameLensDataLoader()
            all_data = data_loader.load_all_data()
            combined_data = data_loader.combine_platform_data(all_data)
            return combined_data, data_loader
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None

    @st.cache_data
    def create_features(combined_data):
        """Create and cache features"""
        try:
            feature_engineer = GameLensFeatureEngineer()
            features_df = feature_engineer.create_features(combined_data)
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
            
            # Model parameters
            st.subheader("Model Parameters")
            n_estimators = st.slider("Number of Estimators", 50, 500, 100)
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.05, 0.01)
            max_depth = st.slider("Max Depth", 3, 10, 6)
        
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
                    
                    # Prepare features and target
                    exclude_cols = ['platform', 'subdirectory', 'data_type', target_col]
                    feature_cols = [col for col in valid_data.columns if col not in exclude_cols]
                    
                    X = valid_data[feature_cols].fillna(0)
                    y = valid_data[target_col]
                    
                    # Initialize and train model
                    forecaster = GameLensROASForecaster()
                    
                    # Train model
                    models = forecaster.train_model(
                        X, y, 
                        quantiles=[0.1, 0.5, 0.9],
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        random_state=random_state
                    )
                    
                    # Store model in session state
                    st.session_state['forecaster'] = forecaster
                    st.session_state['X'] = X
                    st.session_state['y'] = y
                    st.session_state['feature_cols'] = feature_cols
                    st.session_state['target_col'] = target_col
                    st.session_state['target_day'] = target_day
                    
                    st.success(f"✅ Model trained successfully! Target: D{target_day} ROAS")
                    
                    # Show model info
                    feature_importance = forecaster.get_feature_importance()
                    if feature_importance:
                        st.subheader("Top 10 Feature Importance")
                        top_features = list(feature_importance.items())[:10]
                        importance_df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
                        st.dataframe(importance_df, use_container_width=True)
                    
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
        
        # Make predictions
        predictions = forecaster.predict_with_confidence(X)
        
        # Model performance metrics
        metrics = forecaster.evaluate_model(X, y)
        
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

    def main():
        """Main application"""
        # Header
        st.markdown('<h1 class="main-header">🎮 GameLens AI - ROAS Forecasting Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("### Phase 1: Multi-Platform ROAS Forecasting with Unity Ads & Mistplay")
        
        # Sidebar
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Choose a page",
            ["Data Overview", "Feature Engineering", "Model Training", "Predictions", "Recommendations"]
        )
        
        # Load data
        combined_data, data_loader = load_data()
        
        if combined_data is None:
            st.error("Failed to load data. Please check your data directory.")
            return
        
        # Create features
        features_df, feature_engineer = create_features(combined_data)
        
        # Page routing
        if page == "Data Overview":
            show_data_overview(combined_data)
        elif page == "Feature Engineering":
            show_feature_engineering(features_df, feature_engineer)
        elif page == "Model Training":
            show_model_training(features_df)
        elif page == "Predictions":
            show_predictions()
        elif page == "Recommendations":
            show_recommendations()
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.markdown("**GameLens AI Phase 1**")
        st.sidebar.markdown("Multi-platform ROAS forecasting")
        st.sidebar.markdown("Unity Ads + Mistplay support")

    if __name__ == "__main__":
        main()
