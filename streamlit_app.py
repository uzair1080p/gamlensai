import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import re

# Add src to path
sys.path.append('src')

from utils.data_loader import GameLensDataLoader
from utils.feature_engineering import GameLensFeatureEngineer
from utils.roas_forecaster import GameLensROASForecaster

# Page configuration
st.set_page_config(
    page_title="GameLens AI - ROAS Forecasting",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
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
    .recommendation-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🎮 GameLens AI - ROAS Forecasting</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Forecasting & Optimization for Mobile Games")
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Target ROAS
    target_roas = st.sidebar.slider(
        "Target ROAS (%)", 
        min_value=50, 
        max_value=200, 
        value=100, 
        step=10
    ) / 100
    
    # Target day
    target_day = st.sidebar.selectbox(
        "Target Day",
        options=[15, 30, 45, 90],
        index=1
    )
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🤖 Model Training", "🔮 Predictions", "💡 Recommendations"])
    
    with tab1:
        show_data_overview()
    
    with tab2:
        show_model_training(target_day)
    
    with tab3:
        show_predictions(target_day)
    
    with tab4:
        show_recommendations(target_day, target_roas)

def show_data_overview():
    st.header("📊 Data Overview")
    
    try:
        # Load data
        data_loader = GameLensDataLoader()
        platform_data = data_loader.load_all_data()
        combined_data = data_loader.combine_platforms(platform_data)
        
        # Data summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Platforms", len(platform_data))
        
        with col2:
            total_countries = len(combined_data.get('retention', pd.DataFrame()))
            st.metric("Countries", total_countries)
        
        with col3:
            total_installs = combined_data.get('retention', pd.DataFrame())['installs'].sum()
            st.metric("Total Installs", f"{total_installs:,}")
        
        with col4:
            data_types = len(combined_data)
            st.metric("Data Types", data_types)
        
        # Platform breakdown
        st.subheader("Platform Breakdown")
        platform_summary = []
        for platform, data_types in platform_data.items():
            for data_type, df in data_types.items():
                platform_summary.append({
                    'Platform': platform,
                    'Data Type': data_type,
                    'Countries': len(df),
                    'Records': len(df)
                })
        
        platform_df = pd.DataFrame(platform_summary)
        st.dataframe(platform_df, use_container_width=True)
        
        # Data quality check
        st.subheader("Data Quality Check")
        validation_issues = data_loader.validate_data(combined_data)
        
        if validation_issues:
            st.warning("⚠️ Data quality issues found:")
            for data_type, issues in validation_issues.items():
                st.write(f"**{data_type}:**")
                for issue in issues:
                    st.write(f"  - {issue}")
        else:
            st.success("✅ No data quality issues found")
        
        # Sample data visualization
        if 'retention' in combined_data and not combined_data['retention'].empty:
            st.subheader("Retention Data Sample")
            
            retention_df = combined_data['retention']
            
            # Retention heatmap
            retention_cols = [col for col in retention_df.columns if 'retention_rate' in col]
            if retention_cols:
                heatmap_data = retention_df.set_index('country')[retention_cols]
                
                fig = px.imshow(
                    heatmap_data,
                    title="Retention Rates by Country and Day",
                    labels=dict(x="Day", y="Country", color="Retention Rate"),
                    aspect="auto"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        if 'roas' in combined_data and not combined_data['roas'].empty:
            st.subheader("ROAS Data Sample")
            
            roas_df = combined_data['roas']
            
            # ROAS trends
            roas_cols = [col for col in roas_df.columns if 'roas_d' in col]
            if roas_cols:
                # Select top countries by installs
                top_countries = roas_df.nlargest(10, 'installs')['country'].tolist()
                top_roas = roas_df[roas_df['country'].isin(top_countries)]
                
                # Prepare data for plotting
                plot_data = []
                for _, row in top_roas.iterrows():
                    for col in roas_cols:
                        m = re.search(r"(\d+)$", col)
                        if not m:
                            continue
                        day = int(m.group(1))
                        plot_data.append({
                            'Country': row['country'],
                            'Day': day,
                            'ROAS': row[col]
                        })
                
                plot_df = pd.DataFrame(plot_data)
                
                fig = px.line(
                    plot_df,
                    x='Day',
                    y='ROAS',
                    color='Country',
                    title="ROAS Trends by Country",
                    labels={'ROAS': 'ROAS (%)', 'Day': 'Day'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure the data files are in the correct location: Campaign Data/Unity Ads/")

def show_model_training(target_day):
    st.header("🤖 Model Training")
    
    try:
        # Load and prepare data
        with st.spinner("Loading and preparing data..."):
            data_loader = GameLensDataLoader()
            platform_data = data_loader.load_all_data()
            combined_data = data_loader.combine_platforms(platform_data)
            
            feature_engineer = GameLensFeatureEngineer()
            features_df = feature_engineer.create_cohort_features(combined_data)
            X, y = feature_engineer.prepare_training_data(features_df, target_day=target_day)
        
        st.success(f"✅ Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Model training
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model..."):
                forecaster = GameLensROASForecaster(target_day=target_day)
                models = forecaster.train_model(X, y)
                
                # Save model
                os.makedirs('models', exist_ok=True)
                forecaster.save_model(f'models/roas_forecaster_d{target_day}.pkl')
                
                st.success(f"✅ Model trained and saved for D{target_day} ROAS prediction")
                
                # Performance metrics
                metrics = forecaster.evaluate_model(X, y)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("MAPE", f"{metrics['mape']:.2%}")
                
                with col2:
                    st.metric("RMSE", f"{metrics['rmse']:.4f}")
                
                with col3:
                    st.metric("MAE", f"{metrics['mae']:.4f}")
                
                with col4:
                    st.metric("R²", f"{metrics['r2']:.4f}")
                
                # Feature importance
                st.subheader("Feature Importance")
                importance_df = forecaster.get_feature_importance(top_n=15)
                
                fig = px.bar(
                    importance_df,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title=f"Top 15 Features for D{target_day} ROAS Prediction"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Cross-validation results
                st.subheader("Cross-Validation Results")
                cv_scores = forecaster.cross_validate(X, y)
                
                cv_df = pd.DataFrame({
                    'Metric': ['MAPE', 'RMSE'],
                    'Mean': [cv_scores['mape'].mean(), cv_scores['rmse'].mean()],
                    'Std': [cv_scores['mape'].std(), cv_scores['rmse'].std()]
                })
                
                st.dataframe(cv_df, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")

def show_predictions(target_day):
    st.header("🔮 ROAS Predictions")
    
    try:
        # Load model
        model_path = f'models/roas_forecaster_d{target_day}.pkl'
        if not os.path.exists(model_path):
            st.warning(f"Model not found: {model_path}")
            st.info("Please train the model first in the 'Model Training' tab")
            return
        
        forecaster = GameLensROASForecaster(target_day=target_day)
        forecaster.load_model(model_path)
        
        # Load data for predictions
        data_loader = GameLensDataLoader()
        platform_data = data_loader.load_all_data()
        combined_data = data_loader.combine_platforms(platform_data)
        
        feature_engineer = GameLensFeatureEngineer()
        features_df = feature_engineer.create_cohort_features(combined_data)
        X, y = feature_engineer.prepare_training_data(features_df, target_day=target_day)
        
        # Make predictions
        predictions = forecaster.predict_with_confidence(X)
        
        # Combine with metadata
        results_df = features_df[['country', 'platform', 'installs']].copy()
        results_df = pd.concat([results_df, predictions], axis=1)
        
        # Display predictions
        st.subheader(f"D{target_day} ROAS Predictions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean Predicted ROAS", f"{results_df['roas_prediction'].mean():.3f}")
        
        with col2:
            st.metric("Min Predicted ROAS", f"{results_df['roas_prediction'].min():.3f}")
        
        with col3:
            st.metric("Max Predicted ROAS", f"{results_df['roas_prediction'].max():.3f}")
        
        # Predictions table
        st.subheader("Detailed Predictions")
        
        display_cols = ['country', 'platform', 'installs', 'roas_prediction', 
                       'roas_pred_q0.1', 'roas_pred_q0.9', 'confidence_interval_width']
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            platform_filter = st.multiselect(
                "Filter by Platform",
                options=results_df['platform'].unique(),
                default=results_df['platform'].unique()
            )
        
        with col2:
            min_installs = st.number_input("Min Installs", value=0, step=10)
        
        # Apply filters
        filtered_df = results_df[
            (results_df['platform'].isin(platform_filter)) &
            (results_df['installs'] >= min_installs)
        ]
        
        st.dataframe(filtered_df[display_cols].sort_values('roas_prediction', ascending=False), 
                    use_container_width=True)
        
        # Visualization
        st.subheader("Predictions Visualization")
        
        # Scatter plot of predictions vs confidence
        fig = px.scatter(
            filtered_df,
            x='roas_prediction',
            y='confidence_interval_width',
            size='installs',
            color='platform',
            hover_data=['country'],
            title=f"Predicted D{target_day} ROAS vs Confidence Interval Width"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution of predictions
        fig = px.histogram(
            filtered_df,
            x='roas_prediction',
            color='platform',
            title=f"Distribution of D{target_day} ROAS Predictions",
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading predictions: {str(e)}")

def show_recommendations(target_day, target_roas):
    st.header("💡 Actionable Recommendations")
    
    try:
        # Load model and data
        model_path = f'models/roas_forecaster_d{target_day}.pkl'
        if not os.path.exists(model_path):
            st.warning(f"Model not found: {model_path}")
            st.info("Please train the model first in the 'Model Training' tab")
            return
        
        forecaster = GameLensROASForecaster(target_day=target_day)
        forecaster.load_model(model_path)
        
        data_loader = GameLensDataLoader()
        platform_data = data_loader.load_all_data()
        combined_data = data_loader.combine_platforms(platform_data)
        
        feature_engineer = GameLensFeatureEngineer()
        features_df = feature_engineer.create_cohort_features(combined_data)
        X, y = feature_engineer.prepare_training_data(features_df, target_day=target_day)
        
        # Generate recommendations
        recommendations = forecaster.generate_recommendations(X, y, target_roas=target_roas)
        
        # Combine with metadata
        final_recommendations = pd.concat([
            features_df[['country', 'platform', 'installs']], 
            recommendations
        ], axis=1)
        
        # Summary metrics
        st.subheader(f"Recommendations Summary (Target ROAS: {target_roas:.1%})")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            scale_count = len(final_recommendations[
                final_recommendations['recommendation'].str.contains('Scale')
            ])
            st.metric("Scale Opportunities", scale_count)
        
        with col2:
            maintain_count = len(final_recommendations[
                final_recommendations['recommendation'].str.contains('Maintain')
            ])
            st.metric("Maintain Spend", maintain_count)
        
        with col3:
            reduce_count = len(final_recommendations[
                final_recommendations['recommendation'].str.contains('Reduce|Cut')
            ])
            st.metric("Reduce/Cut Spend", reduce_count)
        
        with col4:
            avg_gap = final_recommendations['roas_gap'].mean()
            st.metric("Avg ROAS Gap", f"{avg_gap:.3f}")
        
        # Recommendations breakdown
        st.subheader("Recommendations by Category")
        
        rec_summary = final_recommendations['recommendation'].value_counts()
        
        fig = px.pie(
            values=rec_summary.values,
            names=rec_summary.index,
            title="Distribution of Recommendations"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top scaling opportunities
        st.subheader("🚀 Top Scaling Opportunities")
        
        top_opportunities = final_recommendations[
            final_recommendations['recommendation'].str.contains('Scale')
        ].sort_values('predicted_roas', ascending=False)
        
        if not top_opportunities.empty:
            display_cols = ['country', 'platform', 'installs', 'predicted_roas', 
                           'confidence_lower', 'confidence_upper', 'recommendation']
            
            for _, row in top_opportunities.head(10).iterrows():
                with st.container():
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h4>{row['country']} ({row['platform']})</h4>
                        <p><strong>Predicted ROAS:</strong> {row['predicted_roas']:.3f} 
                        (CI: {row['confidence_lower']:.3f} - {row['confidence_upper']:.3f})</p>
                        <p><strong>Installs:</strong> {row['installs']:,}</p>
                        <p><strong>Recommendation:</strong> {row['recommendation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Risk areas
        st.subheader("⚠️ Risk Areas (Reduce/Cut Spend)")
        
        risk_areas = final_recommendations[
            final_recommendations['recommendation'].str.contains('Reduce|Cut')
        ].sort_values('predicted_roas')
        
        if not risk_areas.empty:
            display_cols = ['country', 'platform', 'installs', 'predicted_roas', 
                           'confidence_lower', 'confidence_upper', 'recommendation']
            
            for _, row in risk_areas.head(10).iterrows():
                with st.container():
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h4>{row['country']} ({row['platform']})</h4>
                        <p><strong>Predicted ROAS:</strong> {row['predicted_roas']:.3f} 
                        (CI: {row['confidence_lower']:.3f} - {row['confidence_upper']:.3f})</p>
                        <p><strong>Installs:</strong> {row['installs']:,}</p>
                        <p><strong>Recommendation:</strong> {row['recommendation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Export recommendations
        st.subheader("📤 Export Results")
        
        if st.button("Download Recommendations CSV"):
            csv = final_recommendations.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"roas_recommendations_d{target_day}.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")

if __name__ == "__main__":
    main()
