# GameLens AI - ROAS Forecasting Dashboard Usage Guide

## Overview
The GameLens AI dashboard is a comprehensive ROAS (Return on Ad Spend) forecasting platform designed for mobile game advertising optimization. This guide provides detailed instructions for using each page of the dashboard.

## Dashboard Navigation
The dashboard consists of 6 main pages accessible via the sidebar:
1. **Data Overview** - Data loading and validation
2. **Feature Engineering** - Data preprocessing and feature creation
3. **Model Training** - Machine learning model training and configuration
4. **Predictions** - ROAS predictions and model performance analysis
5. **Recommendations** - Actionable insights and campaign recommendations
6. **FAQ** - AI-powered question answering system

---

## Page 1: Data Overview 📊

### Purpose
Load, validate, and explore your advertising data from multiple platforms (Unity Ads, Mistplay, etc.).

### How to Use
1. **Data Loading**:
   - The system automatically scans the `Campaign Data/` folder
   - Supports CSV files from Unity Ads and Mistplay platforms
   - Handles both Android and iOS data

2. **Data Validation**:
   - Check data quality metrics (missing values, duplicates)
   - Verify platform distribution
   - Review date ranges and sample sizes

3. **Data Preview**:
   - View sample records from each platform
   - Examine column structures and data types
   - Identify any data quality issues

### Key Features
- **Multi-platform support**: Unity Ads, Mistplay
- **Automatic data detection**: Scans for CSV files
- **Data quality metrics**: Missing values, duplicates, platform distribution
- **Data preview**: Sample records and column information

### What to Look For
- ✅ All expected platforms are loaded
- ✅ Data covers your desired date range
- ✅ No significant missing values or duplicates
- ✅ ROAS columns are present (roas_d0, roas_d7, roas_d30, etc.)

---

## Page 2: Feature Engineering 🔧

### Purpose
Transform raw advertising data into predictive features for machine learning models.

### How to Use
1. **Target Day Selection**:
   - Choose which ROAS day to predict (D7, D15, D30, D45, D90)
   - This determines what the model will learn to predict

2. **Feature Creation**:
   - Click "Create Features" to generate predictive features
   - Features include retention rates, early ROAS, level progression, cost metrics
   - Process may take a few minutes for large datasets

3. **Feature Analysis**:
   - Review feature summary statistics
   - Check feature importance rankings
   - Identify top predictive features

### Key Features
- **Retention features**: D1, D3, D7 retention rates
- **Early ROAS indicators**: D0, D3, D7 ROAS values
- **Engagement metrics**: Level progression, session data
- **Cost efficiency**: CPI, cost per level, revenue ratios
- **Platform-specific features**: Platform performance indicators

### What to Look For
- ✅ Features created successfully (no errors)
- ✅ Reasonable feature importance scores
- ✅ Features align with your business logic
- ✅ No missing values in critical features

---

## Page 3: Model Training 🤖

### Purpose
Train machine learning models to predict ROAS with confidence intervals.

### How to Use
1. **Model Configuration**:
   - **Target Day**: Select the ROAS day to predict
   - **Number of Estimators**: Trees in the model (default: 100)
   - **Learning Rate**: How fast the model learns (default: 0.1)
   - **Max Depth**: Tree depth (default: 6)
   - **Random State**: For reproducible results (default: 42)

2. **Training Process**:
   - Click "Train Model" to start training
   - Process may take 2-5 minutes depending on data size
   - Monitor progress with status messages

3. **Model Validation**:
   - Review training metrics (R², RMSE, MAPE, MAE)
   - Check confidence interval coverage
   - Examine feature importance rankings

### Key Features
- **Quantile Regression**: Predicts ROAS with confidence intervals
- **LightGBM/XGBoost**: High-performance gradient boosting
- **Feature Importance**: Identifies key predictive factors
- **Model Validation**: Comprehensive performance metrics

### What to Look For
- ✅ R² score > 0.3 (good model fit)
- ✅ MAPE < 50% (reasonable prediction accuracy)
- ✅ Confidence coverage ~80-90% (reliable intervals)
- ✅ Feature importance makes business sense

---

## Page 4: Predictions 📈

### Purpose
Generate ROAS predictions and analyze model performance on your data.

### How to Use
1. **Prediction Generation**:
   - Predictions are automatically generated after model training
   - Shows ROAS predictions with confidence intervals
   - Displays prediction distribution and statistics

2. **Performance Analysis**:
   - Review model performance metrics
   - Check prediction accuracy and confidence
   - Analyze ROAS distribution across campaigns

3. **Data Insights**:
   - View prediction summaries (mean, percentiles)
   - Examine confidence interval widths
   - Identify high/low performing campaigns

### Key Features
- **ROAS Predictions**: Point estimates with confidence bounds
- **Performance Metrics**: R², RMSE, MAPE, MAE, confidence coverage
- **Distribution Analysis**: ROAS ranges and percentiles
- **Campaign Insights**: High/medium/low ROAS categorization

### What to Look For
- ✅ Predictions completed successfully
- ✅ Reasonable ROAS ranges (0.1 to 5.0 typical)
- ✅ Good model performance metrics
- ✅ Confidence intervals are not too wide

---

## Page 5: Recommendations 💡

### Purpose
Generate actionable recommendations for campaign optimization based on model predictions.

### How to Use
1. **Recommendation Settings**:
   - **Target ROAS**: Set your desired ROAS goal (default: 0.5)
   - **Number of Recommendations**: How many to show (default: 10)
   - **Confidence Threshold**: Minimum confidence level (default: 0.8)

2. **Generate Recommendations**:
   - Click "Generate Recommendations"
   - Review campaign-specific recommendations
   - Analyze improvement potential

3. **Action Planning**:
   - Identify campaigns to scale or cut
   - Prioritize based on improvement potential
   - Consider confidence levels in decisions

### Key Features
- **Campaign Prioritization**: Ranked by improvement potential
- **Confidence Filtering**: Only high-confidence recommendations
- **ROAS Improvement**: Quantified potential gains
- **Actionable Insights**: Clear next steps for each campaign

### What to Look For
- ✅ Recommendations generated successfully
- ✅ Clear improvement potential for each campaign
- ✅ Confidence levels are reasonable
- ✅ Recommendations align with business goals

---

## Page 6: FAQ ❓

### Purpose
AI-powered question answering system that provides insights about your data and model performance.

### How to Use
1. **Question Selection**:
   - Browse pre-loaded questions from your FAQ document
   - Questions are automatically extracted from FAQ.docx
   - Click on any question to get an AI-powered answer

2. **AI-Powered Answers**:
   - Answers combine your FAQ content with current dashboard data
   - Provides data-driven insights and recommendations
   - Uses your actual model performance and predictions

3. **Context-Aware Responses**:
   - Answers are tailored to your specific data
   - Includes current model metrics and performance
   - Provides actionable insights based on your results

### Key Features
- **Document Integration**: Reads questions from FAQ.docx
- **AI-Powered**: Uses GPT for intelligent responses
- **Data-Driven**: Combines FAQ knowledge with dashboard data
- **Context-Aware**: Tailored to your specific results

### What to Look For
- ✅ FAQ content loaded successfully
- ✅ Questions extracted from your document
- ✅ AI answers are relevant and helpful
- ✅ Responses include your actual data insights

---

## Best Practices

### Data Preparation
1. **Ensure data quality**: Clean, complete datasets work best
2. **Include sufficient history**: At least 30 days of data recommended
3. **Verify ROAS columns**: Ensure target ROAS days are present
4. **Check platform coverage**: Include all relevant advertising platforms

### Model Training
1. **Start with defaults**: Use default parameters initially
2. **Monitor performance**: Watch for overfitting or underfitting
3. **Validate results**: Check that metrics make business sense
4. **Iterate if needed**: Adjust parameters based on performance

### Using Predictions
1. **Consider confidence**: Use confidence intervals in decision-making
2. **Focus on trends**: Look at patterns rather than individual predictions
3. **Validate with business logic**: Ensure predictions align with expectations
4. **Monitor over time**: Track prediction accuracy as new data comes in

### Making Recommendations
1. **Set realistic targets**: Use achievable ROAS goals
2. **Consider confidence levels**: Prioritize high-confidence recommendations
3. **Balance risk/reward**: Consider both potential gains and confidence
4. **Test and learn**: Start with small changes and scale successful ones

---

## Troubleshooting

### Common Issues
1. **Data loading errors**: Check CSV file formats and column names
2. **Feature engineering failures**: Ensure ROAS columns are present
3. **Model training issues**: Verify data quality and feature creation
4. **Poor model performance**: Check data quality and feature relevance

### Getting Help
1. **Check the FAQ page**: AI-powered answers for common questions
2. **Review error messages**: Detailed error information is provided
3. **Validate data quality**: Use the Data Overview page to check data
4. **Contact support**: For technical issues beyond the dashboard

---

## Technical Requirements

### System Requirements
- **RAM**: 32GB recommended for large datasets
- **Storage**: Sufficient space for data files and models
- **Network**: Stable internet for AI-powered FAQ (if enabled)

### Data Requirements
- **Format**: CSV files with standard column names
- **Platforms**: Unity Ads, Mistplay (others can be added)
- **Time Range**: Minimum 30 days of historical data
- **ROAS Columns**: Must include target ROAS days (D7, D15, D30, etc.)

### Optional Features
- **AI FAQ**: Requires OpenAI API key for enhanced question answering
- **Advanced Analytics**: Additional features available with premium data
- **Custom Models**: Specialized models for specific game types

---

This dashboard provides a complete solution for ROAS forecasting and campaign optimization. Each page builds upon the previous one, creating a comprehensive workflow from data loading to actionable recommendations.
