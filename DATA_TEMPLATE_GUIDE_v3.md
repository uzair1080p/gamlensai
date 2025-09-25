# GameLens AI Data Template Guide v3

## Overview
This enhanced template includes critical gaming metrics for accurate ROAS predictions, including retention and level progression data.

## Required Fields

### Core Campaign Data
- **game**: Game/app name
- **channel**: Traffic source (Unity Ads, Mistplay, Facebook Ads, etc.)
- **platform**: Device platform (Android, iOS)
- **country**: Target country/region
- **date**: Campaign date (YYYY-MM-DD)
- **installs**: Number of app installs
- **cost**: Total ad spend (USD)
- **ad_revenue**: Revenue from ads (USD)
- **revenue**: Total revenue (USD)

### ROAS Metrics (Return on Ad Spend)
- **roas_d0**: ROAS on day 0
- **roas_d1**: ROAS on day 1
- **roas_d3**: ROAS on day 3
- **roas_d7**: ROAS on day 7
- **roas_d14**: ROAS on day 14
- **roas_d30**: ROAS on day 30
- **roas_d60**: ROAS on day 60
- **roas_d90**: ROAS on day 90

### Retention Metrics (Critical for Gaming)
- **retention_d1**: % of users who return on day 1
- **retention_d3**: % of users who return on day 3
- **retention_d7**: % of users who return on day 7
- **retention_d14**: % of users who return on day 14
- **retention_d30**: % of users who return on day 30

### Level Progression (Gaming Engagement)
- **level_1_completion**: % of users who complete level 1
- **level_5_completion**: % of users who complete level 5
- **level_10_completion**: % of users who complete level 10
- **level_20_completion**: % of users who complete level 20
- **tutorial_completion**: % of users who complete tutorial

### Monetization Metrics
- **first_purchase_rate**: % of users who make their first purchase
- **d1_purchasers**: Number of purchasers on day 1
- **d7_purchasers**: Number of purchasers on day 7
- **d30_purchasers**: Number of purchasers on day 30

### Revenue per User
- **arpu_d1**: Average revenue per user on day 1
- **arpu_d7**: Average revenue per user on day 7
- **arpu_d30**: Average revenue per user on day 30

### Lifetime Value Predictions
- **ltv_30d**: Predicted lifetime value after 30 days
- **ltv_90d**: Predicted lifetime value after 90 days
- **ltv_365d**: Predicted lifetime value after 365 days

## Why These Fields Matter

### Retention Data
- **Critical for ROAS accuracy**: Users who don't return can't generate revenue
- **Predicts long-term value**: Higher retention = higher LTV
- **Identifies quality users**: Good retention indicates engaged players

### Level Progression
- **Engagement indicator**: Shows how deeply users engage with gameplay
- **Predicts monetization**: Users who progress further are more likely to pay
- **Quality signal**: Helps identify high-value user cohorts

### Monetization Metrics
- **Direct revenue impact**: Shows actual spending behavior
- **Purchase timing**: Helps understand when users convert
- **Revenue patterns**: Identifies optimal monetization strategies

## Data Collection Tips

1. **Track from day 1**: Start collecting retention data immediately
2. **Level completion**: Monitor progression through key game milestones
3. **Purchase events**: Track all in-app purchase events with timestamps
4. **Cohort analysis**: Group users by acquisition date for accurate metrics
5. **Regular updates**: Update data daily for best prediction accuracy

## Template Usage

1. Download the CSV template
2. Replace sample data with your actual metrics
3. Ensure all fields are populated (use 0 for missing data)
4. Upload to GameLens AI for analysis

## Benefits of Enhanced Template

- **More accurate predictions**: Retention and progression data significantly improve ROAS forecasting
- **Better campaign optimization**: Identify high-quality user acquisition channels
- **Improved budget allocation**: Focus spend on channels that drive engaged users
- **Enhanced insights**: Understand user behavior patterns and monetization timing

This enhanced template provides the comprehensive data needed for accurate gaming ROAS predictions and campaign optimization.
