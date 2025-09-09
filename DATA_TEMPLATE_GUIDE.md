# GameLens AI - Data Template Guide

## Overview
This guide explains how to structure your data for GameLens AI following the **Game > Platform > Channel > Countries** hierarchy as requested by the client.

## Data Hierarchy Structure

### 1. Game (Top Level)
- **Field**: `game`
- **Description**: Name of your game/app
- **Example**: "My Awesome Game", "Puzzle Quest", "Racing Legends"

### 2. Platform (Second Level)
- **Field**: `platform`
- **Description**: Advertising platform used
- **Examples**: 
  - Unity Ads
  - Mistplay
  - Facebook Ads
  - Google Ads
  - TikTok Ads
  - Apple Search Ads

### 3. Channel (Third Level)
- **Field**: `channel`
- **Description**: Device/platform channel
- **Examples**:
  - Android
  - iOS
  - Web
  - Desktop

### 4. Countries (Fourth Level)
- **Field**: `country`
- **Description**: Geographic location
- **Examples**:
  - United States
  - Canada
  - United Kingdom
  - Germany
  - Japan
  - Australia

## Required Data Fields

### Core Fields (Required for all data types)
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `game` | String | Game/app name | "My Awesome Game" |
| `platform` | String | Advertising platform | "Unity Ads" |
| `channel` | String | Device channel | "Android" |
| `country` | String | Country name | "United States" |
| `date` | Date | Date in YYYY-MM-DD format | "2025-01-01" |
| `installs` | Integer | Number of installs | 100 |

### Adspend & Revenue Fields
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `cost` | Float | Total advertising spend | 50.0 |
| `ad_revenue` | Float | Revenue from ads only | 25.0 |
| `revenue` | Float | Total revenue (ads + IAP) | 30.0 |

### ROAS Fields (Return on Ad Spend)
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `roas_d0` | Float | ROAS on day 0 | 0.6 |
| `roas_d1` | Float | ROAS on day 1 | 0.7 |
| `roas_d3` | Float | ROAS on day 3 | 0.8 |
| `roas_d7` | Float | ROAS on day 7 | 0.9 |
| `roas_d14` | Float | ROAS on day 14 | 1.0 |
| `roas_d30` | Float | ROAS on day 30 | 1.2 |
| `roas_d60` | Float | ROAS on day 60 | 1.4 |
| `roas_d90` | Float | ROAS on day 90 | 1.6 |

### Retention Fields
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `retention_rate_d1` | Float | Day 1 retention rate | 0.3 |
| `retention_rate_d2` | Float | Day 2 retention rate | 0.2 |
| `retention_rate_d3` | Float | Day 3 retention rate | 0.15 |
| `retention_rate_d7` | Float | Day 7 retention rate | 0.1 |
| `retention_rate_d14` | Float | Day 14 retention rate | 0.08 |
| `retention_rate_d30` | Float | Day 30 retention rate | 0.05 |

### Level Progression Fields
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `level_1_events` | Integer | Users reaching level 1 | 95 |
| `level_5_events` | Integer | Users reaching level 5 | 80 |
| `level_10_events` | Integer | Users reaching level 10 | 60 |
| `level_15_events` | Integer | Users reaching level 15 | 45 |
| `level_20_events` | Integer | Users reaching level 20 | 35 |
| `level_25_events` | Integer | Users reaching level 25 | 25 |
| `level_30_events` | Integer | Users reaching level 30 | 20 |
| `level_40_events` | Integer | Users reaching level 40 | 15 |
| `level_50_events` | Integer | Users reaching level 50 | 10 |

## Data Upload Instructions

### 1. Prepare Your Data
- Use the provided `Data_Template_GameLens_AI.csv` as a starting point
- Ensure all required fields are present
- Follow the exact column names shown in the template
- Use consistent naming for games, platforms, channels, and countries

### 2. Data Quality Requirements
- **No missing values** in core fields (game, platform, channel, country, date, installs)
- **Consistent naming**: Use the same exact names for games, platforms, channels, and countries across all rows
- **Date format**: Use YYYY-MM-DD format (e.g., "2025-01-01")
- **Numeric values**: Use decimal points for float values (e.g., 0.6, not 0,6)

### 3. Upload Process
1. Go to the **Data Ingestion** page in the GameLens AI dashboard
2. Upload your CSV file using the file uploader
3. The system will automatically validate your data structure
4. Preview your data to ensure it's loaded correctly
5. The data will be available for analysis and predictions

## Example Data Structure

```csv
game,platform,channel,country,date,installs,cost,ad_revenue,revenue,roas_d0,roas_d1,roas_d3,roas_d7,roas_d14,roas_d30,roas_d60,roas_d90,retention_rate_d1,retention_rate_d2,retention_rate_d3,retention_rate_d7,retention_rate_d14,retention_rate_d30,level_1_events,level_5_events,level_10_events,level_15_events,level_20_events,level_25_events,level_30_events,level_40_events,level_50_events
My Awesome Game,Unity Ads,Android,United States,2025-01-01,100,50.0,25.0,30.0,0.6,0.7,0.8,0.9,1.0,1.2,1.4,1.6,0.3,0.2,0.15,0.1,0.08,0.05,95,80,60,45,35,25,20,15,10
My Awesome Game,Unity Ads,iOS,United States,2025-01-01,120,60.0,30.0,36.0,0.6,0.7,0.8,0.9,1.0,1.2,1.4,1.6,0.35,0.25,0.2,0.15,0.12,0.08,110,95,75,60,50,40,35,25,18
```

## Data Validation

The system will automatically validate your data and report any issues:

- ✅ **Structure validation**: Checks for required fields
- ✅ **Data type validation**: Ensures numeric fields contain numbers
- ✅ **Consistency validation**: Checks for consistent naming across rows
- ✅ **Completeness validation**: Identifies missing values in critical fields

## Troubleshooting

### Common Issues and Solutions

1. **"Missing required column" error**
   - Ensure all required fields are present in your CSV
   - Check that column names match exactly (case-sensitive)

2. **"Invalid data type" error**
   - Ensure numeric fields contain only numbers
   - Use decimal points (.) not commas (,) for decimal numbers
   - Remove any text or special characters from numeric fields

3. **"Inconsistent naming" warning**
   - Use the same exact names for games, platforms, channels, and countries
   - Avoid variations like "Unity Ads" vs "unity ads" vs "UnityAds"

4. **"Date format error"**
   - Use YYYY-MM-DD format (e.g., "2025-01-01")
   - Avoid formats like "01/01/2025" or "Jan 1, 2025"

## Support

If you encounter any issues with data upload or validation, please:
1. Check this guide for common solutions
2. Verify your data matches the template structure
3. Contact the development team with specific error messages

---

**Note**: This template enforces the Game > Platform > Channel > Countries hierarchy as requested by the client, ensuring proper data ingestion and enabling accurate predictions, analysis, and FAQ responses.
