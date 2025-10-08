# PostgreSQL & n8n Integration Guide

## Overview
This guide explains the new PostgreSQL and n8n webhook integration for GameLens AI, which replaces the CSV parsing issues with a robust database-backed solution.

## What Changed

### New Files Created
1. **`glai/models_pg.py`** - PostgreSQL database models
2. **`glai/webhook_upload.py`** - n8n webhook integration utilities
3. **`glai/ai_helpers.py`** - AI analysis functions (build_payload_for_ai, ask_one_question, add_core_kpis)
4. **`env.example`** - Template for environment variables

### Modified Files
1. **`pages/2_🚀_Train_Predict_Validate_FAQ.py`** - Updated predictions and datasets tabs to use PostgreSQL
2. **`glai/ingest.py`** - Added robust CSV parsing with debug logs

## Server Deployment

### Step 1: SSH into Server
```bash
ssh root@170.64.236.80
cd ~/new_gam/gamlensai
```

### Step 2: Stop Streamlit
```bash
pkill -f streamlit
```

### Step 3: Pull Latest Code
```bash
git fetch origin
git checkout feature/postgresql-n8n-integration
git pull origin feature/postgresql-n8n-integration
```

### Step 4: Create .env File
```bash
cat > .env << 'EOF'
# OpenAI Configuration
OPENAI_API_KEY=your_actual_openai_key_here
OPENAI_MODEL=gpt-4o-mini

# PostgreSQL Database (use your actual credentials)
DATABASE_URL=postgresql://username:password@host:port/database?sslmode=require

# n8n Webhook (use your actual webhook URL)
N8N_WEBHOOK_URL=http://your-server:5678/webhook/your-webhook-id
EOF
```

**Note**: Replace the placeholders with your actual credentials from `env.example`

### Step 5: Install Dependencies
```bash
source gamlens_env/bin/activate
pip install psycopg2-binary requests
```

### Step 6: Restart Streamlit
```bash
nohup streamlit run streamlit_app.py --server.headless=true --server.port=8501 > streamlit.log 2>&1 &
```

### Step 7: Verify
```bash
ps aux | grep streamlit
tail -f streamlit.log
```

## New Workflow

### 1. Upload CSV via n8n Webhook
- Go to "Dataset Management" tab
- Use "Option 1: Upload via n8n Webhook (Recommended)"
- Upload CSV file
- n8n processes and stores in PostgreSQL

### 2. Select Dataset in Predictions
- Go to "Predictions & AI Recommendations" tab
- Select dataset from dropdown (populated from PostgreSQL)
- Data loads automatically from database

### 3. AI Analysis
- View campaign summary metrics
- Ask preset or custom questions
- Get GPT-powered insights with full ROAS progression data

## Features

### Predictions Page (2_🚀_Train_Predict_Validate_FAQ.py)
- ✅ Dataset selection dropdown from PostgreSQL
- ✅ Campaign summary metrics (Cost, Revenue, Installs, ROAS)
- ✅ AI-powered recommendations with preset questions:
  - "Which campaigns should we pause?"
  - "Required CPI for D30 profitability?"
  - "Which geo is ready to scale?"
- ✅ Custom question input for AI analysis
- ✅ Chat history with clear functionality
- ✅ Full ROAS time-series data sent to GPT
- ✅ Consistent ROAS interpretation (decimal format)

### Datasets Page
- ✅ n8n webhook upload option (recommended)
- ✅ Legacy direct upload option
- ✅ Dataset preview from PostgreSQL
- ✅ Summary statistics (games, channels, platforms, countries, date range)
- ✅ Sample data display (first 10 records)

### AI Helpers (glai/ai_helpers.py)
- `build_payload_for_ai()` - Creates comprehensive data payload for GPT
- `ask_one_question()` - Sends question to GPT with campaign data
- `add_core_kpis()` - Calculates CPI, ARPU, ROAS, ROI 100% day, retention

## Database Schema

### csv_uploads Table
```sql
CREATE TABLE IF NOT EXISTS csv_uploads (
  id SERIAL PRIMARY KEY,
  source_file TEXT NOT NULL,
  game TEXT,
  channel TEXT,
  platform TEXT,
  country TEXT,
  date DATE,
  installs INTEGER,
  cost INTEGER,
  ad_revenue DOUBLE PRECISION,
  revenue DOUBLE PRECISION,
  roas_d0 DOUBLE PRECISION,
  roas_d1 DOUBLE PRECISION,
  roas_d3 DOUBLE PRECISION,
  roas_d7 DOUBLE PRECISION,
  roas_d14 TEXT,
  roas_d30 TEXT,
  roas_d60 TEXT,
  roas_d90 TEXT,
  retention_rate_d1 DOUBLE PRECISION,
  retention_rate_d2 DOUBLE PRECISION,
  retention_rate_d3 DOUBLE PRECISION,
  retention_rate_d7 TEXT,
  retention_rate_d14 TEXT,
  retention_rate_d30 TEXT,
  level_1_events TEXT,
  level_5_events TEXT,
  level_10_events TEXT,
  level_15_events TEXT,
  level_20_events TEXT,
  level_25_events TEXT,
  level_30_events TEXT,
  level_40_events TEXT,
  level_50_events TEXT
);
```

## Testing

1. **Upload Test CSV**:
   - Navigate to `http://170.64.236.80:8501`
   - Go to "Dataset Management"
   - Upload CSV via n8n webhook

2. **Verify in PostgreSQL**:
   ```sql
   SELECT DISTINCT source_file FROM csv_uploads;
   ```

3. **Test Predictions**:
   - Go to "Predictions & AI Recommendations"
   - Select uploaded dataset
   - View metrics and ask AI questions

## Troubleshooting

### Issue: "Error connecting to PostgreSQL"
- Check DATABASE_URL in .env file
- Verify PostgreSQL is accessible from server
- Check firewall rules

### Issue: "No datasets found"
- Ensure CSV was uploaded successfully via n8n webhook
- Check PostgreSQL for data: `SELECT COUNT(*) FROM csv_uploads;`

### Issue: "Error getting AI response"
- Check OPENAI_API_KEY in .env file
- Ensure API key is valid
- Check OpenAI API usage limits

## Next Steps

To merge this feature to main:

```bash
# On local machine
git checkout main
git merge feature/postgresql-n8n-integration
git push origin main
```

Then on server:
```bash
git checkout main
git pull origin main
```

This ensures the server is always on the stable `main` branch.

