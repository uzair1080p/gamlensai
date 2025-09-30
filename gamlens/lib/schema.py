# Template schema (data template CSV schema in use)
TEMPLATE_COLS = [
    "game","channel","platform","country","date","installs","cost","ad_revenue","revenue",
    "roas_d0","roas_d1","roas_d3","roas_d7","roas_d14","roas_d30","roas_d60","roas_d90",
    "retention_rate_d1","retention_rate_d2","retention_rate_d3","retention_rate_d7",
    "retention_rate_d14","retention_rate_d30",
    "level_1_events","level_5_events","level_10_events","level_15_events","level_20_events",
    "level_25_events","level_30_events","level_40_events","level_50_events"
]

DIM_COLS = ["game","channel","platform","country","date"]
NUM_COLS = [c for c in TEMPLATE_COLS if c not in DIM_COLS]

ROAS_COLS = ["roas_d0","roas_d1","roas_d3","roas_d7",
             "roas_d14","roas_d30","roas_d60","roas_d90"]