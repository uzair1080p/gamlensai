# streamlit_app/pages/2_🧠_Model_Training_&_Predictions.py

import os
import streamlit as st
from dotenv import load_dotenv

from lib.utils import ensure_dirs, read_csv_strict, DATA_DIR
from lib.kpis import add_core_kpis
from lib.ai import build_payload_for_ai, ask_one_question

# ---------- Page setup ----------
st.set_page_config(page_title="Model Training & Predictions", layout="wide")
st.title("🧠 Model Training & Predictions")

# Load .env so OPENAI_API_KEY can be used automatically
load_dotenv()

# Ensure data dir exists
ensure_dirs()

# ---------- Dataset selection ----------
files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
dataset = st.selectbox("Choose dataset", files, index=0 if files else None)
if not dataset:
    st.info("No datasets found. Upload one on the **Model Datasets** page.")
    st.stop()

path = os.path.join(DATA_DIR, dataset)

# Cache dataset + KPIs per selected file
if "df_kpi" not in st.session_state or st.session_state.get("ds_name") != dataset:
    df = read_csv_strict(path)
    df_kpi, table = add_core_kpis(df)
    st.session_state.df_kpi = df_kpi
    st.session_state.table = table
    st.session_state.ds_name = dataset
    st.session_state.chat = []

# ---------- Summary table (always first) ----------
st.caption("Summary table (always shown first)")
st.dataframe(st.session_state.table, use_container_width=True)

# ---------- Optional classical model block ----------
with st.expander("⚙️ Optional: Train a baseline model (placeholder)"):
    st.write(
        "Add your LightGBM/XGBoost training and validation here if you want a non-LLM baseline "
        "for ROAS/LTV forecasting (e.g., fit per-channel ROAS_D30 and compare with GPT answers)."
    )

# ---------- Ask GPT (one question at a time) ----------
st.subheader("Ask GPT (one question at a time)")

# API key input (optional if OPENAI_API_KEY in .env)
env_key = os.getenv("OPENAI_API_KEY", "").strip()
api_key = st.text_input(
    "OpenAI API Key (optional if OPENAI_API_KEY is set in .env)",
    type="password",
    value=""  # leave empty to rely on .env
)

# Free-text question
q = st.text_input("Your question (e.g., 'When will ROI reach 100% per channel?')")

# Preset question buttons
c1, c2, c3 = st.columns(3)
with c1:
    b1 = st.button("Which campaigns should we pause?")
with c2:
    b2 = st.button("Required CPI for D30 profitability?")
with c3:
    b3 = st.button("Which geo is ready to scale?")

ask = st.button("Ask")

# Handle ask / presets
if ask or b1 or b2 or b3:
    # Determine question text
    question = (
        q if ask and q else
        "Which campaigns should be paused due to poor ROI/retention? Provide bullets with thresholds and reasons."
        if b1 else
        "What CPI is needed to reach profitability by D30 per channel? Provide a table with assumptions."
        if b2 else
        "Which geo is ready to scale aggressively? Provide criteria, data support, and risks."
    )

    # Build compact payload for the current dataset
    payload = build_payload_for_ai(st.session_state.df_kpi, dataset)

    try:
        # ask_one_question() will use the UI key if provided,
        # otherwise it falls back to OPENAI_API_KEY from .env
        answer = ask_one_question(api_key, question, payload)
        st.session_state.chat.append(("You", question))
        st.session_state.chat.append(("AI", answer))
    except Exception as e:
        st.error(str(e))

# ---------- Chat history ----------
st.subheader("Chat")
for role, msg in st.session_state.get("chat", []):
    st.markdown(f"**{role}:** {msg}")