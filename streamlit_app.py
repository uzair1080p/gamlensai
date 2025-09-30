import streamlit as st
import os
import sys
from pathlib import Path

# Add gamlens to path
sys.path.append(str(Path(__file__).parent / "gamlens"))

# Import gamlens modules
from lib.utils import ensure_dirs, read_csv_strict, DATA_DIR
from lib.kpis import add_core_kpis
from lib.ai import build_payload_for_ai, ask_one_question
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="GameLens AI - UA Analytics",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Main title
st.markdown('<h1 class="main-header">🎮 GameLens AI - UA Analytics</h1>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["📊 Dashboard", "📦 Model Datasets", "🧠 Model Training & Predictions", "❓ FAQ"]
)

# Ensure data directory exists
ensure_dirs()

if page == "📊 Dashboard":
    st.title("📊 Dashboard")
    
    # Dataset selection
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not files:
        st.info("No datasets found. Upload one on the **Model Datasets** page.")
    else:
        dataset = st.selectbox("Choose dataset", files, index=0)
        path = os.path.join(DATA_DIR, dataset)
        
        # Cache dataset + KPIs per selected file
        if "df_kpi" not in st.session_state or st.session_state.get("ds_name") != dataset:
            try:
                df = read_csv_strict(path)
                df_kpi, table = add_core_kpis(df)
                st.session_state.df_kpi = df_kpi
                st.session_state.table = table
                st.session_state.ds_name = dataset
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")
                st.stop()
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_cost = st.session_state.df_kpi["cost"].sum()
            st.metric("Total Cost", f"${total_cost:,.2f}")
        with col2:
            total_revenue = st.session_state.df_kpi["revenue"].sum()
            st.metric("Total Revenue", f"${total_revenue:,.2f}")
        with col3:
            total_installs = st.session_state.df_kpi["installs"].sum()
            st.metric("Total Installs", f"{total_installs:,.0f}")
        with col4:
            avg_roas = st.session_state.df_kpi["ROAS"].mean()
            st.metric("Average ROAS", f"{avg_roas:.2f}")
        
        # Summary table
        st.subheader("Campaign Summary")
        st.dataframe(st.session_state.table, use_container_width=True)

elif page == "📦 Model Datasets":
    st.title("📦 Model Datasets")
    
    # File upload
    uploaded = st.file_uploader("Upload data template CSV", type=["csv"])
    if uploaded:
        path = os.path.join(DATA_DIR, uploaded.name)
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Saved: {path}")
        
        # Clear session state to force reload
        if "ds_name" in st.session_state:
            del st.session_state["ds_name"]
    
    # Display stored datasets
    st.subheader("Stored datasets")
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not files:
        st.info("No datasets yet. Upload a CSV above.")
    else:
        for f in files:
            with st.expander(f"Preview • {f}", expanded=False):
                try:
                    df = read_csv_strict(os.path.join(DATA_DIR, f))
                    st.dataframe(df.head(15), use_container_width=True)
                except Exception as e:
                    st.error(str(e))

elif page == "🧠 Model Training & Predictions":
    st.title("🧠 Model Training & Predictions")
    
    # Dataset selection
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    dataset = st.selectbox("Choose dataset", files, index=0 if files else None)
    if not dataset:
        st.info("No datasets found. Upload one on the **Model Datasets** page.")
        st.stop()

    path = os.path.join(DATA_DIR, dataset)

    # Cache dataset + KPIs per selected file
    if "df_kpi" not in st.session_state or st.session_state.get("ds_name") != dataset:
        try:
            df = read_csv_strict(path)
            df_kpi, table = add_core_kpis(df)
            st.session_state.df_kpi = df_kpi
            st.session_state.table = table
            st.session_state.ds_name = dataset
            st.session_state.chat = []
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            st.stop()

    # Summary table (always first)
    st.caption("Summary table (always shown first)")
    st.dataframe(st.session_state.table, use_container_width=True)

    # Optional classical model block
    with st.expander("⚙️ Optional: Train a baseline model (placeholder)"):
        st.write(
            "Add your LightGBM/XGBoost training and validation here if you want a non-LLM baseline "
            "for ROAS/LTV forecasting (e.g., fit per-channel ROAS_D30 and compare with GPT answers)."
        )

    # Ask GPT (one question at a time)
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

    # Chat history
    st.subheader("Chat")
    for role, msg in st.session_state.get("chat", []):
        st.markdown(f"**{role}:** {msg}")

elif page == "❓ FAQ":
    st.title("❓ FAQ")
    
    st.info("""
    **GameLens AI FAQ**
    
    **Q: What data format should I use?**
    A: Upload CSV files that match the data template schema with columns: game, channel, platform, country, date, installs, cost, ad_revenue, revenue, roas_d0-d90, retention_rate_d1-d30, level_1_events-level_50_events.
    
    **Q: How do I get started?**
    A: 1. Upload your CSV data on the Model Datasets page, 2. Select it on the Dashboard or Model Training page, 3. Ask questions using the AI assistant.
    
    **Q: What questions can I ask?**
    A: You can ask about campaign performance, ROI analysis, scaling recommendations, and more. Use the preset buttons or type your own questions.
    
    **Q: Do I need an OpenAI API key?**
    A: Yes, for AI recommendations. Set OPENAI_API_KEY in your .env file or enter it in the UI.
    """)

# Footer
st.markdown("---")
st.markdown("**GameLens AI** - UA Analytics Platform")
