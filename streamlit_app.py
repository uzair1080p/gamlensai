import streamlit as st

st.set_page_config(page_title="GameLens AI", layout="wide")

st.sidebar.title("GameLens AI")
st.sidebar.page_link("pages/2_🚀_Train_Predict_Validate_FAQ.py", label="🚀 Train Predict Validate FAQ")
st.sidebar.page_link("pages/4_💸_GPT_Usage.py", label="💸 GPT Usage")

st.title("GameLens AI - ROAS Forecasting Dashboard")
st.write("Use the sidebar to navigate to different sections.")