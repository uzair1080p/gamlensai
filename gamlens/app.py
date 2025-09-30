import streamlit as st

st.set_page_config(page_title="UA Analytics", layout="wide")

st.sidebar.title("UA Analytics")
st.sidebar.page_link("pages/1_📦_Model_Datasets.py", label="Model Datasets")
st.sidebar.page_link("pages/2_🧠_Model_Training_&_Predictions.py", label="Model Training & Predictions")

st.title("UA Analytics")
st.write("Use the sidebar to navigate.")