import os
import streamlit as st
from lib.utils import ensure_dirs, read_csv_strict, DATA_DIR

st.title("📦 Model Datasets")
ensure_dirs()

uploaded = st.file_uploader("Upload data template CSV", type=["csv"])
if uploaded:
    path = os.path.join(DATA_DIR, uploaded.name)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success(f"Saved: {path}")

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