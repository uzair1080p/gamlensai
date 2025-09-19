import streamlit as st
from glai.usage import get_usage

st.set_page_config(page_title="GPT Usage", page_icon="💸", layout="wide")

st.title("💸 GPT Usage & Cost")

data = get_usage()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total input tokens", f"{int(data.get('total_input', 0)):,}")
with col2:
    st.metric("Total output tokens", f"{int(data.get('total_output', 0)):,}")
with col3:
    st.metric("Total cost (USD)", f"${float(data.get('total_cost', 0.0)):.6f}")

st.subheader("Events")
events = data.get("events", [])[::-1]
if events:
    for evt in events[:200]:
        with st.expander(f"{evt.get('ts')} • {evt.get('kind')} • {evt.get('model')} • ${evt.get('cost')}"):
            st.json(evt)
else:
    st.info("No GPT calls have been recorded yet. Run predictions/FAQs to populate usage.")

st.caption("Pricing assumed: GPT-5 mini — $0.25/M input tokens, $2.00/M output tokens. Adjust in glai/usage.py if needed.")


