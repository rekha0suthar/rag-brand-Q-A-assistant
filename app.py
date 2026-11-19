import streamlit as st
from chain import load_chain

st.set_page_config(page_title="RAG for Marketing Briefs", layout="centered")
st.title("RAG for Marketing Briefs")

@st.cache_resource(show_spinner=False)
def get_chain():
    # Cached across reruns -- Streamlit re-executes this script on every
    # interaction, and reloading embeddings + FAISS each time is seconds of lag.
    return load_chain()

with st.spinner("Loading embeddings and index (first run only)…"):
    ask = get_chain()

q = st.text_input("Ask about the brand/campaign/compliance…", placeholder="e.g., Propose 2 ad copy options in our brand tone for Diwali sale.")
if st.button("Ask") and q:
    with st.spinner("Thinking..."):
        out = ask(q)
    st.write("### Answer")
    st.write(out["answer"])
    st.caption(f"Sources: {', '.join(out['sources'])} | Latency: {out['latency']}s")

st.divider()
st.markdown("**Tip:** If the answer isn't in the docs, you'll see 'Not enough info in the docs.' Add or update files in `/docs` and re-run `ingest.py`.")
