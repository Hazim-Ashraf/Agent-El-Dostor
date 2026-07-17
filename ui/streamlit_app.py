"""Minimal Streamlit testing GUI for Agent El-Dostor (M0).

Ask a question about Egyptian law; watch the agent retrieve + reason, then see
its answer and the tool-call trace. Grows into the full contract-upload GUI (M2).
"""
from __future__ import annotations

import streamlit as st

from app.agent.loop import run_agent
from app.core.config import settings
from app.retrieval import store

st.set_page_config(page_title="Agent El-Dostor", page_icon="⚖️")

st.title("⚖️ Agent El-Dostor")
st.caption("Egyptian contract-intelligence agent — grounded, tool-calling. (M0 skeleton)")
st.warning(
    "General legal information only — not a substitute for advice from a licensed "
    "Egyptian lawyer. The knowledge base currently contains **SAMPLE** data."
)


def _kb_count() -> int | None:
    try:
        conn = store.connect(retries=1, delay=0.0)
        try:
            return store.count(conn)
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 - surfaced in the sidebar
        return None


with st.sidebar:
    st.subheader("Status")
    st.write(f"Reasoning model: `{settings.reasoning_model}`")
    st.write(f"Embeddings: `{settings.embedding_model}`")
    n = _kb_count()
    if n is None:
        st.error("Database not reachable yet.")
    elif n == 0:
        st.warning("Knowledge base is empty — run ingestion (see README).")
    else:
        st.success(f"{n} legislation chunks loaded.")

question = st.text_area(
    "Ask a question about Egyptian law",
    placeholder="e.g. What is the maximum probation period for an employee?",
)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Thinking (retrieving + reasoning)…"):
        result = run_agent(question.strip())

    st.markdown("### Answer")
    st.markdown(result["answer"] or "_(no answer produced)_")

    trace = result.get("trace", [])
    with st.expander(f"Trace — {len(trace)} tool call(s) over {result.get('steps', 0)} step(s)"):
        if not trace:
            st.write("The agent answered without calling any tools.")
        for entry in trace:
            st.markdown(f"**{entry['tool']}** · args: `{entry['args']}`")
            st.code(entry["result"], language="json")
