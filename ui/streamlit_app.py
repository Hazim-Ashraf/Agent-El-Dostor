"""Agent El-Dostor — full testing GUI (M2).

Upload a contract, chat with the grounded agent, inspect the tool-call trace and
citations, and track token + USD usage for the session.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from app.agent.loop import run_agent
from app.core.config import settings
from app.core.usage import Usage
from app.ingestion.contracts import ingest_contract_bytes
from app.retrieval import store

st.set_page_config(page_title="Agent El-Dostor", page_icon="⚖️", layout="wide")

# --- session state ---------------------------------------------------------- #
st.session_state.setdefault("messages", [])          # [{role, content, meta?}]
st.session_state.setdefault("active_contract", None)  # {contract_id, filename, ...}
st.session_state.setdefault("session_usage", Usage())


def get_status() -> dict[str, Any] | None:
    try:
        conn = store.connect(retries=1, delay=0.0)
        try:
            return {"kb": store.count(conn), "contracts": store.list_contracts(conn)}
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 - reported in the sidebar
        return None


_VERIFICATION_BADGE = {
    "passed": "✅ Verified — every finding is grounded in a cited source.",
    "partial": "⚠️ Partially verified — some ungrounded findings were dropped.",
    "ungated": "ℹ️ Answered without the structured verification gate.",
    "unverified": "⚠️ Could not produce a verified answer within the step limit.",
}


def render_trace(meta: dict[str, Any]) -> None:
    trace = meta.get("trace", [])
    usage = meta.get("usage", {}) or {}
    verification = meta.get("verification") or {}
    status = verification.get("status")
    if status:
        badge = _VERIFICATION_BADGE.get(status, f"Verification: {status}")
        st.caption(f"{badge}  ·  {verification.get('findings', 0)} grounded finding(s)")

    header = (
        f"Trace — {len(trace)} tool call(s), {meta.get('steps', 0)} step(s) · "
        f"{usage.get('total_tokens', 0):,} tokens · ${usage.get('cost_usd', 0.0):.4f}"
    )
    with st.expander(header):
        if not trace:
            st.write("The agent answered without calling any tools.")
        for entry in trace:
            st.markdown(f"**{entry['tool']}** · `{entry['args']}`")
            st.code(entry["result"], language="json")


status = get_status()

# --- sidebar ---------------------------------------------------------------- #
with st.sidebar:
    st.header("⚖️ Agent El-Dostor")
    st.caption("Egyptian contract intelligence — grounded, tool-calling.")

    st.subheader("Status")
    st.write(f"Model: `{settings.reasoning_model}`")
    st.write(f"Embeddings: `{settings.embedding_model}`")
    if status is None:
        st.error("Database not reachable.")
    else:
        st.write(f"Legislation chunks: **{status['kb']}**")
        st.write(f"Contracts stored: **{len(status['contracts'])}**")
        if status["kb"] == 0:
            st.warning("KB empty — run legislation ingestion (see README).")

    st.divider()
    st.subheader("💰 Usage (this session)")
    u: Usage = st.session_state.session_usage
    col1, col2 = st.columns(2)
    col1.metric("Tokens", f"{u.total_tokens:,}")
    col2.metric("Cost (USD)", f"${u.cost_usd:.4f}" if u.cost_known else f"~${u.cost_usd:.4f}")
    st.caption(
        f"{u.calls} model call(s) · prompt {u.prompt_tokens:,} / completion {u.completion_tokens:,}"
    )
    if not u.cost_known:
        st.caption("⚠️ This model/provider didn't report a cost.")
    if st.button("Reset usage"):
        st.session_state.session_usage = Usage()
        st.rerun()

# --- header ----------------------------------------------------------------- #
st.title("⚖️ Agent El-Dostor")
st.warning(
    "General legal information only — not a substitute for a licensed Egyptian lawyer. "
    "The knowledge base currently contains **SAMPLE** data."
)

# --- contract section ------------------------------------------------------- #
st.subheader("📄 Contract")
active = st.session_state.active_contract
if active:
    st.success(
        f"Active contract: **{active['filename']}** "
        f"({active.get('contract_type')}, {active.get('n_clauses')} clauses)"
    )
    if st.button("Clear active contract"):
        st.session_state.active_contract = None
        st.rerun()

with st.expander("Upload or select a contract", expanded=active is None):
    uploaded = st.file_uploader(
        "Upload (PDF, DOCX, TXT, or image — scanned files are OCR'd)",
        type=["pdf", "docx", "txt", "md", "png", "jpg", "jpeg", "tiff", "bmp"],
    )
    contract_type = st.selectbox(
        "Contract type",
        ["employment", "rental", "shareholder", "service", "commercial", "unknown"],
    )
    if st.button("Ingest contract", disabled=uploaded is None):
        with st.spinner("Parsing, segmenting, embedding… (first run downloads the model)"):
            try:
                meta = ingest_contract_bytes(uploaded.name, uploaded.getvalue(), contract_type)
                st.session_state.active_contract = meta
                st.success(f"Ingested {meta['n_clauses']} clauses from {meta['filename']}.")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Ingestion failed: {exc}")

    if status and status["contracts"]:
        st.markdown("**Or reuse a previously uploaded contract:**")
        options = {
            f"{c['filename']} ({c['contract_type']}, {c['n_clauses']} cl.)": c
            for c in status["contracts"]
        }
        pick = st.selectbox("Existing contracts", ["—", *options.keys()])
        if pick != "—" and st.button("Use selected"):
            c = options[pick]
            st.session_state.active_contract = {
                "contract_id": c["id"],
                "filename": c["filename"],
                "contract_type": c["contract_type"],
                "lang": c.get("lang"),
                "n_clauses": c["n_clauses"],
            }
            st.rerun()

# --- chat ------------------------------------------------------------------- #
st.subheader("💬 Ask")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("meta"):
            render_trace(message["meta"])

prompt = st.chat_input("Ask about your contract or Egyptian law…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    contract_id = active["contract_id"] if active else None
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ][-8:]

    try:
        with st.spinner("Retrieving + reasoning…"):
            result = run_agent(prompt, contract_id=contract_id, history=history)
    except Exception as exc:  # noqa: BLE001
        st.session_state.messages.append(
            {"role": "assistant", "content": f"⚠️ Request failed: {exc}"}
        )
        st.rerun()

    st.session_state.session_usage.add(Usage(**result["usage"]))
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"] or "_(no answer produced)_",
            "meta": {
                "trace": result.get("trace", []),
                "steps": result.get("steps", 0),
                "usage": result.get("usage", {}),
                "verification": result.get("verification", {}),
            },
        }
    )
    st.rerun()
