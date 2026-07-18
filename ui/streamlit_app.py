"""Agent El-Dostor — testing GUI (M2 + M5).

Two modes:
  • Review   — upload a contract, chat with the grounded agent, inspect the
               tool-call trace, citations, and verification gate.
  • Generate — draft a bilingual (AR/EN) contract grounded in Egyptian law and
               download it as a correctly-formatted PDF.

A token + USD usage panel tracks the whole session.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from app.agent.loop import run_agent
from app.core.config import settings
from app.core.usage import Usage
from app.generation.loop import generate_contract, refine_contract
from app.generation.pdf import render_contract_pdfs
from app.generation.schema import GeneratedContract
from app.ingestion.contracts import ingest_contract_bytes
from app.retrieval import store

st.set_page_config(page_title="Agent El-Dostor", page_icon="⚖️", layout="wide")

# --- session state ---------------------------------------------------------- #
st.session_state.setdefault("messages", [])           # [{role, content, meta?}]
st.session_state.setdefault("active_contract", None)  # {contract_id, filename, ...}
st.session_state.setdefault("session_usage", Usage())
st.session_state.setdefault("generated", None)        # {result, pdfs: [(name, bytes)]}

st.session_state.setdefault("gen_messages", [])       # Chat history for generation mode
st.session_state.setdefault("gen_contract_obj", None) # The current GeneratedContract object


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

_GEN_BADGE = {
    "grounded": "✅ Grounded — every legal reference resolves to a real article.",
    "partial": "⚠️ Partially grounded — some references didn't resolve and were dropped.",
    "unverified": "⚠️ No cited article resolved to the knowledge base.",
    "none": "ℹ️ No legal references attached (is the legislation KB loaded?).",
}

_LANG_OPTS = {"bilingual": "Bilingual (Arabic + English)", "ar": "Arabic only", "en": "English only"}
_LAYOUT_OPTS = {
    "side_by_side": "Side-by-side (English | Arabic)",
    "sequential": "Sequential (English, then Arabic)",
    "separate": "Two separate PDFs",
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

    mode = st.radio("Mode", ["📄 Review a contract", "🖊️ Generate a contract"])
    if mode.startswith("🖊️"):
        st.divider()
        st.subheader("Contract Settings")
        st.session_state.setdefault("gen_ctype", "employment")
        st.session_state.setdefault("gen_lang", "bilingual")
        
        ctype = st.selectbox(
            "Contract type", 
            ["employment", "rental", "shareholder", "service", "commercial"],
            index=["employment", "rental", "shareholder", "service", "commercial"].index(st.session_state.gen_ctype)
        )
        language = st.selectbox(
            "Language", 
            list(_LANG_OPTS), 
            format_func=_LANG_OPTS.get,
            index=list(_LANG_OPTS).index(st.session_state.gen_lang)
        )
        
        # If settings change, we should probably warn or reset, but let's just update state.
        st.session_state.gen_ctype = ctype
        st.session_state.gen_lang = language

        if st.button("New Contract (Reset)", type="primary"):
            st.session_state.gen_messages = []
            st.session_state.gen_contract_obj = None
            st.session_state.generated = None
            st.rerun()

    st.divider()
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
    st.caption(f"{u.calls} model call(s) · prompt {u.prompt_tokens:,} / completion {u.completion_tokens:,}")
    if not u.cost_known:
        st.caption("⚠️ This model/provider didn't report a cost.")
    if st.button("Reset usage"):
        st.session_state.session_usage = Usage()
        st.rerun()

# --- header ----------------------------------------------------------------- #
st.title("⚖️ Agent El-Dostor")
st.warning(
    "General legal information only — not a substitute for a licensed Egyptian lawyer. "
    "Generated contracts are **drafts** to be reviewed by a lawyer before use."
)


# =========================================================================== #
# MODE: REVIEW
# =========================================================================== #
def review_mode() -> None:
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
            st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Request failed: {exc}"})
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


# =========================================================================== #
# MODE: GENERATE
# =========================================================================== #
def _render_preview(contract: dict[str, Any]) -> None:
    if contract.get("title_en"):
        st.markdown(f"### {contract['title_en']}")
    if contract.get("title_ar"):
        st.markdown(f"#### {contract['title_ar']}")
    meta = " · ".join(x for x in [contract.get("place"), contract.get("date")] if x)
    if meta:
        st.caption(meta)

    if contract.get("parties"):
        st.markdown("**Parties**")
        for p in contract["parties"]:
            bits = [p.get("role_en") or p.get("role_ar"), p.get("name_en") or p.get("name_ar")]
            st.markdown("- " + " — ".join(b for b in bits if b))

    with st.expander("Drafted clauses", expanded=True):
        for c in contract.get("clauses", []):
            head_en = f"**Article ({c['number']}) {c.get('heading_en', '')}**".strip()
            st.markdown(head_en)
            if c.get("body_en"):
                st.write(c["body_en"])
            if c.get("heading_ar") or c.get("body_ar"):
                st.markdown(f"**المادة ({c['number']}) {c.get('heading_ar', '')}**")
                if c.get("body_ar"):
                    st.write(c["body_ar"])
            if c.get("legal_basis"):
                refs = "؛ ".join(
                    f"{b.get('ref', '')}{(' — ' + b['law']) if b.get('law') else ''}"
                    for b in c["legal_basis"]
                )
                st.caption(f"⚖️ Legal basis: {refs}")
            st.divider()


def generate_mode() -> None:
    st.subheader("🖊️ Generate a contract")
    st.caption(
        "Describe the contract you want. The agent will draft it clause-by-clause, "
        "and cite the Egyptian Civil Code. You can ask for changes iteratively."
    )
    if status and status["kb"] == 0:
        st.warning("The legislation KB is empty — clauses can't be grounded. Ingest it first (README step 3).")

    # Display chat history
    for message in st.session_state.gen_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("meta"):
                render_trace(message["meta"])

    prompt = st.chat_input("Describe the contract or ask for changes (e.g. 'Make the notice period 3 months')...")
    
    if prompt:
        st.session_state.gen_messages.append({"role": "user", "content": prompt})
        
        ctype = st.session_state.gen_ctype
        language = st.session_state.gen_lang
        
        # History format expected by the LLM
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.gen_messages[:-1]
        ][-6:]
        
        try:
            current_contract = st.session_state.gen_contract_obj
            if current_contract is None:
                # Initial generation
                with st.spinner("Researching the law + drafting initial contract…"):
                    result = generate_contract(ctype, language=language, brief=prompt)
            else:
                # Refinement
                with st.spinner("Researching and applying your changes…"):
                    result = refine_contract(
                        current_contract=current_contract,
                        user_request=prompt,
                        contract_type=ctype,
                        language=language,
                        history=history
                    )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Generation failed: {exc}")
            st.session_state.gen_messages.append({"role": "assistant", "content": f"⚠️ Request failed: {exc}"})
            st.rerun()
            return
            
        st.session_state.session_usage.add(Usage(**result["usage"]))
        
        if result.get("status") in ("generated", "failed") and result.get("contract"):
            # Update the stored contract object
            st.session_state.gen_contract_obj = GeneratedContract.model_validate(result["contract"])
            st.session_state.generated = {"result": result, "pdfs": []}
            
            reply = "I have updated the contract draft." if current_contract else "Here is the initial contract draft."
        else:
            reply = f"Failed to produce the contract. {result.get('error', '')}"

        st.session_state.gen_messages.append(
            {
                "role": "assistant",
                "content": reply,
                "meta": {
                    "trace": result.get("trace", []),
                    "steps": result.get("steps", 0),
                    "usage": result.get("usage", {}),
                    "verification": result.get("verification", {}),
                },
            }
        )
        st.rerun()

    gen = st.session_state.generated
    if not gen:
        return

    result = gen["result"]
    contract = result.get("contract")
    if not contract:
        return

    ver = result.get("verification", {})
    st.divider()
    st.caption(
        f"{_GEN_BADGE.get(ver.get('status'), ver.get('status', ''))}  ·  "
        f"{ver.get('n_verified', 0)}/{ver.get('n_legal_basis', 0)} legal references verified"
        f" across {ver.get('n_clauses', 0)} clause(s)."
    )
    if ver.get("dropped"):
        with st.expander(f"{len(ver['dropped'])} citation(s) dropped (didn't resolve)"):
            st.json(ver["dropped"])

    _render_preview(contract)

    # ---- PDF export ----
    st.subheader("📑 Export PDF")
    language = contract.get("language", "bilingual")
    layout = "side_by_side"
    if language == "bilingual":
        layout = st.radio("Bilingual layout", list(_LAYOUT_OPTS), format_func=_LAYOUT_OPTS.get, horizontal=True)
    if st.button("Build PDF"):
        try:
            with st.spinner("Rendering PDF…"):
                st.session_state.generated["pdfs"] = render_contract_pdfs(contract, language, layout)
        except Exception as exc:  # noqa: BLE001
            st.error(f"PDF rendering failed: {exc}")

    for name, data in st.session_state.generated.get("pdfs", []):
        st.download_button(
            f"⬇️ Download {name}", data=data, file_name=name, mime="application/pdf", key=f"dl_{name}"
        )


if mode.startswith("📄"):
    review_mode()
else:
    generate_mode()
