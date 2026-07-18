"""Agent El-Dostor — the GUI.

Design goals:
  • Trustworthy, calm, institutional look (ink-navy + gold, serif headings).
  • The agent SHOWS ITS THINKING live, Claude-Code-style: each retrieval step,
    what it found, and the verification gate's verdict stream into a status box;
    a collapsed "Reasoning" record stays attached to every answer.
  • No debug panels: raw tool payloads, traces, and token/cost tracking live in
    the logs (`docker compose logs app`, logs/agent_runs.jsonl) — not the GUI.
  • After a review, the uploaded contract is shown as an ANNOTATED MAP: clauses
    grounded in Egyptian law are highlighted gold with their article chips;
    risky clauses are highlighted red with the risk explained.
"""
from __future__ import annotations

import html
import time
from typing import Any

import streamlit as st

from app.agent.loop import run_agent
from app.core.config import settings
from app.core.logging import get_logger
from app.generation.loop import generate_contract, refine_contract
from app.generation.pdf import render_contract_pdfs
from app.generation.schema import GeneratedContract
from app.ingestion.contracts import ingest_contract_bytes
from app.retrieval import store

log = get_logger("ui")

st.set_page_config(page_title="Agent El-Dostor", page_icon="⚖️", layout="wide")

# --------------------------------------------------------------------------- #
# Design system (CSS)
# --------------------------------------------------------------------------- #
st.markdown(
    """
<style>
#MainMenu, footer, .stDeployButton {visibility: hidden;}
/* Slightly larger type everywhere (Streamlit sizes are rem-based). */
html {font-size: 17.5px;}
h1, h2, h3 {font-family: Georgia, 'Times New Roman', serif; letter-spacing: .01em; color: #24231F;}

/* Hero */
.eld-hero {padding: 1.05rem 1.4rem; border: 1px solid rgba(140,109,20,.35); border-radius: 14px;
  background: linear-gradient(135deg, rgba(201,162,39,.14), rgba(251,248,241,0) 60%); margin-bottom: .55rem;}
.eld-hero h1 {margin: 0; font-size: 1.95rem;}
.eld-hero .tag {color: #6f6a5d; font-size: 1rem; margin-top: .2rem;}

/* Chips */
.chips {display: flex; gap: .45rem; flex-wrap: wrap; margin: .35rem 0 .9rem;}
.chip {display: inline-block; padding: .18rem .66rem; border-radius: 999px; font-size: .84rem;
  border: 1px solid rgba(0,0,0,.14); background: rgba(0,0,0,.03); color: #55524a;}
.chip-gold {border-color: rgba(140,109,20,.5); background: rgba(201,162,39,.10); color: #7A5F10;}
.chip-red  {border-color: rgba(179,38,30,.45); background: rgba(179,38,30,.07);  color: #A32B24;}
.chip-green{border-color: rgba(30,122,52,.45); background: rgba(30,122,52,.07);  color: #1E7A34;}
.chip-dim  {opacity: .8;}

/* Finding cards */
.eld-card {border: 1px solid rgba(0,0,0,.10); border-left: 4px solid var(--c, #8a8577);
  border-radius: 10px; padding: .65rem .9rem; margin: .45rem 0; background: #FFFFFF;}
.eld-card.risk {--c:#C6363C;} .eld-card.right {--c:#2E8B44;} .eld-card.obligation {--c:#2F6FBF;}
.eld-card.action {--c:#8C6D14;} .eld-card.info {--c:#8a8577;}
.eld-card .head {font-size: .82rem; color: #6f6a5d; margin-bottom: .2rem; text-transform: uppercase; letter-spacing: .06em;}
.eld-card .claim {line-height: 1.6;}
.eld-card .cites {margin-top: .45rem; display: flex; gap: .35rem; flex-wrap: wrap;}

/* Clause map cards */
.cl-card {border: 1px solid rgba(0,0,0,.10); border-left: 4px solid rgba(0,0,0,.18);
  border-radius: 10px; padding: .7rem .95rem; margin: .5rem 0; background: #FFFFFF;}
.cl-card.law  {border-left-color: #8C6D14; background: #FDF9EC;}
.cl-card.risk {border-left-color: #C6363C; background: #FDF1F0;}
.cl-card .cl-head {font-size: .84rem; color: #6f6a5d; margin-bottom: .25rem; letter-spacing: .04em;}
.cl-card .cl-text {line-height: 1.75; white-space: pre-wrap;}
.cl-card .cl-tags {margin-top: .5rem; display: flex; gap: .35rem; flex-wrap: wrap;}
mark {background: rgba(201,162,39,.35); color: inherit; padding: 0 .12em; border-radius: 3px;}
.cl-card.risk mark {background: rgba(198,54,60,.22);}

/* Thinking lines */
.think-line {font-size: .95rem; line-height: 1.6; margin: .14rem 0;}
.think-line b {color: #7A5F10;}
.dim {color: #6f6a5d;} .mono {font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .86rem;}

/* Disclaimer bar */
.eld-disclaimer {border-top: 1px solid rgba(0,0,0,.12); margin-top: 1.6rem; padding-top: .55rem;
  color: #7d7869; font-size: .86rem;}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
st.session_state.setdefault("messages", [])            # consult chat [{role, content, findings?, verification?, events?}]
st.session_state.setdefault("active_contract", None)   # {contract_id, filename, ...}
st.session_state.setdefault("annotations", {})         # contract_id -> {clause_id: {laws, risks, notes, quotes}}
st.session_state.setdefault("gen_messages", [])        # generation chat
st.session_state.setdefault("gen_contract_obj", None)  # current GeneratedContract
st.session_state.setdefault("generated", None)         # {result, pdfs: [(name, bytes)]}
st.session_state.setdefault("gen_ctype", "employment")
st.session_state.setdefault("gen_lang", "bilingual")

_LANG_OPTS = {"bilingual": "Bilingual (Arabic + English)", "ar": "Arabic only", "en": "English only"}
_LAYOUT_OPTS = {
    "side_by_side": "Side-by-side (English | Arabic)",
    "sequential": "Sequential (English, then Arabic)",
    "separate": "Two separate PDFs",
}
_CTYPES = ["employment", "rental", "shareholder", "service", "commercial"]

_TOOL_LABEL = {
    "search_legislation": "Searching Egyptian legislation",
    "get_legal_article": "Reading a law article",
    "search_contract": "Reading the contract",
    "get_contract_clause": "Fetching a contract clause",
}

_TYPE_META = {
    "risk": ("⚠️", "Risk", "risk"),
    "right": ("🛡️", "Right", "right"),
    "obligation": ("📌", "Obligation", "obligation"),
    "action": ("🧭", "Suggested action", "action"),
    "info": ("ℹ️", "Note", "info"),
}

_VERIFY_CHIP = {
    "passed": '<span class="chip chip-green">✓ Verified — every claim cited &amp; checked against its source</span>',
    "partial": '<span class="chip chip-gold">◐ Partially verified — unverifiable claims were removed</span>',
    "unverified": '<span class="chip chip-red">⚠ Could not verify an answer within limits</span>',
    "ungated": "",
}

_GEN_CHIP = {
    "grounded": '<span class="chip chip-green">✓ Grounded — every legal reference resolves to a real article</span>',
    "partial": '<span class="chip chip-gold">◐ Partially grounded — unresolved references were dropped</span>',
    "unverified": '<span class="chip chip-red">⚠ No cited article resolved to the knowledge base</span>',
    "none": '<span class="chip chip-dim">No legal references attached</span>',
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def get_status() -> dict[str, Any] | None:
    try:
        conn = store.connect(retries=1, delay=0.0)
        try:
            return {"kb": store.count(conn), "contracts": store.list_contracts(conn)}
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 - reported in the sidebar
        return None


def _short_law(law: str | None) -> str:
    if not law:
        return ""
    if "Civil Code" in law or "المدني" in law:
        return "Civil Code"
    if "Labour" in law or "Labor" in law or "العمل" in law:
        return "Labour Law"
    return law if len(law) <= 30 else law[:27] + "…"


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _fmt_args(tool: str | None, args: dict[str, Any]) -> str:
    if args.get("query"):
        return f'“{_clip(str(args["query"]), 70)}”'
    if tool == "get_legal_article":
        return f'{args.get("article_ref", "")} · {_short_law(args.get("law"))}'
    if args.get("clause_ref"):
        return f'clause {args["clause_ref"]}'
    return ""


class ThinkingView:
    """Streams agent events into a live status box, Claude-Code-style."""

    def __init__(self, label: str):
        self.box = st.status(label, expanded=True)
        self.events: list[dict[str, Any]] = []
        self.n_tools = 0
        self.t0 = time.monotonic()

    def __call__(self, ev: dict[str, Any]) -> None:
        self.events.append(ev)
        if ev.get("type") == "tool_call":
            self.n_tools += 1
        self.box.markdown(_event_html(ev), unsafe_allow_html=True)

    def finish(self, ok: bool = True, note: str = "") -> None:
        secs = max(1, int(time.monotonic() - self.t0))
        label = f"Thought for {secs}s · {self.n_tools} tool call(s)"
        if note:
            label += f" · {note}"
        self.box.update(label=label, state="complete" if ok else "error", expanded=False)


def _event_html(ev: dict[str, Any]) -> str:
    kind = ev.get("type")
    if kind == "thinking":
        return f'<div class="think-line dim">✻ {html.escape(_clip(ev.get("text", ""), 300))}</div>'
    if kind == "tool_call":
        label = _TOOL_LABEL.get(ev.get("tool", ""), ev.get("tool", ""))
        args = html.escape(_fmt_args(ev.get("tool"), ev.get("args") or {}))
        return f'<div class="think-line">⏺ <b>{html.escape(label)}</b> <span class="dim mono">{args}</span></div>'
    if kind == "tool_result":
        return f'<div class="think-line dim">&nbsp;&nbsp;⎿&nbsp; {html.escape(ev.get("summary", ""))}</div>'
    if kind == "gate":
        icon = {"passed": "✅", "rejected": "♻️", "invalid": "♻️"}.get(ev.get("status", ""), "🔎")
        return f'<div class="think-line">{icon} {html.escape(ev.get("detail", ""))}</div>'
    return ""


def render_reasoning(events: list[dict[str, Any]]) -> None:
    """The persistent, collapsed thought record attached to an answer."""
    if not events:
        return
    n_tools = sum(1 for e in events if e.get("type") == "tool_call")
    with st.expander(f"🧠 Reasoning — {n_tools} tool call(s)"):
        st.markdown("".join(_event_html(e) for e in events), unsafe_allow_html=True)


def _citation_chips(citations: list[dict[str, Any]]) -> str:
    chips = []
    for c in citations:
        quote = html.escape(_clip(str(c.get("quote", "")), 220))
        if c.get("source") == "law":
            label = html.escape(f"⚖ {c.get('ref', '')} · {_short_law(c.get('law'))}")
            chips.append(f'<span class="chip chip-gold" title="{quote}">{label}</span>')
        else:
            label = html.escape(f"📄 {c.get('ref', '')}")
            chips.append(f'<span class="chip" title="{quote}">{label}</span>')
    return "".join(chips)


def _finding_card(f: dict[str, Any]) -> str:
    icon, label, cls = _TYPE_META.get(f.get("type", "info"), _TYPE_META["info"])
    conf = f.get("confidence") or ""
    head = f"{icon} {label}" + (f' · <span class="dim">{html.escape(conf)} confidence</span>' if conf else "")
    return (
        f'<div class="eld-card {cls}"><div class="head">{head}</div>'
        f'<div class="claim" dir="auto">{html.escape(f.get("claim", ""))}</div>'
        f'<div class="cites">{_citation_chips(f.get("citations", []))}</div></div>'
    )


def render_assistant_message(msg: dict[str, Any]) -> None:
    if msg.get("content"):
        st.markdown(msg["content"])
    findings = msg.get("findings") or []
    if findings:
        st.markdown("".join(_finding_card(f) for f in findings), unsafe_allow_html=True)
    chip = _VERIFY_CHIP.get((msg.get("verification") or {}).get("status", ""), "")
    if chip:
        st.markdown(f'<div class="chips">{chip}</div>', unsafe_allow_html=True)
    render_reasoning(msg.get("events") or [])


# ----- contract annotations (requirement 4) -------------------------------- #
def _merge_findings(contract_id: str, findings: list[dict[str, Any]]) -> None:
    """Fold an answer's findings into the per-clause annotation map."""
    if not (contract_id and findings):
        return
    ann = st.session_state.annotations.setdefault(contract_id, {})
    try:
        conn = store.connect(retries=1, delay=0.0)
    except Exception:  # noqa: BLE001
        return
    try:
        for f in findings:
            law_cits = [c for c in f.get("citations", []) if c.get("source") == "law"]
            for c in f.get("citations", []):
                if c.get("source") != "contract":
                    continue
                ref = str(c.get("ref", "")).strip()
                rows = store.get_clause(conn, contract_id, ref)
                if not rows:
                    digits = "".join(ch for ch in ref if ch.isdigit())
                    if digits and digits != ref:
                        rows = store.get_clause(conn, contract_id, digits)
                for row in rows:
                    e = ann.setdefault(row["id"], {"laws": [], "risks": [], "notes": [], "quotes": []})
                    if c.get("quote") and c["quote"] not in e["quotes"]:
                        e["quotes"].append(c["quote"])
                    claim = f.get("claim", "")
                    if f.get("type") == "risk":
                        if claim and claim not in e["risks"]:
                            e["risks"].append(claim)
                    elif claim and claim not in [n for _, n in e["notes"]]:
                        e["notes"].append((f.get("type", "info"), claim))
                    for lc in law_cits:
                        pair = (lc.get("ref", ""), _short_law(lc.get("law")))
                        if pair not in e["laws"]:
                            e["laws"].append(pair)
    finally:
        conn.close()


def _clause_card(row: dict[str, Any], ann: dict[str, Any] | None) -> str:
    cls, badge = "", ""
    if ann and ann.get("risks"):
        cls, badge = "risk", '<span class="chip chip-red">⚠ risk flagged</span>'
    elif ann and ann.get("laws"):
        cls, badge = "law", '<span class="chip chip-gold">⚖ grounded in law</span>'

    text = html.escape(row.get("text") or "")
    if ann:
        for q in ann.get("quotes", []):
            eq = html.escape(q)
            if len(eq) >= 12 and eq in text:
                text = text.replace(eq, f"<mark>{eq}</mark>", 1)

    tags = [badge] if badge else []
    if ann:
        for ref, lawname in ann.get("laws", []):
            tags.append(f'<span class="chip chip-gold">⚖ {html.escape(str(ref))}{(" · " + html.escape(lawname)) if lawname else ""}</span>')
        for risk in ann.get("risks", []):
            tags.append(f'<span class="chip chip-red" title="{html.escape(risk)}">⚠ {html.escape(_clip(risk, 90))}</span>')
        for ntype, note in ann.get("notes", []):
            icon = _TYPE_META.get(ntype, _TYPE_META["info"])[0]
            tags.append(f'<span class="chip" title="{html.escape(note)}">{icon} {html.escape(_clip(note, 90))}</span>')

    head = f"Clause {row.get('clause_index')}" + (f" · {html.escape(row['clause_ref'])}" if row.get("clause_ref") else "")
    tags_html = f'<div class="cl-tags">{"".join(tags)}</div>' if tags else ""
    return (
        f'<div class="cl-card {cls}"><div class="cl-head">{head}</div>'
        f'<div class="cl-text" dir="auto">{text}</div>{tags_html}</div>'
    )


def run_full_review(active: dict[str, Any]) -> None:
    """Comprehensive clause-by-clause review; feeds the annotation map + chat."""
    lang = {"ar": "Arabic", "en": "English"}.get(str(active.get("lang") or ""), "the contract's language")
    question = (
        "Perform a FULL clause-by-clause legal review of the uploaded contract under Egyptian law. "
        "For EVERY clause with a legal implication, add one finding that cites the contract clause "
        "(source='contract', with a verbatim quote) and, where applicable, the governing Egyptian law "
        "article (source='law'). Use type='risk' for any clause that is risky, unfair, or likely "
        f"non-compliant with Egyptian law. Reply in {lang}."
    )
    log.info("full review requested for contract %s", active["contract_id"])
    thinking = ThinkingView("⚖️ Reviewing the contract against Egyptian law…")
    try:
        result = run_agent(question, contract_id=active["contract_id"], on_event=thinking)
    except Exception as exc:  # noqa: BLE001
        thinking.finish(ok=False)
        log.exception("full review failed")
        st.error(f"Review failed: {exc}")
        return
    status_ = result["verification"]["status"]
    thinking.finish(ok=status_ in ("passed", "partial"), note=f"{result['verification']['findings']} finding(s)")
    _merge_findings(active["contract_id"], result.get("findings", []))
    st.session_state.messages.append({"role": "user", "content": "🔍 Full legal review of the contract"})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.get("summary_text") or "",
            "findings": result.get("findings", []),
            "verification": result.get("verification", {}),
            "events": thinking.events,
        }
    )
    st.rerun()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
status = get_status()

with st.sidebar:
    st.markdown("## ⚖️ Agent El-Dostor")
    st.caption("Egyptian contract intelligence — every claim grounded in law.")

    mode = st.radio("Mode", ["📄 Review a contract", "🖊️ Generate a contract"], label_visibility="collapsed")

    if mode.startswith("🖊️"):
        st.divider()
        st.markdown("**Contract settings**")
        st.session_state.gen_ctype = st.selectbox(
            "Contract type", _CTYPES, index=_CTYPES.index(st.session_state.gen_ctype)
        )
        st.session_state.gen_lang = st.selectbox(
            "Language", list(_LANG_OPTS), format_func=_LANG_OPTS.get,
            index=list(_LANG_OPTS).index(st.session_state.gen_lang),
        )
        if st.button("✦ New contract (reset)", use_container_width=True):
            st.session_state.gen_messages = []
            st.session_state.gen_contract_obj = None
            st.session_state.generated = None
            st.rerun()

    st.divider()
    if status is None:
        st.markdown('<span class="chip chip-red">Database unreachable</span>', unsafe_allow_html=True)
    else:
        kb = status["kb"]
        kb_chip = (
            f'<span class="chip chip-gold">🏛 {kb:,} law articles indexed</span>'
            if kb
            else '<span class="chip chip-red">Knowledge base empty — see README step 3</span>'
        )
        st.markdown(
            f'<div class="chips">{kb_chip}'
            f'<span class="chip">📄 {len(status["contracts"])} contract(s) on file</span>'
            f'<span class="chip chip-dim">🔒 Runs privately in your Docker</span></div>',
            unsafe_allow_html=True,
        )
    st.caption(f"Reasoning model · `{settings.reasoning_model}`")


# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
st.markdown(
    """
<div class="eld-hero">
  <h1>⚖️ Agent El-Dostor · <span dir="rtl">الدستور</span></h1>
  <div class="tag">Reviews and drafts contracts under Egyptian law — bilingual, cited, and verified against the Civil Code.</div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================================== #
# MODE: REVIEW
# =========================================================================== #
def contract_section() -> dict[str, Any] | None:
    active = st.session_state.active_contract
    if active:
        st.markdown(
            f'<div class="chips"><span class="chip chip-gold">📄 {html.escape(active["filename"])}</span>'
            f'<span class="chip">{html.escape(str(active.get("contract_type")))}</span>'
            f'<span class="chip">{active.get("n_clauses")} clauses</span></div>',
            unsafe_allow_html=True,
        )
        if st.button("Change contract"):
            st.session_state.active_contract = None
            st.rerun()

    with st.expander("Upload or select a contract", expanded=active is None):
        uploaded = st.file_uploader(
            "Upload (PDF, DOCX, TXT, or image — scanned files are OCR'd)",
            type=["pdf", "docx", "txt", "md", "png", "jpg", "jpeg", "tiff", "bmp"],
        )
        contract_type = st.selectbox("Contract type", [*_CTYPES, "unknown"])
        if st.button("Analyze contract", type="primary", disabled=uploaded is None):
            with st.spinner("Parsing, segmenting, indexing…"):
                try:
                    meta = ingest_contract_bytes(uploaded.name, uploaded.getvalue(), contract_type)
                    st.session_state.active_contract = meta
                    st.session_state.annotations.pop(meta["contract_id"], None)
                    log.info("ingested contract %s (%s clauses)", meta["contract_id"], meta["n_clauses"])
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    log.exception("contract ingestion failed")
                    st.error(f"Could not analyze that file: {exc}")

        if status and status["contracts"]:
            options = {
                f"{c['filename']} ({c['contract_type']}, {c['n_clauses']} cl.)": c
                for c in status["contracts"]
            }
            pick = st.selectbox("…or reuse a contract on file", ["—", *options.keys()])
            if pick != "—" and st.button("Use selected contract"):
                c = options[pick]
                st.session_state.active_contract = {
                    "contract_id": c["id"],
                    "filename": c["filename"],
                    "contract_type": c["contract_type"],
                    "lang": c.get("lang"),
                    "n_clauses": c["n_clauses"],
                }
                st.rerun()
    return st.session_state.active_contract


def consult_tab(active: dict[str, Any] | None) -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_message(message)
            else:
                st.markdown(message["content"])

    prompt = st.chat_input("Ask about your contract or Egyptian law…", key="consult_input")
    if not prompt:
        return
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]
    ][-8:]
    contract_id = active["contract_id"] if active else None
    log.info("consult question (contract=%s): %s", contract_id, prompt[:120])

    with st.chat_message("assistant"):
        thinking = ThinkingView("⚖️ Consulting the contract and the law…")
        try:
            result = run_agent(prompt, contract_id=contract_id, history=history, on_event=thinking)
        except Exception as exc:  # noqa: BLE001
            thinking.finish(ok=False)
            log.exception("consult failed")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Something went wrong while consulting the law: {exc}"}
            )
            st.rerun()
        status_ = result["verification"]["status"]
        thinking.finish(
            ok=status_ in ("passed", "partial", "ungated"),
            note={"passed": "verified", "partial": "partially verified"}.get(status_, ""),
        )

    if contract_id:
        _merge_findings(contract_id, result.get("findings", []))
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.get("summary_text") or "_(no answer produced)_",
            "findings": result.get("findings", []),
            "verification": result.get("verification", {}),
            "events": thinking.events,
        }
    )
    st.rerun()


def contract_map_tab(active: dict[str, Any] | None) -> None:
    if not active:
        st.info("Upload or select a contract first — its annotated clause map will appear here.")
        return
    cid = active["contract_id"]
    try:
        conn = store.connect(retries=1, delay=0.0)
        try:
            clauses = store.list_clauses(conn, cid)
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        st.error("Could not load the contract from the database.")
        return
    if not clauses:
        st.info("No clauses found for this contract.")
        return

    ann = st.session_state.annotations.get(cid, {})
    n_risk = sum(1 for c in clauses if ann.get(c["id"], {}).get("risks"))
    n_law = sum(1 for c in clauses if ann.get(c["id"], {}).get("laws") and not ann.get(c["id"], {}).get("risks"))

    left, right = st.columns([3, 1])
    with left:
        st.markdown(
            f'<div class="chips"><span class="chip chip-red">⚠ {n_risk} risky clause(s)</span>'
            f'<span class="chip chip-gold">⚖ {n_law} grounded in law</span>'
            f'<span class="chip chip-dim">{len(clauses)} clauses total</span></div>',
            unsafe_allow_html=True,
        )
    with right:
        run_review = st.button("🔍 Run full legal review", type="primary", use_container_width=True)
    if run_review:
        run_full_review(active)

    if not ann:
        st.caption(
            "Clauses are highlighted as the agent reviews them — run the full legal review above, "
            "or ask questions in the Consult tab."
        )
    st.markdown("".join(_clause_card(c, ann.get(c["id"])) for c in clauses), unsafe_allow_html=True)


def review_mode() -> None:
    active = contract_section()
    tab_consult, tab_map = st.tabs(["💬 Consult", "📑 Annotated contract"])
    with tab_consult:
        consult_tab(active)
    with tab_map:
        contract_map_tab(active)


# =========================================================================== #
# MODE: GENERATE
# =========================================================================== #
def _clause_preview(c: dict[str, Any]) -> str:
    parts = ['<div class="cl-card law">' if c.get("legal_basis") else '<div class="cl-card">']
    head_en = f"Article ({c.get('number')})" + (f" · {html.escape(c['heading_en'])}" if c.get("heading_en") else "")
    parts.append(f'<div class="cl-head">{head_en}</div>')
    if c.get("body_en"):
        parts.append(f'<div class="cl-text">{html.escape(c["body_en"])}</div>')
    if c.get("heading_ar") or c.get("body_ar"):
        head_ar = f"المادة ({c.get('number')})" + (f" · {html.escape(c['heading_ar'])}" if c.get("heading_ar") else "")
        parts.append(f'<div class="cl-head" dir="rtl" style="margin-top:.45rem">{head_ar}</div>')
        if c.get("body_ar"):
            parts.append(f'<div class="cl-text" dir="rtl">{html.escape(c["body_ar"])}</div>')
    basis = c.get("legal_basis") or []
    if basis:
        chips = "".join(
            f'<span class="chip chip-gold" title="{html.escape(b.get("note", ""))}">⚖ {html.escape(str(b.get("ref", "")))}'
            f'{(" · " + html.escape(_short_law(b.get("law")))) if b.get("law") else ""}</span>'
            for b in basis
        )
        parts.append(f'<div class="cl-tags">{chips}</div>')
    parts.append("</div>")
    return "".join(parts)


def render_draft_preview(contract: dict[str, Any], verification: dict[str, Any]) -> None:
    if contract.get("title_en"):
        st.markdown(f"### {contract['title_en']}")
    if contract.get("title_ar"):
        st.markdown(f'<h4 dir="rtl">{html.escape(contract["title_ar"])}</h4>', unsafe_allow_html=True)
    meta = "  ·  ".join(x for x in [contract.get("place"), contract.get("date")] if x)
    if meta:
        st.caption(meta)
    chip = _GEN_CHIP.get(verification.get("status", ""), "")
    if chip:
        st.markdown(f'<div class="chips">{chip}</div>', unsafe_allow_html=True)

    if contract.get("parties"):
        rows = []
        for p in contract["parties"]:
            bits = [p.get("role_en") or p.get("role_ar"), p.get("name_en") or p.get("name_ar")]
            rows.append('<span class="chip">👤 ' + html.escape(" — ".join(b for b in bits if b)) + "</span>")
        st.markdown(f'<div class="chips">{"".join(rows)}</div>', unsafe_allow_html=True)

    with st.expander("📜 Draft clauses", expanded=True):
        st.markdown("".join(_clause_preview(c) for c in contract.get("clauses", [])), unsafe_allow_html=True)


def generate_mode() -> None:
    st.caption(
        "Describe the contract you want — the agent researches the Civil Code, drafts clause-by-clause "
        "with cited legal grounding, and you can refine it in conversation before exporting the PDF."
    )
    if status and status["kb"] == 0:
        st.warning("The legislation knowledge base is empty — clauses can't be legally grounded yet (README step 3).")

    for message in st.session_state.gen_messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_message(message)
            else:
                st.markdown(message["content"])

    prompt = st.chat_input(
        "Describe the contract, or ask for changes (e.g. “make the notice period 3 months”)…",
        key="gen_input",
    )
    if prompt:
        st.session_state.gen_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        ctype = st.session_state.gen_ctype
        language = st.session_state.gen_lang
        history = [
            {"role": m["role"], "content": m["content"]} for m in st.session_state.gen_messages[:-1]
        ][-6:]
        current = st.session_state.gen_contract_obj
        log.info("generate/refine request (type=%s lang=%s refine=%s): %s", ctype, language, current is not None, prompt[:120])

        with st.chat_message("assistant"):
            thinking = ThinkingView("🖊️ Researching the law and drafting…")
            try:
                if current is None:
                    result = generate_contract(ctype, language=language, brief=prompt, on_event=thinking)
                else:
                    result = refine_contract(
                        current_contract=current, user_request=prompt,
                        contract_type=ctype, language=language, history=history, on_event=thinking,
                    )
            except Exception as exc:  # noqa: BLE001
                thinking.finish(ok=False)
                log.exception("generation failed")
                st.session_state.gen_messages.append(
                    {"role": "assistant", "content": f"Something went wrong while drafting: {exc}"}
                )
                st.rerun()
            ok = result.get("status") == "generated" and result.get("contract")
            thinking.finish(ok=bool(ok), note=result.get("summary", ""))

        if result.get("contract"):
            st.session_state.gen_contract_obj = GeneratedContract.model_validate(result["contract"])
            st.session_state.generated = {"result": result, "pdfs": []}
            reply = (
                "The draft is updated below — review it and keep refining, or export the PDF."
                if current is not None
                else "Here is the first draft — review it below, then refine it in chat or export the PDF."
            )
        else:
            reply = f"I couldn't complete the draft. {result.get('error', '')}"

        st.session_state.gen_messages.append(
            {
                "role": "assistant",
                "content": reply,
                "verification": {},  # generation verdict is shown on the draft itself
                "events": thinking.events,
            }
        )
        st.rerun()

    gen = st.session_state.generated
    if not gen or not gen["result"].get("contract"):
        return

    result = gen["result"]
    contract = result["contract"]
    st.divider()
    render_draft_preview(contract, result.get("verification", {}))

    st.markdown("#### 📑 Export as PDF")
    language = contract.get("language", "bilingual")
    layout = "side_by_side"
    if language == "bilingual":
        layout = st.radio(
            "Bilingual layout", list(_LAYOUT_OPTS), format_func=_LAYOUT_OPTS.get, horizontal=True
        )
    if st.button("Build PDF", type="primary"):
        try:
            with st.spinner("Typesetting…"):
                st.session_state.generated["pdfs"] = render_contract_pdfs(contract, language, layout)
            log.info("PDF built (%s, %s)", language, layout)
        except Exception as exc:  # noqa: BLE001
            log.exception("PDF rendering failed")
            st.error(f"PDF rendering failed: {exc}")
    for name, data in st.session_state.generated.get("pdfs", []):
        st.download_button(
            f"⬇️ {name}", data=data, file_name=name, mime="application/pdf", key=f"dl_{name}"
        )


# --------------------------------------------------------------------------- #
if mode.startswith("📄"):
    review_mode()
else:
    generate_mode()

st.markdown(
    '<div class="eld-disclaimer">General legal information, not legal advice — a licensed Egyptian lawyer '
    'must review any output before use. · معلومات قانونية عامة وليست استشارة قانونية.</div>',
    unsafe_allow_html=True,
)
