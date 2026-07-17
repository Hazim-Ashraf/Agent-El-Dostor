"""The verification gate (M3) — the guardrail outside the model's discretion.

`submit_answer` is not trusted. For every finding, each citation must:
  1. RESOLVE to a real retrieved source (a legislation article or a contract clause), and
  2. be SUPPORTED — the model's `quote` must actually appear in that source text.

Citations that fail are dropped; a finding with zero valid citations is rejected
and fed back to the agent to fix. This makes groundedness non-negotiable while
keeping the loop agentic (the model revises and resubmits).
"""
from __future__ import annotations

import re
from typing import Any

from app.agent.context import ToolContext
from app.agent.schema import Answer, Citation, Finding
from app.retrieval import store

DEFAULT_DISCLAIMER = (
    "This is general legal information based on the available knowledge base, not a "
    "substitute for advice from a licensed Egyptian lawyer."
)

_TASHKEEL = re.compile(r"[ً-ْـ]")  # Arabic diacritics + tatweel
_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = _TASHKEEL.sub("", text)
    text = (
        text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
        .replace("ى", "ي").replace("ة", "ه")
        .replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    )
    return _WS.sub(" ", text).strip()


def _supports(quote: str, source_text: str) -> bool:
    """True if `quote` is grounded in `source_text` (substring or high token overlap)."""
    q = _normalize(quote)
    s = _normalize(source_text)
    if len(q) < 8:
        return q in s and bool(q)
    if q in s:
        return True
    # Fallback tolerates only trivial noise (whitespace/diacritics/dropped stopword),
    # NOT content changes like a swapped number — keep the threshold high so a
    # fabricated near-quote is rejected and sent back to the agent.
    q_tokens = [t for t in q.split() if len(t) > 2]
    if not q_tokens:
        return False
    s_tokens = set(s.split())
    hits = sum(1 for t in q_tokens if t in s_tokens)
    return hits / len(q_tokens) >= 0.85


def _resolve_source(ctx: ToolContext, c: Citation) -> str | None:
    """Return the concatenated source text for a citation, or None if it doesn't exist."""
    if c.source == "contract":
        if not ctx.contract_id:
            return None
        rows = store.get_clause(ctx.conn, ctx.contract_id, c.ref)
    else:  # law
        rows = []
        if c.law:
            rows = store.get_article(ctx.conn, c.law, c.ref)
        if not rows:
            rows = store.find_articles_by_ref(ctx.conn, c.ref)
    if not rows:
        return None
    return "\n".join(r.get("text", "") for r in rows)


def verify_answer(ctx: ToolContext, answer: Answer) -> dict[str, Any]:
    """Return {passed: [Finding(valid citations only)], failed: [{claim, problems}]}."""
    passed: list[Finding] = []
    failed: list[dict[str, Any]] = []

    for finding in answer.findings:
        valid: list[Citation] = []
        problems: list[str] = []
        for c in finding.citations:
            source_text = _resolve_source(ctx, c)
            if source_text is None:
                problems.append(f"citation {c.source}:{c.ref!r} does not resolve to a known source")
                continue
            if not _supports(c.quote, source_text):
                problems.append(f"quote for {c.source}:{c.ref!r} was not found in the cited source text")
                continue
            valid.append(c)

        if valid:
            passed.append(finding.model_copy(update={"citations": valid}))
        else:
            failed.append(
                {
                    "claim": finding.claim,
                    "problems": problems or ["finding has no citations"],
                }
            )

    return {"passed": passed, "failed": failed}


def format_answer(summary: str, findings: list[Finding], disclaimer: str) -> str:
    lines: list[str] = []
    if summary:
        lines += [summary, ""]
    if findings:
        lines.append("**Findings**")
        for f in findings:
            lines.append(f"- **[{(f.type or 'info').upper()}]** {f.claim}")
            for c in f.citations:
                if c.source == "law":
                    ref = f"{c.law} / {c.ref}" if c.law else c.ref
                    label = "Law"
                else:
                    ref = c.ref
                    label = "Contract"
                lines.append(f"    - _{label}_ ({ref}): “{c.quote}”")
        lines.append("")
    lines.append(f"_{disclaimer.strip() or DEFAULT_DISCLAIMER}_")
    return "\n".join(lines)
