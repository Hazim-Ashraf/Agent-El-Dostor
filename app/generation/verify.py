"""Verify a generated contract's legal citations against the real knowledge base.

Grounding for generation is the mirror of the review gate: we cannot deterministically
prove a clause "adheres to" the law, but we CAN prove that every article the draft
cites actually exists in the corpus. Any `legal_basis` reference that does not resolve
to a stored article is dropped and reported, so the contract only ever cites real law.
"""
from __future__ import annotations

from typing import Any

import psycopg

from app.generation.schema import GeneratedContract
from app.retrieval import store


def verify_contract(conn: psycopg.Connection, contract: GeneratedContract) -> dict[str, Any]:
    """Resolve every clause's legal_basis; drop unresolved refs (mutates `contract`)."""
    total = 0
    verified = 0
    dropped: list[dict[str, Any]] = []

    for clause in contract.clauses:
        kept = []
        for lb in clause.legal_basis:
            total += 1
            rows: list[dict[str, Any]] = []
            if lb.law:
                rows = store.get_article(conn, lb.law, lb.ref)
            if not rows:
                rows = store.find_articles_by_ref(conn, lb.ref)
            if rows:
                # Normalise to the stored law name / ref so the PDF cites it consistently.
                lb.law = rows[0].get("law") or lb.law
                lb.ref = rows[0].get("article_ref") or lb.ref
                kept.append(lb)
                verified += 1
            else:
                dropped.append({"clause": clause.number, "ref": lb.ref, "law": lb.law})
        clause.legal_basis = kept

    if total == 0:
        status = "none"
    elif verified == 0:
        status = "unverified"
    elif dropped:
        status = "partial"
    else:
        status = "grounded"

    return {
        "status": status,
        "n_clauses": len(contract.clauses),
        "n_legal_basis": total,
        "n_verified": verified,
        "n_dropped": len(dropped),
        "dropped": dropped,
    }
