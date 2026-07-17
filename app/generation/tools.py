"""Tools for the generation agent: the legislation search tools (reused from the
review agent) plus `submit_contract`, the only way to finish."""
from __future__ import annotations

from typing import Any

from app.agent.tools import LEGISLATION_TOOLS
from app.agent.tools import dispatch as legislation_dispatch  # noqa: F401  (re-exported for the loop)

_LEGAL_BASIS_SCHEMA = {
    "type": "array",
    "description": "Egyptian legislation this clause is grounded in. Empty for purely commercial/neutral clauses.",
    "items": {
        "type": "object",
        "properties": {
            "law": {"type": "string", "description": "Law name, copied EXACTLY from search results."},
            "ref": {"type": "string", "description": "article_ref, copied exactly (e.g. 'مادة 558')."},
            "note": {"type": "string", "description": "Short note on how the clause reflects this article."},
        },
        "required": ["ref"],
    },
}

_CLAUSE_SCHEMA = {
    "type": "object",
    "properties": {
        "number": {"type": "integer"},
        "heading_en": {"type": "string"},
        "heading_ar": {"type": "string"},
        "body_en": {"type": "string"},
        "body_ar": {"type": "string"},
        "legal_basis": _LEGAL_BASIS_SCHEMA,
    },
    "required": ["number"],
}

_PARTY_SCHEMA = {
    "type": "object",
    "properties": {
        "name_en": {"type": "string"},
        "name_ar": {"type": "string"},
        "role_en": {"type": "string", "description": "e.g. 'First Party (Employer)'."},
        "role_ar": {"type": "string", "description": "e.g. 'الطرف الأول (صاحب العمل)'."},
        "details_en": {"type": "string", "description": "capacity / national ID / address (use blanks if unknown)."},
        "details_ar": {"type": "string"},
    },
}

SUBMIT_CONTRACT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_contract",
        "description": (
            "Finish by submitting the complete drafted contract as structured data. "
            "Fill the fields for every requested language (both EN and AR when bilingual). "
            "Attach `legal_basis` (real article_refs from your searches) to every clause "
            "that reflects a legal rule — a verification step checks them against the "
            "knowledge base and drops any that do not resolve."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "contract_type": {"type": "string"},
                "language": {"type": "string", "enum": ["ar", "en", "bilingual"]},
                "title_en": {"type": "string"},
                "title_ar": {"type": "string"},
                "place": {"type": "string", "description": "Place of signing (e.g. 'Cairo' / 'القاهرة')."},
                "date": {"type": "string", "description": "Contract date, ISO (YYYY-MM-DD)."},
                "parties": {"type": "array", "items": _PARTY_SCHEMA},
                "preamble_en": {"type": "string", "description": "Recitals ('Whereas …'). Optional."},
                "preamble_ar": {"type": "string"},
                "clauses": {"type": "array", "items": _CLAUSE_SCHEMA},
                "governing_law_en": {"type": "string", "description": "Governing-law + (for bilingual) governing-language clause text."},
                "governing_law_ar": {"type": "string"},
            },
            "required": ["contract_type", "language", "clauses"],
        },
    },
}


def build_generation_tools() -> list[dict[str, Any]]:
    return LEGISLATION_TOOLS + [SUBMIT_CONTRACT_TOOL]
