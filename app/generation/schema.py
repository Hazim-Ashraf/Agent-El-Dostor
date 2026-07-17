"""Structured schema for a GENERATED contract (submit_contract).

The agent drafts the contract into this structure; the PDF renderer turns it into
a formatted document. Each clause may carry `legal_basis` references to Egyptian
legislation — those are checked against the real knowledge base (app/generation/verify.py)
so the contract "cites correctly" and is anchored to actual articles.

Every field has a default so a partial draft never hard-fails validation; the
renderer simply omits what the language in use doesn't need (e.g. no Arabic fields
for an English-only contract).
"""
from __future__ import annotations

from pydantic import BaseModel, Field

# Languages a single generation can target.
LANGUAGES = ("ar", "en", "bilingual")
# How a bilingual contract is laid out in the PDF (user-selectable at generation time).
BILINGUAL_LAYOUTS = ("side_by_side", "sequential", "separate")


class LegalBasis(BaseModel):
    """A reference to the Egyptian legislation a clause is grounded in."""

    law: str | None = None      # e.g. "Egyptian Civil Code (Law 131 of 1948)"
    ref: str = ""               # article_ref, e.g. "مادة 558" / "558"
    note: str = ""              # short EN/AR note on how the clause reflects the article


class Party(BaseModel):
    name_en: str = ""
    name_ar: str = ""
    role_en: str = ""           # e.g. "First Party (Employer)"
    role_ar: str = ""           # e.g. "الطرف الأول (صاحب العمل)"
    details_en: str = ""        # national ID / address / capacity
    details_ar: str = ""


class Clause(BaseModel):
    number: int = 0
    heading_en: str = ""
    heading_ar: str = ""
    body_en: str = ""
    body_ar: str = ""
    legal_basis: list[LegalBasis] = Field(default_factory=list)


class GeneratedContract(BaseModel):
    contract_type: str = "unknown"
    language: str = "bilingual"          # ar | en | bilingual
    title_en: str = ""
    title_ar: str = ""
    place: str = ""                      # place of signing, e.g. "Cairo" / "القاهرة"
    date: str = ""                       # e.g. "2026-07-18"
    parties: list[Party] = Field(default_factory=list)
    preamble_en: str = ""                # recitals ("Whereas …")
    preamble_ar: str = ""
    clauses: list[Clause] = Field(default_factory=list)
    governing_law_en: str = ""           # governing-law + language clause text
    governing_law_ar: str = ""

    def wants(self, lang: str) -> bool:
        """Whether a given language's fields are needed for this contract."""
        if self.language == "bilingual":
            return True
        return self.language == lang
