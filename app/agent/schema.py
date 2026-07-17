"""Structured output schema for the agent's final answer (submit_answer)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    # "law" or "contract". Kept permissive (str) so a schema nit doesn't fail the
    # whole submission — the real check is the verification gate.
    source: str = "law"
    law: str | None = None            # law name for source="law"
    ref: str = ""                     # article_ref (law) or clause ref/number (contract)
    quote: str = ""                   # VERBATIM excerpt from the cited source text


class Finding(BaseModel):
    claim: str = ""
    type: str = "info"                # right | obligation | risk | action | info
    citations: list[Citation] = Field(default_factory=list)
    confidence: str = "medium"        # low | medium | high


class Answer(BaseModel):
    summary: str = ""
    findings: list[Finding] = Field(default_factory=list)
    disclaimer: str = ""
