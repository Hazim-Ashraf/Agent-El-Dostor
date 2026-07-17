"""Tool definitions (OpenAI/OpenRouter function-calling) and dispatch.

Legislation tools are always available. Contract tools are added only when a
contract is loaded; both are scoped to the active contract via ToolContext.
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any

from app.agent.context import ToolContext
from app.retrieval import embeddings, store

LEGISLATION_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_legislation",
            "description": (
                "Semantic search over the Egyptian legislation knowledge base. Returns "
                "in-force articles most relevant to the query (law, article_ref, language, "
                "text). Use this to find the legal basis for any claim."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for, Arabic or English."},
                    "law": {"type": "string", "description": "Optional: restrict to a specific law."},
                    "as_of_date": {"type": "string", "description": "Optional ISO date (YYYY-MM-DD)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_legal_article",
            "description": "Fetch the exact text of a specific article by law and article reference.",
            "parameters": {
                "type": "object",
                "properties": {
                    "law": {"type": "string"},
                    "article_ref": {"type": "string"},
                },
                "required": ["law", "article_ref"],
            },
        },
    },
]

CONTRACT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_contract",
            "description": (
                "Semantic search over the user's UPLOADED contract. Returns the clauses "
                "most relevant to the query (clause_index, clause_ref, text). Use this to "
                "find what the contract actually says before comparing it to the law."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for in the contract."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_contract_clause",
            "description": "Fetch a specific clause of the uploaded contract by its reference or number (e.g. 'Clause 5' or '5').",
            "parameters": {
                "type": "object",
                "properties": {
                    "clause_ref": {"type": "string"},
                },
                "required": ["clause_ref"],
            },
        },
    },
]


def build_tools(has_contract: bool) -> list[dict[str, Any]]:
    return LEGISLATION_TOOLS + (CONTRACT_TOOLS if has_contract else [])


def _fmt_article(row: dict[str, Any]) -> dict[str, Any]:
    out = {
        "id": row.get("id"),
        "law": row.get("law"),
        "article_ref": row.get("article_ref"),
        "lang": row.get("lang"),
        "title": row.get("title"),
        "text": row.get("text"),
    }
    eff = row.get("effective_date")
    if isinstance(eff, date):
        out["effective_date"] = eff.isoformat()
    return out


def _fmt_clause(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "clause_index": row.get("clause_index"),
        "clause_ref": row.get("clause_ref"),
        "lang": row.get("lang"),
        "text": row.get("text"),
    }


def dispatch(ctx: ToolContext, name: str, args: dict[str, Any]) -> str:
    """Execute a tool call; always returns a JSON string for the model."""
    conn = ctx.conn

    if name == "search_legislation":
        query = (args.get("query") or "").strip()
        if not query:
            return json.dumps({"error": "empty query"})
        emb = embeddings.embed_query(query)
        rows = store.search(conn, emb, law=args.get("law"), as_of_date=args.get("as_of_date"), limit=6)
        return json.dumps([_fmt_article(r) for r in rows], ensure_ascii=False)

    if name == "get_legal_article":
        rows = store.get_article(conn, args.get("law", ""), args.get("article_ref", ""))
        if not rows:
            return json.dumps({"error": "article not found in knowledge base"})
        return json.dumps([_fmt_article(r) for r in rows], ensure_ascii=False)

    if name == "search_contract":
        if not ctx.contract_id:
            return json.dumps({"error": "no contract is currently loaded"})
        query = (args.get("query") or "").strip()
        if not query:
            return json.dumps({"error": "empty query"})
        emb = embeddings.embed_query(query)
        rows = store.search_contract(conn, ctx.contract_id, emb, limit=6)
        return json.dumps([_fmt_clause(r) for r in rows], ensure_ascii=False)

    if name == "get_contract_clause":
        if not ctx.contract_id:
            return json.dumps({"error": "no contract is currently loaded"})
        rows = store.get_clause(conn, ctx.contract_id, args.get("clause_ref", ""))
        if not rows:
            return json.dumps({"error": "clause not found in the uploaded contract"})
        return json.dumps([_fmt_clause(r) for r in rows], ensure_ascii=False)

    return json.dumps({"error": f"unknown tool '{name}'"})
