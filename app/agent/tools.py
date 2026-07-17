"""Tool definitions (OpenAI/OpenRouter function-calling schema) and dispatch.

M0 exposes two legislation tools. Contract tools, verify_support, and
submit_answer arrive in later milestones.
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any

import psycopg

from app.retrieval import embeddings, store

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_legislation",
            "description": (
                "Semantic search over the Egyptian legislation knowledge base. Returns "
                "in-force articles most relevant to the query, each with its law, "
                "article_ref, language and text. Use this to find the legal basis for a claim."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for, in Arabic or English.",
                    },
                    "law": {
                        "type": "string",
                        "description": "Optional: restrict to a specific law name/id.",
                    },
                    "as_of_date": {
                        "type": "string",
                        "description": "Optional ISO date (YYYY-MM-DD): only articles in force on that date.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_legal_article",
            "description": "Fetch the exact text of a specific article by law and article reference (all available languages).",
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


def _fmt(row: dict[str, Any]) -> dict[str, Any]:
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


def dispatch(conn: psycopg.Connection, name: str, args: dict[str, Any]) -> str:
    """Execute a tool call; always returns a JSON string for the model."""
    if name == "search_legislation":
        query = (args.get("query") or "").strip()
        if not query:
            return json.dumps({"error": "empty query"})
        emb = embeddings.embed_query(query)
        rows = store.search(
            conn,
            emb,
            law=args.get("law"),
            as_of_date=args.get("as_of_date"),
            limit=6,
        )
        return json.dumps([_fmt(r) for r in rows], ensure_ascii=False)

    if name == "get_legal_article":
        rows = store.get_article(conn, args.get("law", ""), args.get("article_ref", ""))
        if not rows:
            return json.dumps({"error": "article not found in knowledge base"})
        return json.dumps([_fmt(r) for r in rows], ensure_ascii=False)

    return json.dumps({"error": f"unknown tool '{name}'"})
