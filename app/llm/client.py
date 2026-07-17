"""Thin wrapper around the OpenAI SDK pointed at OpenRouter.

OpenRouter is OpenAI-compatible, so we use the official `openai` client with a
custom `base_url`. This is the reasoning LLM for the agent; embeddings are a
separate, local, free model (see app/retrieval/embeddings.py).
"""
from __future__ import annotations

from typing import Any

from openai import OpenAI

from app.core.config import settings

_client: OpenAI | None = None


def _client_instance() -> OpenAI:
    global _client
    if _client is None:
        if not settings.openrouter_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = OpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
            # Optional OpenRouter attribution headers.
            default_headers={
                "HTTP-Referer": "https://github.com/agent-eldostor",
                "X-Title": "Agent El-Dostor",
            },
        )
    return _client


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    model: str | None = None,
    temperature: float = 0.0,
):
    """One chat-completions call. Returns the raw OpenAI response object."""
    kwargs: dict[str, Any] = {
        "model": model or settings.reasoning_model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return _client_instance().chat.completions.create(**kwargs)
