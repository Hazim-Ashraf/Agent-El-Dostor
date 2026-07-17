"""The agent loop: a goal-driven, tool-calling loop over OpenRouter.

This is the genuinely agentic core — the model decides which tools to call, in
what order, and when it's done. (Not a fixed workflow.)
"""
from __future__ import annotations

import json
from typing import Any

import psycopg

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import TOOLS, dispatch
from app.core.config import settings
from app.core.logging import get_logger
from app.llm import client as llm
from app.retrieval import store

log = get_logger(__name__)


def run_agent(question: str, conn: psycopg.Connection | None = None) -> dict[str, Any]:
    """Answer `question`. Returns {answer, trace, steps}."""
    own_conn = conn is None
    if own_conn:
        conn = store.connect()
    try:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        trace: list[dict[str, Any]] = []

        for step in range(1, settings.max_agent_iterations + 1):
            response = llm.chat(messages, tools=TOOLS)
            msg = response.choices[0].message

            assistant: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                assistant["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant)

            if not msg.tool_calls:
                return {"answer": msg.content or "", "trace": trace, "steps": step}

            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                log.info("tool_call step=%s name=%s args=%s", step, tc.function.name, args)
                result = dispatch(conn, tc.function.name, args)
                trace.append(
                    {"step": step, "tool": tc.function.name, "args": args, "result": result}
                )
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )

        return {
            "answer": "(Reached the tool-call limit without a final answer.)",
            "trace": trace,
            "steps": settings.max_agent_iterations,
        }
    finally:
        if own_conn:
            conn.close()
