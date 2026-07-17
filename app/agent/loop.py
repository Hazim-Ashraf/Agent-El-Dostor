"""The agent loop: a goal-driven, tool-calling loop over OpenRouter.

The model decides which tools to call, in what order, and when it's done — over
both the uploaded contract and the Egyptian legislation KB. Token/USD usage is
accumulated across all LLM calls in the run.
"""
from __future__ import annotations

import json
from typing import Any

import psycopg

from app.agent.context import ToolContext
from app.agent.prompts import build_system_prompt
from app.agent.tools import build_tools, dispatch
from app.core.config import settings
from app.core.logging import get_logger
from app.core.usage import Usage
from app.llm import client as llm
from app.retrieval import store

log = get_logger(__name__)


def run_agent(
    question: str,
    contract_id: str | None = None,
    history: list[dict[str, Any]] | None = None,
    conn: psycopg.Connection | None = None,
) -> dict[str, Any]:
    """Answer `question`. Returns {answer, trace, steps, usage}."""
    own_conn = conn is None
    if own_conn:
        conn = store.connect()
    try:
        contract_meta = store.get_contract(conn, contract_id) if contract_id else None
        ctx = ToolContext(conn=conn, contract_id=contract_id)
        tools = build_tools(has_contract=contract_meta is not None)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt(contract_meta)}
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})

        trace: list[dict[str, Any]] = []
        usage = Usage(cost_known=True)

        for step in range(1, settings.max_agent_iterations + 1):
            response = llm.chat(messages, tools=tools)
            usage.add(Usage.from_response(response))
            msg = response.choices[0].message

            assistant: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                assistant["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant)

            if not msg.tool_calls:
                return {"answer": msg.content or "", "trace": trace, "steps": step, "usage": usage.as_dict()}

            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                log.info("tool_call step=%s name=%s args=%s", step, tc.function.name, args)
                result = dispatch(ctx, tc.function.name, args)
                trace.append({"step": step, "tool": tc.function.name, "args": args, "result": result})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        return {
            "answer": "(Reached the tool-call limit without a final answer.)",
            "trace": trace,
            "steps": settings.max_agent_iterations,
            "usage": usage.as_dict(),
        }
    finally:
        if own_conn:
            conn.close()
