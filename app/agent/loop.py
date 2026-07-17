"""The agent loop with the M3 verification gate.

The model decides which tools to call and must FINISH by calling `submit_answer`.
That submission is not trusted: a hard gate (app/agent/verify.py) checks every
finding's citation against real retrieved text. Failures are fed back and the
agent revises. Token/USD usage is accumulated across all LLM calls.
"""
from __future__ import annotations

import json
from typing import Any

import psycopg
from pydantic import ValidationError

from app.agent.context import ToolContext
from app.agent.prompts import build_system_prompt
from app.agent.schema import Answer
from app.agent.tools import build_tools, dispatch
from app.agent.verify import format_answer, verify_answer
from app.core.config import settings
from app.core.logging import get_logger
from app.core.usage import Usage
from app.llm import client as llm
from app.retrieval import store

log = get_logger(__name__)


def _handle_submit(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Validate + verify a submit_answer call. Returns an outcome dict."""
    try:
        answer = Answer.model_validate(args)
    except ValidationError as err:
        return {
            "status": "invalid",
            "feedback": json.dumps(
                {
                    "status": "invalid",
                    "error": "submit_answer did not match the schema",
                    "details": err.errors(),
                },
                ensure_ascii=False,
                default=str,
            ),
        }

    result = verify_answer(ctx, answer)
    passed, failed = result["passed"], result["failed"]
    answer_md = format_answer(answer.summary, passed, answer.disclaimer)

    if not failed:
        return {"status": "passed", "answer_md": answer_md, "n_findings": len(passed)}

    feedback = json.dumps(
        {
            "status": "rejected",
            "accepted_findings": len(passed),
            "failed_findings": failed,
            "instruction": (
                "For each failed finding, cite a real article/clause and copy an EXACT "
                "quote from its retrieved text, or remove the claim. Then call submit_answer again."
            ),
        },
        ensure_ascii=False,
    )
    return {"status": "rejected", "feedback": feedback, "answer_md": answer_md, "n_findings": len(passed)}


def run_agent(
    question: str,
    contract_id: str | None = None,
    history: list[dict[str, Any]] | None = None,
    conn: psycopg.Connection | None = None,
) -> dict[str, Any]:
    """Answer `question`. Returns {answer, trace, steps, usage, verification}."""
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
        last_partial: dict[str, Any] | None = None

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
                # The model answered without submit_answer (ungated fallback).
                if last_partial:
                    return _result(last_partial["answer_md"], trace, step, usage, "partial", last_partial["n_findings"])
                return _result(msg.content or "(no answer)", trace, step, usage, "ungated", 0)

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                if name == "submit_answer":
                    outcome = _handle_submit(ctx, args)
                    trace.append(
                        {
                            "step": step,
                            "tool": "submit_answer",
                            "args": args,
                            "result": json.dumps(
                                {"status": outcome["status"], "accepted_findings": outcome.get("n_findings", 0)},
                                ensure_ascii=False,
                            ),
                        }
                    )
                    if outcome["status"] == "passed":
                        return _result(outcome["answer_md"], trace, step, usage, "passed", outcome["n_findings"])
                    if outcome.get("answer_md") and outcome.get("n_findings"):
                        last_partial = {"answer_md": outcome["answer_md"], "n_findings": outcome["n_findings"]}
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": outcome["feedback"]})
                else:
                    log.info("tool_call step=%s name=%s args=%s", step, name, args)
                    result = dispatch(ctx, name, args)
                    trace.append({"step": step, "tool": name, "args": args, "result": result})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        if last_partial:
            return _result(
                last_partial["answer_md"], trace, settings.max_agent_iterations, usage, "partial", last_partial["n_findings"]
            )
        return _result(
            "(Reached the tool-call limit before producing a verified answer.)",
            trace,
            settings.max_agent_iterations,
            usage,
            "unverified",
            0,
        )
    finally:
        if own_conn:
            conn.close()


def _result(
    answer: str, trace: list, steps: int, usage: Usage, status: str, findings: int
) -> dict[str, Any]:
    return {
        "answer": answer,
        "trace": trace,
        "steps": steps,
        "usage": usage.as_dict(),
        "verification": {"status": status, "findings": findings},
    }
