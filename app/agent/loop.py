"""The agent loop with the M3 verification gate and M4 tracing.

The model decides which tools to call and must FINISH by calling `submit_answer`.
That submission is not trusted: a hard gate (app/agent/verify.py) checks every
finding's citation against real retrieved text. Failures are fed back and the
agent revises.

Observability lives in the LOGS, not the GUI: every tool call/result is logged
(summary at INFO, full payload at DEBUG), each run is appended to
logs/agent_runs.jsonl with tokens/cost/latency, and a one-line run summary is
logged at INFO. The optional `on_event` callback streams high-level progress
events (thinking / tool_call / tool_result / gate) so a UI can show the agent's
reasoning live, Claude-Code-style, without any raw debug payloads.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable

import psycopg
from pydantic import ValidationError

from app.agent.context import ToolContext
from app.agent.prompts import build_system_prompt
from app.agent.schema import Answer
from app.agent.tools import build_tools, dispatch, summarize_result
from app.agent.verify import format_answer, verify_answer
from app.core import trace as tracing
from app.core.config import settings
from app.core.logging import get_logger
from app.core.usage import Usage
from app.llm import client as llm
from app.retrieval import store

log = get_logger(__name__)

# on_event receives dicts like:
#   {"type": "thinking",    "text": str}
#   {"type": "tool_call",   "tool": str, "args": dict}
#   {"type": "tool_result", "tool": str, "summary": str}
#   {"type": "gate",        "status": "passed"|"rejected"|"invalid", "detail": str}
EventCallback = Callable[[dict[str, Any]], None]


def _handle_submit(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Validate + verify a submit_answer call. Returns an outcome dict."""
    try:
        answer = Answer.model_validate(args)
    except ValidationError as err:
        return {
            "status": "invalid",
            "feedback": json.dumps(
                {"status": "invalid", "error": "submit_answer did not match the schema", "details": err.errors()},
                ensure_ascii=False,
                default=str,
            ),
        }

    result = verify_answer(ctx, answer)
    passed, failed = result["passed"], result["failed"]
    answer_md = format_answer(answer.summary, passed, answer.disclaimer)
    citations = [
        {"source": c.source, "law": c.law, "ref": c.ref} for f in passed for c in f.citations
    ]
    findings_data = [f.model_dump() for f in passed]

    if not failed:
        return {
            "status": "passed",
            "answer_md": answer_md,
            "summary": answer.summary,
            "n_findings": len(passed),
            "n_failed": 0,
            "citations": citations,
            "findings": findings_data,
        }

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
    return {
        "status": "rejected",
        "feedback": feedback,
        "answer_md": answer_md,
        "summary": answer.summary,
        "n_findings": len(passed),
        "n_failed": len(failed),
        "citations": citations,
        "findings": findings_data,
    }


def run_agent(
    question: str,
    contract_id: str | None = None,
    history: list[dict[str, Any]] | None = None,
    conn: psycopg.Connection | None = None,
    on_event: EventCallback | None = None,
) -> dict[str, Any]:
    """Answer `question`.

    Returns {answer, summary_text, findings, trace, steps, usage, verification,
    citations, run_id}. `answer` is the full formatted markdown (API/eval);
    `summary_text` + structured `findings` let a UI render the answer natively.
    """
    t0 = time.monotonic()
    run_id = tracing.new_run_id()
    own_conn = conn is None
    if own_conn:
        conn = store.connect()

    trace: list[dict[str, Any]] = []
    usage = Usage(cost_known=True)

    def emit(event: dict[str, Any]) -> None:
        if on_event is None:
            return
        try:
            on_event(event)
        except Exception:  # noqa: BLE001 - a UI callback must never break the run
            log.debug("on_event callback failed", exc_info=True)

    def finalize(
        answer: str,
        steps: int,
        status: str,
        findings: int,
        citations: list | None = None,
        findings_data: list | None = None,
        summary_text: str = "",
    ) -> dict[str, Any]:
        result = {
            "answer": answer,
            "summary_text": summary_text or answer,
            "findings": findings_data or [],
            "trace": trace,
            "steps": steps,
            "usage": usage.as_dict(),
            "verification": {"status": status, "findings": findings},
            "citations": citations or [],
            "run_id": run_id,
        }
        tracing.record_run(
            {
                "run_id": run_id,
                "question": question[:200],
                "contract_id": contract_id,
                "verification_status": status,
                "n_findings": findings,
                "n_tool_calls": len(trace),
                "tools": [t["tool"] for t in trace],
                "steps": steps,
                "latency_s": round(time.monotonic() - t0, 3),
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "cost_usd": usage.cost_usd,
            }
        )
        # The run's debugging/usage summary lives here (and in the JSONL), not in the GUI.
        log.info(
            "run %s finished: status=%s findings=%s steps=%s tool_calls=%s tokens=%s cost=$%.4f latency=%.1fs",
            run_id, status, findings, steps, len(trace),
            usage.total_tokens, usage.cost_usd, time.monotonic() - t0,
        )
        return result

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

        last_partial: dict[str, Any] | None = None

        for step in range(1, settings.max_agent_iterations + 1):
            response = llm.chat(messages, tools=tools)
            usage.add(Usage.from_response(response))
            msg = response.choices[0].message

            if msg.content and msg.content.strip():
                emit({"type": "thinking", "text": msg.content.strip()})

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
                if last_partial:
                    return finalize(
                        last_partial["answer_md"], step, "partial", last_partial["n_findings"],
                        last_partial["citations"], last_partial["findings"], last_partial["summary"],
                    )
                return finalize(msg.content or "(no answer)", step, "ungated", 0)

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
                        emit({
                            "type": "gate", "status": "passed",
                            "detail": f"verification passed — {outcome['n_findings']} grounded finding(s)",
                        })
                        return finalize(
                            outcome["answer_md"], step, "passed", outcome["n_findings"],
                            outcome["citations"], outcome["findings"], outcome["summary"],
                        )
                    if outcome["status"] == "invalid":
                        emit({
                            "type": "gate", "status": "invalid",
                            "detail": "submission did not match the schema — retrying",
                        })
                    else:
                        emit({
                            "type": "gate", "status": "rejected",
                            "detail": (
                                f"{outcome.get('n_failed', 0)} claim(s) failed source verification — revising"
                            ),
                        })
                    if outcome.get("n_findings"):
                        last_partial = {
                            "answer_md": outcome["answer_md"],
                            "summary": outcome.get("summary", ""),
                            "n_findings": outcome["n_findings"],
                            "citations": outcome.get("citations", []),
                            "findings": outcome.get("findings", []),
                        }
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": outcome["feedback"]})
                else:
                    emit({"type": "tool_call", "tool": name, "args": args})
                    result = dispatch(ctx, name, args)
                    summary = summarize_result(result)
                    emit({"type": "tool_result", "tool": name, "summary": summary})
                    log.info("tool_call step=%s name=%s args=%s -> %s", step, name, args, summary)
                    log.debug("tool_result step=%s name=%s full=%s", step, name, result)
                    trace.append({"step": step, "tool": name, "args": args, "result": result})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        if last_partial:
            return finalize(
                last_partial["answer_md"], settings.max_agent_iterations, "partial",
                last_partial["n_findings"], last_partial["citations"],
                last_partial["findings"], last_partial["summary"],
            )
        return finalize(
            "(Reached the tool-call limit before producing a verified answer.)",
            settings.max_agent_iterations,
            "unverified",
            0,
        )
    finally:
        if own_conn:
            conn.close()
