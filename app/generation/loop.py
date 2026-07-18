"""The contract-generation agent loop (M5).

The model autonomously researches the relevant Egyptian legislation with
`search_legislation` / `get_legal_article`, then FINISHES by calling
`submit_contract` with the full structured draft. That draft's legal citations are
verified against the real knowledge base before it is returned.

Observability lives in the LOGS, not the GUI: tool calls/results are logged
(summary at INFO, full payload at DEBUG), each run is traced to
logs/agent_runs.jsonl with tokens/cost/latency, and a one-line run summary is
logged. The optional `on_event` callback streams high-level progress events
(thinking / tool_call / tool_result / gate) so a UI can show the drafting
process live, Claude-Code-style.
"""
from __future__ import annotations

import json
import time
from datetime import date
from typing import Any, Callable

import psycopg
from pydantic import ValidationError

from app.agent.context import ToolContext
from app.agent.tools import summarize_result
from app.generation.prompts import build_generation_prompt, build_refinement_prompt, draft_summary
from app.generation.schema import GeneratedContract
from app.generation.tools import build_generation_tools, legislation_dispatch
from app.generation.verify import verify_contract
from app.core import trace as tracing
from app.core.config import settings
from app.core.logging import get_logger
from app.core.usage import Usage
from app.llm import client as llm
from app.retrieval import store

log = get_logger(__name__)

EventCallback = Callable[[dict[str, Any]], None]

_NUDGE = (
    "You have not produced the contract yet. Research the relevant Egyptian legislation "
    "if needed, then finish by calling submit_contract with the full structured draft."
)

_REFINE_NUDGE = (
    "You have not submitted the updated contract yet. Apply the user's requested changes "
    "to the existing contract and call submit_contract with the FULL updated contract."
)

_EMPTY_VERIFICATION: dict[str, Any] = {
    "status": "none", "n_clauses": 0, "n_legal_basis": 0, "n_verified": 0, "n_dropped": 0, "dropped": [],
}


def _safe_emit(on_event: EventCallback | None, event: dict[str, Any]) -> None:
    if on_event is None:
        return
    try:
        on_event(event)
    except Exception:  # noqa: BLE001 - a UI callback must never break the run
        log.debug("on_event callback failed", exc_info=True)


def _run_drafting_loop(
    mode: str,
    system: str,
    first_user_msg: str,
    nudge: str,
    contract_type: str,
    language: str,
    conn: psycopg.Connection,
    on_event: EventCallback | None,
    history: list[dict[str, Any]] | None = None,
    fallback_contract: GeneratedContract | None = None,
) -> dict[str, Any]:
    """Shared agent loop for generate ('generate') and refine ('refine')."""
    t0 = time.monotonic()
    run_id = tracing.new_run_id()
    trace: list[dict[str, Any]] = []
    usage = Usage(cost_known=True)

    def finalize(
        status: str, contract: GeneratedContract | None, verification: dict[str, Any], steps: int, error: str = ""
    ) -> dict[str, Any]:
        result = {
            "status": status,
            "contract": contract.model_dump() if contract else None,
            "verification": verification,
            "trace": trace,
            "usage": usage.as_dict(),
            "steps": steps,
            "run_id": run_id,
            "summary": draft_summary(contract) if contract else "",
            "error": error,
        }
        tracing.record_run(
            {
                "run_id": run_id,
                "mode": mode,
                "contract_type": contract_type,
                "language": language,
                "status": status,
                "verification_status": verification.get("status"),
                "n_clauses": verification.get("n_clauses", 0),
                "n_legal_basis": verification.get("n_legal_basis", 0),
                "n_verified": verification.get("n_verified", 0),
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
        # Debugging/usage summary goes to the logs (and JSONL), not the GUI.
        log.info(
            "%s run %s: status=%s verification=%s clauses=%s refs=%s/%s steps=%s tokens=%s cost=$%.4f latency=%.1fs",
            mode, run_id, status, verification.get("status"),
            verification.get("n_clauses", 0), verification.get("n_verified", 0),
            verification.get("n_legal_basis", 0), steps,
            usage.total_tokens, usage.cost_usd, time.monotonic() - t0,
        )
        return result

    ctx = ToolContext(conn=conn, contract_id=None)
    tools = build_generation_tools()
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": first_user_msg})

    for step in range(1, settings.max_generation_iterations + 1):
        response = llm.chat(messages, tools=tools, max_tokens=settings.generation_max_tokens)
        usage.add(Usage.from_response(response))
        msg = response.choices[0].message

        if msg.content and msg.content.strip():
            _safe_emit(on_event, {"type": "thinking", "text": msg.content.strip()})

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
            messages.append({"role": "user", "content": nudge})
            continue

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "submit_contract":
                try:
                    contract = GeneratedContract.model_validate(args)
                except ValidationError as err:
                    trace.append({"step": step, "tool": "submit_contract", "args": {}, "result": "invalid schema"})
                    _safe_emit(on_event, {
                        "type": "gate", "status": "invalid",
                        "detail": "draft did not match the contract schema — retrying",
                    })
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(
                                {"status": "invalid", "details": err.errors()},
                                ensure_ascii=False,
                                default=str,
                            ),
                        }
                    )
                    continue
                # Force the requested language onto the structure so the renderer is correct.
                contract.language = language
                if not contract.contract_type or contract.contract_type == "unknown":
                    contract.contract_type = contract_type
                verification = verify_contract(conn, contract)
                trace.append(
                    {
                        "step": step,
                        "tool": "submit_contract",
                        "args": {"clauses": len(contract.clauses)},
                        "result": json.dumps(verification, ensure_ascii=False),
                    }
                )
                _safe_emit(on_event, {
                    "type": "gate", "status": "passed",
                    "detail": (
                        f"draft assembled — {verification.get('n_clauses', 0)} clause(s), "
                        f"{verification.get('n_verified', 0)}/{verification.get('n_legal_basis', 0)} "
                        "legal reference(s) verified against the knowledge base"
                    ),
                })
                return finalize("generated", contract, verification, step)

            _safe_emit(on_event, {"type": "tool_call", "tool": name, "args": args})
            result = legislation_dispatch(ctx, name, args)
            summary = summarize_result(result)
            _safe_emit(on_event, {"type": "tool_result", "tool": name, "summary": summary})
            log.info("%s tool_call step=%s name=%s args=%s -> %s", mode, step, name, args, summary)
            log.debug("%s tool_result step=%s name=%s full=%s", mode, step, name, result)
            trace.append({"step": step, "tool": name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # Iteration limit reached.
    if fallback_contract is not None:
        verification = verify_contract(conn, fallback_contract)
        return finalize(
            "failed", fallback_contract, verification, settings.max_generation_iterations,
            error="Reached the iteration limit before the refined contract was submitted.",
        )
    return finalize(
        "failed", None, dict(_EMPTY_VERIFICATION), settings.max_generation_iterations,
        error="Reached the iteration limit before the contract was submitted.",
    )


def generate_contract(
    contract_type: str,
    language: str = "bilingual",
    brief: str = "",
    conn: psycopg.Connection | None = None,
    on_event: EventCallback | None = None,
) -> dict[str, Any]:
    """Draft a contract. Returns {status, contract, verification, trace, usage, steps, run_id, summary}."""
    own_conn = conn is None
    if own_conn:
        conn = store.connect()
    try:
        system = build_generation_prompt(contract_type, language, brief, date.today().isoformat())
        return _run_drafting_loop(
            "generate", system, "Draft the contract now, following the rules above.",
            _NUDGE, contract_type, language, conn, on_event,
        )
    finally:
        if own_conn:
            conn.close()


def refine_contract(
    current_contract: GeneratedContract,
    user_request: str,
    contract_type: str,
    language: str = "bilingual",
    history: list[dict[str, Any]] | None = None,
    conn: psycopg.Connection | None = None,
    on_event: EventCallback | None = None,
) -> dict[str, Any]:
    """Refine an existing contract based on the user's modification request.

    Same return shape as generate_contract(). If the loop runs out of iterations,
    the ORIGINAL contract is returned unchanged (status='failed').
    """
    own_conn = conn is None
    if own_conn:
        conn = store.connect()
    try:
        system = build_refinement_prompt(current_contract, contract_type, language, date.today().isoformat())
        return _run_drafting_loop(
            "refine", system, user_request,
            _REFINE_NUDGE, contract_type, language, conn, on_event,
            history=history, fallback_contract=current_contract,
        )
    finally:
        if own_conn:
            conn.close()
