"""The contract-generation agent loop (M5).

The model autonomously researches the relevant Egyptian legislation with
`search_legislation` / `get_legal_article`, then FINISHES by calling
`submit_contract` with the full structured draft. That draft's legal citations are
verified against the real knowledge base before it is returned. Token/USD usage is
tracked and the run is traced, exactly like the review agent.
"""
from __future__ import annotations

import json
import time
from datetime import date
from typing import Any

import psycopg
from pydantic import ValidationError

from app.agent.context import ToolContext
from app.generation.prompts import build_generation_prompt, draft_summary
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

_NUDGE = (
    "You have not produced the contract yet. Research the relevant Egyptian legislation "
    "if needed, then finish by calling submit_contract with the full structured draft."
)


def generate_contract(
    contract_type: str,
    language: str = "bilingual",
    brief: str = "",
    conn: psycopg.Connection | None = None,
) -> dict[str, Any]:
    """Draft a contract. Returns {status, contract, verification, trace, usage, steps, run_id, summary}."""
    t0 = time.monotonic()
    run_id = tracing.new_run_id()
    own_conn = conn is None
    if own_conn:
        conn = store.connect()

    trace: list[dict[str, Any]] = []
    usage = Usage(cost_known=True)

    def finalize(status: str, contract: GeneratedContract | None, verification: dict[str, Any], steps: int, error: str = "") -> dict[str, Any]:
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
                "mode": "generate",
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
        return result

    try:
        ctx = ToolContext(conn=conn, contract_id=None)
        tools = build_generation_tools()
        system = build_generation_prompt(contract_type, language, brief, date.today().isoformat())
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": "Draft the contract now, following the rules above."},
        ]

        for step in range(1, settings.max_generation_iterations + 1):
            response = llm.chat(messages, tools=tools, max_tokens=settings.generation_max_tokens)
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
                messages.append({"role": "user", "content": _NUDGE})
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
                    return finalize("generated", contract, verification, step)

                log.info("gen tool_call step=%s name=%s", step, name)
                result = legislation_dispatch(ctx, name, args)
                trace.append({"step": step, "tool": name, "args": args, "result": result})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        return finalize(
            "failed",
            None,
            {"status": "none", "n_clauses": 0, "n_legal_basis": 0, "n_verified": 0, "n_dropped": 0, "dropped": []},
            settings.max_generation_iterations,
            error="Reached the iteration limit before the contract was submitted.",
        )
    finally:
        if own_conn:
            conn.close()
