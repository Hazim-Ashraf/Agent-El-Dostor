"""Lightweight run-level observability.

Each agent run appends one JSON line to a trace log (default `logs/agent_runs.jsonl`,
under the bind-mounted repo) and emits a one-line summary to the app logs — tool
calls, verification outcome, tokens, USD cost, and latency.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from app.core.logging import get_logger

log = get_logger("trace")

TRACE_PATH = os.environ.get("TRACE_LOG", "logs/agent_runs.jsonl")


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def record_run(record: dict[str, Any]) -> None:
    entry = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    try:
        directory = os.path.dirname(TRACE_PATH)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(TRACE_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as err:  # noqa: BLE001 - tracing must never break a run
        log.warning("trace write failed: %s", err)

    log.info(
        "run %s: status=%s findings=%s tools=%s tokens=%s cost=$%.5f latency=%.2fs",
        record.get("run_id"),
        record.get("verification_status"),
        record.get("n_findings"),
        record.get("n_tool_calls"),
        record.get("total_tokens"),
        record.get("cost_usd", 0.0) or 0.0,
        record.get("latency_s", 0.0) or 0.0,
    )
