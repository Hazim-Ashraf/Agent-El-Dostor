"""Per-run context passed to tool dispatch (DB connection + active contract)."""
from __future__ import annotations

from dataclasses import dataclass

import psycopg


@dataclass
class ToolContext:
    conn: psycopg.Connection
    contract_id: str | None = None
