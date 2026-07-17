"""Token + cost usage tracking for OpenRouter calls.

OpenRouter returns token counts on every response, and — when the request asks
for it (`usage: {include: true}`) — a `cost` field in USD. We accumulate both
per agent run and (in the GUI) across a session.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    cost_known: bool = True  # False if the provider didn't report cost

    def add(self, other: "Usage") -> "Usage":
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.cost_usd += other.cost_usd
        self.calls += other.calls
        self.cost_known = self.cost_known and other.cost_known
        return self

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_response(cls, response: Any) -> "Usage":
        """Extract usage from an OpenAI/OpenRouter chat-completions response."""
        raw = getattr(response, "usage", None)
        if raw is None:
            return cls(calls=1, cost_known=False)

        data = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)
        cost = data.get("cost")
        # OpenRouter sometimes nests total under cost_details; fall back gracefully.
        if cost is None and isinstance(data.get("cost_details"), dict):
            cost = data["cost_details"].get("total_cost")

        return cls(
            prompt_tokens=int(data.get("prompt_tokens") or 0),
            completion_tokens=int(data.get("completion_tokens") or 0),
            total_tokens=int(data.get("total_tokens") or 0),
            cost_usd=float(cost) if cost is not None else 0.0,
            calls=1,
            cost_known=cost is not None,
        )
