"""Small stateless agent primitives used by the research orchestrator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class AgentResult:
    name: str
    evidence: list[dict[str, Any]]
    observation: str
    duration_ms: float


class ResearchWorker(Protocol):
    name: str

    async def run(self, goal: str, max_results: int) -> AgentResult: ...
