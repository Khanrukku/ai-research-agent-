"""Multi-agent research orchestration with concurrent stateless workers."""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Any

from app.core.config import settings
from app.core.llm import generate
from .base import AgentResult
from .workers import ArxivSearchAgent, KnowledgeGraphAgent, VectorRagAgent


@dataclass(frozen=True)
class ResearchStep:
    step: int
    action: str
    action_input: str
    observation: str
    duration_ms: float


@dataclass(frozen=True)
class ResearchRun:
    goal: str
    answer: str
    steps: list[ResearchStep]
    total_duration_ms: float
    sources: list[str]


class ResearchAgent:
    """Stateless orchestrator: workers can run concurrently and scale horizontally."""

    def __init__(self, max_sources: int | None = None):
        self.max_sources = max_sources or settings.max_search_results
        self.workers = (ArxivSearchAgent(), VectorRagAgent(), KnowledgeGraphAgent())

    async def research(self, goal: str, depth: str = "standard") -> ResearchRun:
        goal = goal.strip()
        if len(goal) < 3:
            raise ValueError("Research goal cannot be empty")
        if depth not in {"quick", "standard", "comprehensive"}:
            raise ValueError("depth must be one of: quick, standard, comprehensive")
        started = time.perf_counter()
        results = await asyncio.gather(*(w.run(goal, self.max_sources) for w in self.workers), return_exceptions=True)
        successful = [r for r in results if isinstance(r, AgentResult)]
        steps = [ResearchStep(i + 1, r.name, goal, r.observation, round(r.duration_ms, 2)) for i, r in enumerate(successful)]
        evidence = [item for r in successful for item in r.evidence]
        evidence = self._rank(evidence, goal)[: self.max_sources * 3]
        if depth == "comprehensive":
            steps.append(ResearchStep(len(steps) + 1, "evidence_verification", goal, self._verify(evidence), 0.0))
        context = self._context(evidence)
        synth_started = time.perf_counter()
        answer = await self._synthesize(goal, context)
        steps.append(ResearchStep(len(steps) + 1, "synthesis_agent", goal, "Generated cited evidence-grounded report", round((time.perf_counter() - synth_started) * 1000, 2)))
        sources = list(dict.fromkeys(e.get("url") for e in evidence if e.get("url")))
        return ResearchRun(goal, answer, steps, round((time.perf_counter() - started) * 1000, 2), sources)

    @staticmethod
    def _rank(evidence: list[dict[str, Any]], goal: str) -> list[dict[str, Any]]:
        terms = {t.casefold() for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", goal)}
        ranked = []
        for e in evidence:
            lexical = len(terms & {t.casefold() for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", e.get("title", "") + " " + e.get("text", ""))}) / max(len(terms), 1)
            score = 0.7 * float(e.get("score", 0.0)) + 0.3 * lexical
            ranked.append({**e, "final_score": round(score, 4)})
        return sorted(ranked, key=lambda x: x["final_score"], reverse=True)

    @staticmethod
    def _verify(evidence: list[dict[str, Any]]) -> str:
        sources = {e.get("source") for e in evidence if e.get("source")}
        return f"Evidence pool contains {len(evidence)} items across {len(sources)} source types; synthesis is instructed to flag unsupported claims."

    @staticmethod
    def _context(evidence: list[dict[str, Any]]) -> str:
        return "\n\n---\n\n".join(f"[{i}] {e.get('source')} | {e.get('title')} | score={e.get('final_score')}\n{e.get('text', '')}\nURL: {e.get('url', '')}" for i, e in enumerate(evidence, 1)) or "No evidence retrieved."

    async def _synthesize(self, goal: str, context: str) -> str:
        prompt = f"""Research goal: {goal}\n\nEvidence:\n{context}\n\nWrite a concise academic research report. Use only supplied evidence. Cite claims as [1], [2], etc. Distinguish evidence from interpretation, identify conflicting/weak evidence, and end with Sources containing URLs."""
        try:
            return await generate(prompt, system="You are a rigorous research synthesis agent. Never invent evidence or citations.", temperature=0.1, max_tokens=2500)
        except EnvironmentError:
            return "# Research Report\n\nGemini API key is not configured. Retrieved evidence:\n\n" + context


async def run_research_agent(goal: str, depth: str = "standard") -> ResearchRun:
    return await ResearchAgent().research(goal, depth=depth)
