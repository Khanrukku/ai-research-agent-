"""Stateless research workers. Each worker is independently scalable."""
from __future__ import annotations

import asyncio
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests

from app.core.config import settings
from app.graph import client as graph
from app.rag.vector_store import semantic_search
from .base import AgentResult

ARXIV_API_URL = "https://export.arxiv.org/api/query"


def _tokens(text: str) -> set[str]:
    return {t.casefold() for t in text.replace("-", " ").split() if len(t) >= 4}


@dataclass(frozen=True)
class ArxivSearchAgent:
    name: str = "academic_search_agent"

    async def run(self, goal: str, max_results: int) -> AgentResult:
        started = time.perf_counter()
        params = {"search_query": f"all:{goal}", "start": 0, "max_results": min(max_results * 3, 30), "sortBy": "relevance", "sortOrder": "descending"}

        def request() -> str:
            r = requests.get(ARXIV_API_URL, params=params, timeout=settings.arxiv_timeout_seconds, headers={"User-Agent": "ai-research-agent/3.0"})
            r.raise_for_status()
            return r.text

        xml_text = await asyncio.to_thread(request)
        root = ET.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        terms = _tokens(goal)
        evidence: list[dict[str, Any]] = []
        for entry in root.findall("atom:entry", ns):
            title = " ".join((entry.findtext("atom:title", "", ns) or "").split())
            abstract = " ".join((entry.findtext("atom:summary", "", ns) or "").split())
            url = (entry.findtext("atom:id", "", ns) or "").strip()
            published = (entry.findtext("atom:published", "", ns) or "").strip()
            overlap = len(terms & _tokens(title + " " + abstract)) / max(len(terms), 1)
            if title and abstract and url:
                evidence.append({"source": "arXiv", "title": title, "text": abstract, "url": url, "published": published, "score": round(overlap, 4)})
        evidence.sort(key=lambda x: x["score"], reverse=True)
        duration = (time.perf_counter() - started) * 1000
        return AgentResult(self.name, evidence[:max_results], f"Retrieved {min(len(evidence), max_results)} academic sources", duration)


@dataclass(frozen=True)
class VectorRagAgent:
    name: str = "semantic_rag_agent"

    async def run(self, goal: str, max_results: int) -> AgentResult:
        started = time.perf_counter()
        try:
            hits = await semantic_search(goal, n_results=max_results)
        except Exception:
            hits = []
        evidence = [{"source": "ChromaDB", "title": h.get("metadata", {}).get("title", h.get("doc_id", "local")), "text": h.get("text", ""), "url": h.get("metadata", {}).get("source", ""), "doc_id": h.get("doc_id"), "score": h.get("score", 0.0)} for h in hits]
        return AgentResult(self.name, evidence, f"Retrieved {len(evidence)} semantic evidence chunks", (time.perf_counter() - started) * 1000)


@dataclass(frozen=True)
class KnowledgeGraphAgent:
    name: str = "knowledge_graph_agent"

    async def run(self, goal: str, max_results: int) -> AgentResult:
        started = time.perf_counter()
        terms = [t for t in _tokens(goal) if t not in {"what", "which", "compare", "explain", "research"}][:6]
        try:
            matches = await asyncio.gather(*(graph.search_entities(term, limit=3) for term in terms)) if terms else []
            entities = {e["id"]: e for group in matches for e in group if e.get("id")}
            evidence: list[dict[str, Any]] = []
            for entity in list(entities.values())[:max_results]:
                neighborhood = await graph.bfs_entity_neighborhood(entity["name"], max_depth=settings.max_graph_hops, limit=max_results)
                evidence.append({"source": "Neo4j", "title": entity["name"], "text": entity.get("description", ""), "url": "", "entity": entity, "graph": neighborhood, "score": 1.0})
        except Exception:
            evidence = []
        return AgentResult(self.name, evidence, f"Expanded {len(evidence)} graph entities with BFS", (time.perf_counter() - started) * 1000)
