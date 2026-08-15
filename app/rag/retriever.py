"""Evidence-grounded RAG pipeline."""
from __future__ import annotations

import re

from app.core.llm import generate
from app.core.logging import get_logger
from app.graph import client as graph
from app.rag.vector_store import semantic_search

logger = get_logger(__name__)

_RAG_SYSTEM = """You are an evidence-grounded research assistant.
Use only the supplied evidence. Every factual claim must be supported by a source.
If the evidence is insufficient, say that explicitly. Never invent citations.
Cite sources using [doc-id]."""


def _format_vector_context(hits: list[dict]) -> str:
    if not hits:
        return "No vector evidence found."
    return "\n\n---\n\n".join(
        f"[{h['doc_id']}] score={h['score']}\n{h['text']}" for h in hits
    )


def _format_graph_context(entities: list[dict]) -> str:
    if not entities:
        return "No graph evidence found."
    return "\n".join(
        f"- {e.get('type', 'ENTITY')}: {e.get('name', '')} — {e.get('description', '')}"
        for e in entities
    )


def _candidate_terms(question: str, hits: list[dict]) -> list[str]:
    terms = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", question)
    # Prefer named concepts surfaced by retrieved document titles.
    for hit in hits:
        title = hit.get("metadata", {}).get("title", "")
        terms.extend(re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", title))
    stop = {"what", "which", "where", "when", "does", "about", "compare", "explain", "research"}
    return list(dict.fromkeys(t.lower() for t in terms if t.lower() not in stop))[:8]


async def answer_with_rag(question: str, n_vector_results: int = 5, include_graph: bool = True) -> dict:
    vector_hits = await semantic_search(question, n_results=n_vector_results)
    graph_entities: list[dict] = []
    if include_graph:
        for term in _candidate_terms(question, vector_hits):
            try:
                graph_entities.extend(await graph.search_entities(term, limit=3))
            except Exception as exc:
                logger.debug("Graph lookup failed for %s: %s", term, exc)
        graph_entities = list({e["id"]: e for e in graph_entities if e.get("id")}.values())

    prompt = f"""Question: {question}

VECTOR EVIDENCE:
{_format_vector_context(vector_hits)}

KNOWLEDGE GRAPH EVIDENCE:
{_format_graph_context(graph_entities)}

Answer with a short conclusion, supporting evidence, and limitations."""
    answer = await generate(prompt, system=_RAG_SYSTEM, temperature=0.1)
    sources = list(dict.fromkeys(h["doc_id"] for h in vector_hits if h.get("doc_id")))
    return {
        "answer": answer,
        "sources": sources,
        "vector_hits": vector_hits,
        "graph_entities": graph_entities,
    }
