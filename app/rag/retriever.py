"""
app/rag/retriever.py
--------------------
Retrieval-Augmented Generation pipeline.

Two retrieval strategies are combined:
  1. Vector search  — semantic similarity via ChromaDB
  2. Graph context  — entity neighbourhood from Neo4j

The retrieved context is fed into Gemini to produce grounded answers.
"""
from __future__ import annotations

from app.core.llm import generate
from app.core.logging import get_logger
from app.graph import client as graph
from app.rag.vector_store import semantic_search

logger = get_logger(__name__)

# ──────────────────────────────────────────────
#  Prompts
# ──────────────────────────────────────────────

_RAG_SYSTEM = """
You are an expert AI research assistant with access to a curated knowledge base.
Answer the user's question using ONLY the provided context.
If the context doesn't contain enough information, say so — do not hallucinate.
Cite the document IDs when referencing specific facts (e.g. [doc-123]).
Be concise but thorough.
"""

_RAG_PROMPT_TEMPLATE = """
=== VECTOR SEARCH CONTEXT ===
{vector_context}

=== KNOWLEDGE GRAPH CONTEXT ===
{graph_context}

=== USER QUESTION ===
{question}

Please provide a well-structured answer.
"""


# ──────────────────────────────────────────────
#  Context builders
# ──────────────────────────────────────────────

def _format_vector_context(hits: list[dict]) -> str:
    if not hits:
        return "No relevant documents found."
    parts = []
    for h in hits:
        parts.append(
            f"[{h['doc_id']}] (score={h['score']})\n{h['text']}"
        )
    return "\n\n---\n\n".join(parts)


def _format_graph_context(entities: list[dict]) -> str:
    if not entities:
        return "No related entities found in the knowledge graph."
    lines = []
    for e in entities:
        lines.append(f"• {e.get('type', 'ENTITY')}: {e['name']} — {e.get('description', '')}")
    return "\n".join(lines)


# ──────────────────────────────────────────────
#  Main RAG pipeline
# ──────────────────────────────────────────────

async def answer_with_rag(
    question: str,
    n_vector_results: int = 5,
    include_graph: bool = True,
) -> dict:
    """
    Full RAG pipeline:
      1. Semantic search in vector store
      2. Entity lookup in knowledge graph
      3. Combine context and generate answer

    Returns:
        {
          "answer": str,
          "vector_hits": list,
          "graph_entities": list,
          "sources": list[str],
        }
    """
    logger.info("RAG query: %s", question[:80])

    # ── Step 1: Vector retrieval ──
    vector_hits = await semantic_search(question, n_results=n_vector_results)
    vector_context = _format_vector_context(vector_hits)

    # ── Step 2: Graph context ──
    graph_entities: list[dict] = []
    if include_graph:
        # Extract the key noun phrase from the question for entity lookup
        # (A simple heuristic; could be improved with NER)
        keywords = [w for w in question.split() if len(w) > 4][:3]
        for kw in keywords:
            ents = await graph.search_entities(kw, limit=5)
            graph_entities.extend(ents)
        # Deduplicate
        seen = set()
        unique_entities = []
        for e in graph_entities:
            if e["id"] not in seen:
                unique_entities.append(e)
                seen.add(e["id"])
        graph_entities = unique_entities

    graph_context = _format_graph_context(graph_entities)

    # ── Step 3: LLM generation ──
    prompt = _RAG_PROMPT_TEMPLATE.format(
        vector_context=vector_context,
        graph_context=graph_context,
        question=question,
    )
    answer = await generate(prompt, system=_RAG_SYSTEM, temperature=0.1)

    # ── Step 4: Collect source doc IDs ──
    sources = list({h["doc_id"] for h in vector_hits if h.get("doc_id")})

    return {
        "answer": answer,
        "vector_hits": vector_hits,
        "graph_entities": graph_entities,
        "sources": sources,
    }
