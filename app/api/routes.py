"""
app/api/routes.py
-----------------
FastAPI REST API exposing:
  POST /ingest          — ingest a document into vector + graph stores
  POST /query           — RAG query (vector + graph, no agentic loop)
  POST /research        — full multi-step research agent
  GET  /graph/entities  — search entity graph
  GET  /graph/stats     — graph statistics
  GET  /health          — health check
"""
from __future__ import annotations

import uuid
from typing import Any, Annotated

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.graph import client as graph
from app.graph.extractor import extract_and_store
from app.rag.retriever import answer_with_rag
from app.rag.vector_store import ingest_document, get_collection_stats
from app.agents.research_agent import run_research_agent

logger = get_logger(__name__)
router = APIRouter()


# ──────────────────────────────────────────────
#  Request / Response models
# ──────────────────────────────────────────────

class IngestRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Raw document text")
    title: str = Field("Untitled", description="Human-readable title")
    source: str = Field("", description="URL or file path")
    doc_id: str | None = Field(None, description="Custom doc ID (auto-generated if omitted)")

class IngestResponse(BaseModel):
    doc_id: str
    chunks: int
    entities: int
    relations: int

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    n_results: int = Field(5, ge=1, le=20)
    include_graph: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    vector_hits: list[dict]
    graph_entities: list[dict]

class ResearchRequest(BaseModel):
    goal: str = Field(..., min_length=10, description="Research question or task")

class ResearchStep(BaseModel):
    step: int
    thought: str
    action: str
    action_input: str
    observation: str
    duration_ms: float

class ResearchResponse(BaseModel):
    goal: str
    answer: str
    steps: list[ResearchStep]
    total_duration_ms: float
    sources: list[str]


# ──────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────

@router.get("/health")
async def health() -> dict:
    vector_stats = await get_collection_stats()
    try:
        graph_stats = await graph.get_graph_stats()
    except Exception:
        graph_stats = {"error": "Neo4j unavailable"}
    return {
        "status": "ok",
        "vector_store": vector_stats,
        "graph": graph_stats,
    }


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse:
    """
    Ingest a document:
      1. Chunk + embed → ChromaDB
      2. Extract entities/relations → Neo4j
    """
    doc_id = req.doc_id or f"doc-{uuid.uuid4().hex[:8]}"

    # Vector store
    n_chunks = await ingest_document(
        doc_id=doc_id,
        text=req.text,
        metadata={"title": req.title, "source": req.source},
    )

    # Knowledge graph
    kg_result = await extract_and_store(
        text=req.text,
        doc_id=doc_id,
        doc_title=req.title,
        doc_source=req.source,
    )

    return IngestResponse(
        doc_id=doc_id,
        chunks=n_chunks,
        entities=kg_result.get("entities", 0),
        relations=kg_result.get("relations", 0),
    )


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    """Single-turn RAG query (fast, non-agentic)."""
    result = await answer_with_rag(
        question=req.question,
        n_vector_results=req.n_results,
        include_graph=req.include_graph,
    )
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        vector_hits=result["vector_hits"],
        graph_entities=result["graph_entities"],
    )


@router.post("/research", response_model=ResearchResponse)
async def research(req: ResearchRequest) -> ResearchResponse:
    """
    Multi-step agentic research — the agent autonomously decides
    which tools to call until it has enough context to synthesise.
    This is the flagship endpoint for the portfolio demo.
    """
    result = await run_research_agent(goal=req.goal)
    return ResearchResponse(
        goal=result.goal,
        answer=result.answer,
        steps=[
            ResearchStep(
                step=s.step,
                thought=s.thought,
                action=s.action,
                action_input=s.action_input,
                observation=s.observation,
                duration_ms=s.duration_ms,
            )
            for s in result.steps
        ],
        total_duration_ms=result.total_duration_ms,
        sources=result.sources,
    )


@router.get("/graph/entities")
async def search_graph_entities(
    q: Annotated[str, Query(min_length=1)],
    limit: int = 10,
) -> dict:
    entities = await graph.search_entities(q, limit=limit)
    return {"entities": entities, "count": len(entities)}


@router.get("/graph/stats")
async def graph_stats() -> dict:
    return await graph.get_graph_stats()
