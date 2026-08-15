"""REST API for ingestion, RAG, graph search, and research."""
from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.agents.research_agent import run_research_agent
from app.core.logging import get_logger
from app.graph import client as graph
from app.graph.extractor import extract_and_store
from app.rag.retriever import answer_with_rag
from app.rag.vector_store import get_collection_stats, ingest_document

logger = get_logger(__name__)
router = APIRouter()


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=10)
    title: str = Field("Untitled", min_length=1, max_length=300)
    source: str = Field("", max_length=1000)
    doc_id: str | None = Field(None, pattern=r"^[A-Za-z0-9._:-]+$")


class IngestResponse(BaseModel):
    doc_id: str
    chunks: int
    entities: int
    relations: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    n_results: int = Field(5, ge=1, le=20)
    include_graph: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    vector_hits: list[dict]
    graph_entities: list[dict]


class ResearchRequest(BaseModel):
    goal: str = Field(..., min_length=10, max_length=2000)
    depth: str = Field("standard", pattern="^(quick|standard|comprehensive)$")


class ResearchStep(BaseModel):
    step: int
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


@router.get("/health")
async def health() -> dict:
    vector = await get_collection_stats()
    try:
        graph_stats = await graph.get_graph_stats()
        graph_status = "ok"
    except Exception as exc:
        graph_stats = {"error": str(exc)}
        graph_status = "unavailable"
    return {"status": "ok", "vector_store": vector, "graph": graph_stats, "graph_status": graph_status}


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse:
    doc_id = req.doc_id or f"doc-{uuid.uuid4().hex[:10]}"
    try:
        chunks = await ingest_document(doc_id, req.text, {"title": req.title, "source": req.source})
        kg = await extract_and_store(req.text, doc_id, req.title, req.source)
    except Exception as exc:
        logger.exception("Ingestion failed for %s", doc_id)
        raise HTTPException(status_code=502, detail=f"Ingestion failed: {exc}") from exc
    return IngestResponse(doc_id=doc_id, chunks=chunks, entities=kg.get("entities", 0), relations=kg.get("relations", 0))


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    try:
        result = await answer_with_rag(req.question, req.n_results, req.include_graph)
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=502, detail=f"RAG query failed: {exc}") from exc
    return QueryResponse(**result)


@router.post("/research", response_model=ResearchResponse)
async def research(req: ResearchRequest) -> ResearchResponse:
    try:
        result = await run_research_agent(req.goal, depth=req.depth)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Research run failed")
        raise HTTPException(status_code=502, detail=f"Research run failed: {exc}") from exc
    return ResearchResponse(
        goal=result.goal,
        answer=result.answer,
        steps=[ResearchStep(**step.__dict__) for step in result.steps],
        total_duration_ms=result.total_duration_ms,
        sources=result.sources,
    )


@router.get("/graph/entities")
async def search_graph_entities(q: Annotated[str, Query(min_length=1)], limit: int = Query(10, ge=1, le=50)) -> dict:
    try:
        entities = await graph.search_entities(q, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {exc}") from exc
    return {"entities": entities, "count": len(entities)}


@router.get("/graph/bfs")
async def graph_bfs(q: Annotated[str, Query(min_length=1)], max_depth: int = Query(2, ge=1, le=4), limit: int = Query(25, ge=1, le=100)) -> dict:
    try:
        return await graph.bfs_entity_neighborhood(q, max_depth=max_depth, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {exc}") from exc


@router.get("/graph/stats")
async def graph_stats() -> dict:
    try:
        return await graph.get_graph_stats()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {exc}") from exc
