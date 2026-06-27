"""
app/main.py
-----------
FastAPI application factory.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.graph.client import ensure_schema

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 AI Research Agent starting up…")
    try:
        await ensure_schema()
        logger.info("✅ Neo4j schema verified")
    except Exception as exc:
        logger.warning("⚠️  Neo4j not available: %s", exc)
    yield
    logger.info("👋 Shutting down")


app = FastAPI(
    title="AI Research Agent",
    description=(
        "Multi-step research agent with RAG, knowledge graphs, and LLM orchestration. "
        "Built with Gemini · ChromaDB · Neo4j · FastAPI"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "name": "AI Research Agent",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
