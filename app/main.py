"""FastAPI application entry point."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.graph.client import close_driver, ensure_schema

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Research Agent")
    try:
        await ensure_schema()
    except Exception as exc:
        logger.warning("Neo4j unavailable; graph features disabled: %s", exc)
    yield
    close_driver()
    logger.info("AI Research Agent stopped")


app = FastAPI(
    title="AI Research Agent",
    description="Evidence-grounded research using arXiv, vector retrieval, Neo4j, and Gemini.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api/v1")
app.mount("/ui", StaticFiles(directory="app/static", html=True), name="ui")


@app.get("/")
async def root() -> dict:
    return {
        "name": "AI Research Agent",
        "version": app.version,
        "docs": "/docs",
        "health": "/api/v1/health",
        "ui": "/ui",
    }
