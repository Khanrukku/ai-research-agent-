"""Model Context Protocol server exposing research tools.

Run with: python -m app.mcp.server
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from app.agents.research_agent import run_research_agent
from app.graph.client import bfs_entity_neighborhood, search_entities
from app.rag.vector_store import semantic_search

mcp = FastMCP("AI Research Agent")


@mcp.tool()
async def search_papers(query: str, limit: int = 5) -> list[dict]:
    """Search academic evidence through the research agent's paper source."""
    result = await run_research_agent(query, depth="quick")
    return [{"url": url} for url in result.sources[:limit]]


@mcp.tool()
async def semantic_search_tool(query: str, limit: int = 5) -> list[dict]:
    """Search the centralized ChromaDB semantic index."""
    return await semantic_search(query, n_results=limit)


@mcp.tool()
async def graph_search(entity: str, limit: int = 10) -> list[dict]:
    """Find Neo4j entities matching a concept."""
    return await search_entities(entity, limit=limit)


@mcp.tool()
async def graph_bfs(entity: str, max_depth: int = 2, limit: int = 25) -> dict:
    """Expand relationships from an entity using breadth-first traversal."""
    return await bfs_entity_neighborhood(entity, max_depth=max_depth, limit=limit)


if __name__ == "__main__":
    mcp.run()
