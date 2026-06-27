"""
app/graph/client.py
-------------------
Neo4j knowledge-graph layer.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any

from neo4j import GraphDatabase, Driver
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_driver: Driver | None = None


def get_driver() -> Driver:
    global _driver
    if _driver is None:
        if not settings.neo4j_uri or not settings.neo4j_password:
            raise EnvironmentError(
                "Neo4j credentials not set. "
                "Create a free instance at https://neo4j.com/cloud/aura "
                "and add NEO4J_URI / NEO4J_PASSWORD to your .env"
            )
        # Replace +s with +ssc to allow self-signed certificates (Python 3.14 fix)
        uri = (
            settings.neo4j_uri
            .replace("neo4j+s://", "neo4j+ssc://")
            .replace("bolt+s://", "bolt+ssc://")
        )
        _driver = GraphDatabase.driver(
            uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        logger.info("Neo4j driver initialised → %s", uri)
    return _driver


def close_driver() -> None:
    global _driver
    if _driver:
        _driver.close()
        _driver = None


async def _run(cypher: str, **params: Any) -> list[dict]:
    """Execute a Cypher statement in a thread pool and return records as dicts."""
    driver = get_driver()
    loop = asyncio.get_event_loop()

    def _sync():
        with driver.session() as session:
            result = session.run(cypher, **params)
            return [dict(r) for r in result]

    return await loop.run_in_executor(None, _sync)


async def ensure_schema() -> None:
    """Create indexes and constraints if they don't exist yet."""
    stmts = [
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
        "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
        "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    ]
    for stmt in stmts:
        try:
            await _run(stmt)
        except Exception as exc:
            logger.debug("Schema stmt skipped (%s): %s", exc.__class__.__name__, stmt[:60])
    logger.info("Neo4j schema ready")


async def upsert_document(
    doc_id: str,
    title: str,
    source: str,
    content_snippet: str = "",
) -> str:
    await _run(
        """
        MERGE (d:Document {id: $doc_id})
        SET   d.title    = $title,
              d.source   = $source,
              d.snippet  = $content_snippet,
              d.updated  = $ts
        """,
        doc_id=doc_id,
        title=title,
        source=source,
        content_snippet=content_snippet[:500],
        ts=datetime.utcnow().isoformat(),
    )
    return doc_id


async def upsert_entity(
    name: str,
    entity_type: str,
    description: str = "",
) -> str:
    entity_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{entity_type}:{name.lower()}"))
    await _run(
        """
        MERGE (e:Entity {id: $entity_id})
        SET   e.name        = $name,
              e.type        = $entity_type,
              e.description = $description
        """,
        entity_id=entity_id,
        name=name,
        entity_type=entity_type,
        description=description,
    )
    return entity_id


async def link_entity_to_document(entity_id: str, doc_id: str) -> None:
    await _run(
        """
        MATCH (e:Entity {id: $entity_id})
        MATCH (d:Document {id: $doc_id})
        MERGE (d)-[:MENTIONS]->(e)
        """,
        entity_id=entity_id,
        doc_id=doc_id,
    )


async def link_entities(
    entity_id_a: str,
    entity_id_b: str,
    relation: str,
    weight: float = 1.0,
) -> None:
    await _run(
        """
        MATCH (a:Entity {id: $a})
        MATCH (b:Entity {id: $b})
        MERGE (a)-[r:RELATED_TO {relation: $relation}]->(b)
        SET r.weight = $weight
        """,
        a=entity_id_a,
        b=entity_id_b,
        relation=relation,
        weight=weight,
    )


async def get_entity_neighbourhood(
    entity_name: str,
    hops: int | None = None,
) -> dict:
    hops = hops or settings.max_graph_hops
    rows = await _run(
        f"""
        MATCH (start:Entity)
        WHERE toLower(start.name) CONTAINS toLower($name)
        CALL apoc.path.subgraphAll(start, {{maxLevel: {hops}}})
        YIELD nodes, relationships
        RETURN nodes, relationships
        LIMIT 1
        """,
        name=entity_name,
    )

    if not rows:
        rows = await _run(
            """
            MATCH (start:Entity)
            WHERE toLower(start.name) CONTAINS toLower($name)
            OPTIONAL MATCH (start)-[r:RELATED_TO*1..2]-(neighbour:Entity)
            RETURN start, collect(DISTINCT neighbour) AS neighbours,
                   collect(DISTINCT r) AS rels
            LIMIT 1
            """,
            name=entity_name,
        )

    return rows[0] if rows else {"entities": [], "relations": []}


async def search_entities(query: str, limit: int = 10) -> list[dict]:
    rows = await _run(
        """
        MATCH (e:Entity)
        WHERE toLower(e.name)        CONTAINS toLower($q)
           OR toLower(e.description) CONTAINS toLower($q)
        RETURN e.id AS id, e.name AS name, e.type AS type,
               e.description AS description
        LIMIT $limit
        """,
        q=query,
        limit=limit,
    )
    return rows


async def get_entity_documents(entity_name: str) -> list[dict]:
    rows = await _run(
        """
        MATCH (d:Document)-[:MENTIONS]->(e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($name)
        RETURN d.id AS id, d.title AS title, d.source AS source,
               d.snippet AS snippet
        """,
        name=entity_name,
    )
    return rows


async def get_graph_stats() -> dict:
    rows = await _run(
        """
        MATCH (e:Entity)  WITH count(e) AS entities
        MATCH (d:Document) WITH entities, count(d) AS documents
        MATCH ()-[r:RELATED_TO]->() WITH entities, documents, count(r) AS relations
        RETURN entities, documents, relations
        """
    )
    return rows[0] if rows else {}