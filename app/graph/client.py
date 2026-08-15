"""Neo4j persistence helpers. Graph support is optional at runtime."""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

from neo4j import Driver, GraphDatabase

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)
_driver: Driver | None = None


def get_driver() -> Driver:
    global _driver
    if _driver is None:
        if not settings.neo4j_uri or not settings.neo4j_password:
            raise EnvironmentError("Neo4j credentials are not configured")
        _driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
    return _driver


def close_driver() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


async def _run(cypher: str, **params: Any) -> list[dict]:
    driver = get_driver()
    loop = asyncio.get_running_loop()

    def execute() -> list[dict]:
        with driver.session() as session:
            return [dict(record) for record in session.run(cypher, **params)]

    return await loop.run_in_executor(None, execute)


async def ensure_schema() -> None:
    statements = [
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
        "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
    ]
    for statement in statements:
        await _run(statement)


async def upsert_document(doc_id: str, title: str, source: str, content_snippet: str = "") -> str:
    await _run(
        """
        MERGE (d:Document {id: $doc_id})
        SET d.title=$title, d.source=$source, d.snippet=$snippet, d.updated=$ts
        """,
        doc_id=doc_id,
        title=title,
        source=source,
        snippet=content_snippet[:500],
        ts=datetime.now(timezone.utc).isoformat(),
    )
    return doc_id


async def upsert_entity(name: str, entity_type: str, description: str = "") -> str:
    entity_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{entity_type}:{name.casefold()}"))
    await _run(
        """
        MERGE (e:Entity {id:$entity_id})
        SET e.name=$name, e.type=$entity_type, e.description=$description
        """,
        entity_id=entity_id,
        name=name.strip(),
        entity_type=entity_type.upper(),
        description=description[:500],
    )
    return entity_id


async def link_entity_to_document(entity_id: str, doc_id: str) -> None:
    await _run(
        """MATCH (e:Entity {id:$entity_id}), (d:Document {id:$doc_id})
        MERGE (d)-[:MENTIONS]->(e)""",
        entity_id=entity_id,
        doc_id=doc_id,
    )


async def link_entities(entity_id_a: str, entity_id_b: str, relation: str, weight: float = 1.0) -> None:
    await _run(
        """MATCH (a:Entity {id:$a}), (b:Entity {id:$b})
        MERGE (a)-[r:RELATED_TO {relation:$relation}]->(b)
        SET r.weight=$weight""",
        a=entity_id_a,
        b=entity_id_b,
        relation=relation[:120],
        weight=max(0.0, min(1.0, weight)),
    )


async def search_entities(query: str, limit: int = 10) -> list[dict]:
    rows = await _run(
        """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($q)
           OR toLower(coalesce(e.description,'')) CONTAINS toLower($q)
        RETURN e.id AS id, e.name AS name, e.type AS type, e.description AS description
        ORDER BY e.name
        LIMIT $limit
        """,
        q=query,
        limit=min(max(limit, 1), 50),
    )
    return rows


async def get_entity_neighbourhood(entity_name: str, hops: int | None = None) -> dict:
    hops = min(max(hops or settings.max_graph_hops, 1), 3)
    rows = await _run(
        f"""
        MATCH (start:Entity)
        WHERE toLower(start.name) CONTAINS toLower($name)
        OPTIONAL MATCH p=(start)-[:RELATED_TO*1..{hops}]-(neighbor:Entity)
        WITH start, collect(DISTINCT neighbor) AS neighbors, collect(DISTINCT p) AS paths
        RETURN start, neighbors, paths
        LIMIT 1
        """,
        name=entity_name,
    )
    if not rows:
        return {"entities": [], "relations": []}
    row = rows[0]
    entities = [dict(row["start"])] + [dict(n) for n in row["neighbors"] if n]
    return {"entities": entities, "relations": []}


async def get_entity_documents(entity_name: str) -> list[dict]:
    return await _run(
        """MATCH (d:Document)-[:MENTIONS]->(e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($name)
        RETURN d.id AS id, d.title AS title, d.source AS source, d.snippet AS snippet""",
        name=entity_name,
    )


async def get_graph_stats() -> dict:
    rows = await _run(
        """MATCH (e:Entity) WITH count(e) AS entities
        MATCH (d:Document) WITH entities, count(d) AS documents
        MATCH ()-[r:RELATED_TO]->() RETURN entities, documents, count(r) AS relations"""
    )
    return rows[0] if rows else {"entities": 0, "documents": 0, "relations": 0}


async def bfs_entity_neighborhood(entity_name: str, max_depth: int = 2, limit: int = 50) -> dict[str, Any]:
    """Breadth-first graph expansion from a named entity.

    The query carries an explicit depth frontier so the returned nodes are
    grouped by BFS distance rather than relying on an arbitrary path order.
    """
    max_depth = min(max(max_depth, 1), 4)
    limit = min(max(limit, 1), 100)
    rows = await _run(
        f"""
        MATCH (start:Entity)
        WHERE toLower(start.name) CONTAINS toLower($name)
        CALL {{
            WITH start
            MATCH p=(start)-[:RELATED_TO*1..{max_depth}]-(neighbor:Entity)
            WITH neighbor, min(length(p)) AS depth
            RETURN neighbor.id AS id, neighbor.name AS name,
                   neighbor.type AS type, neighbor.description AS description,
                   depth
            ORDER BY depth, name
            LIMIT $limit
        }}
        RETURN start.id AS start_id, start.name AS start_name,
               collect({id:id,name:name,type:type,description:description,depth:depth}) AS nodes
        LIMIT 1
        """,
        name=entity_name,
        limit=limit,
    )
    if not rows:
        return {"start": None, "nodes": [], "algorithm": "BFS", "max_depth": max_depth}
    row = rows[0]
    return {
        "start": {"id": row["start_id"], "name": row["start_name"]},
        "nodes": row["nodes"],
        "algorithm": "BFS",
        "max_depth": max_depth,
    }
