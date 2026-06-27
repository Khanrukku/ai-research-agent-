"""
app/graph/extractor.py
----------------------
Uses Gemini to extract structured entities and relations from raw text,
then persists them to the Neo4j knowledge graph.

Entity types recognised:
  PERSON, ORGANIZATION, TECHNOLOGY, CONCEPT, LOCATION, EVENT, PAPER
"""
from __future__ import annotations

import asyncio
from typing import Any

from app.core.llm import extract_json
from app.core.logging import get_logger
from app.graph import client as graph

logger = get_logger(__name__)

# ──────────────────────────────────────────────
#  Extraction prompt
# ──────────────────────────────────────────────

_EXTRACT_PROMPT = """
Analyse the following research text and extract all named entities and the
relationships between them.

Return a JSON object with this exact shape:
{{
  "entities": [
    {{"name": "...", "type": "PERSON|ORGANIZATION|TECHNOLOGY|CONCEPT|LOCATION|EVENT|PAPER", "description": "one sentence"}}
  ],
  "relations": [
    {{"source": "<entity name>", "target": "<entity name>", "relation": "verb phrase", "weight": 0.0-1.0}}
  ]
}}

Rules:
- Only include entities explicitly mentioned in the text.
- Normalise entity names (e.g. "Google DeepMind" not "deepmind").
- Relation should be a concise active-voice phrase: "developed", "published research on", "acquired", etc.
- Weight reflects confidence (1.0 = certain, 0.5 = inferred).

TEXT:
{text}
"""


# ──────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────

async def extract_and_store(
    text: str,
    doc_id: str,
    doc_title: str = "",
    doc_source: str = "",
) -> dict[str, Any]:
    """
    Extract entities + relations from `text` and write them to Neo4j.

    Returns a summary dict with counts.
    """
    # 1. Ask Gemini to extract structured data
    prompt = _EXTRACT_PROMPT.format(text=text[:4000])  # stay within context
    try:
        data = await extract_json(prompt)
    except Exception as exc:
        logger.error("Entity extraction failed: %s", exc)
        return {"entities": 0, "relations": 0, "error": str(exc)}

    entities: list[dict] = data.get("entities", [])
    relations: list[dict] = data.get("relations", [])

    # 2. Upsert document node
    await graph.upsert_document(
        doc_id=doc_id,
        title=doc_title,
        source=doc_source,
        content_snippet=text[:300],
    )

    # 3. Upsert entities and link to document
    entity_name_to_id: dict[str, str] = {}
    for ent in entities:
        name = ent.get("name", "").strip()
        if not name:
            continue
        eid = await graph.upsert_entity(
            name=name,
            entity_type=ent.get("type", "CONCEPT"),
            description=ent.get("description", ""),
        )
        entity_name_to_id[name.lower()] = eid
        await graph.link_entity_to_document(eid, doc_id)

    # 4. Create relation edges
    for rel in relations:
        src_name = rel.get("source", "").lower()
        tgt_name = rel.get("target", "").lower()
        src_id = entity_name_to_id.get(src_name)
        tgt_id = entity_name_to_id.get(tgt_name)

        if src_id and tgt_id:
            await graph.link_entities(
                entity_id_a=src_id,
                entity_id_b=tgt_id,
                relation=rel.get("relation", "related to"),
                weight=float(rel.get("weight", 1.0)),
            )

    logger.info(
        "Stored %d entities, %d relations for doc '%s'",
        len(entities), len(relations), doc_title,
    )
    return {
        "entities": len(entities),
        "relations": len(relations),
        "entity_names": [e.get("name") for e in entities],
    }
