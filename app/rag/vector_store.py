"""
app/rag/vector_store.py
-----------------------
ChromaDB-backed vector store.

Responsibilities:
  - Chunking documents into overlapping windows
  - Generating Gemini embeddings
  - Upserting chunks with metadata
  - Semantic similarity search
"""
from __future__ import annotations

import hashlib
import os
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.core.llm import embed, embed_query
from app.core.logging import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────
#  ChromaDB client & collection singleton
# ──────────────────────────────────────────────

_chroma_client: chromadb.Client | None = None
_collection: chromadb.Collection | None = None
COLLECTION_NAME = "research_documents"


def get_collection() -> chromadb.Collection:
    global _chroma_client, _collection
    if _collection is None:
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection '%s' ready", COLLECTION_NAME)
    return _collection


# ──────────────────────────────────────────────
#  Document chunking
# ──────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[str]:
    """
    Split text into overlapping word-level chunks.

    Args:
        chunk_size: Target chunk size in characters.
        overlap:    Overlap between consecutive chunks in characters.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Try to break at a sentence boundary
        boundary = text.rfind(". ", start, end)
        if boundary != -1 and boundary > start + overlap:
            end = boundary + 1
        chunks.append(text[start:end].strip())
        start = end - overlap

    return [c for c in chunks if c]


def _chunk_id(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}::chunk::{chunk_index}"


# ──────────────────────────────────────────────
#  Ingest
# ──────────────────────────────────────────────

async def ingest_document(
    doc_id: str,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> int:
    """
    Chunk a document, embed each chunk, and store in ChromaDB.

    Returns the number of chunks stored.
    """
    collection = get_collection()
    metadata = metadata or {}
    chunks = chunk_text(text)

    ids: list[str] = []
    embeddings: list[list[float]] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for i, chunk in enumerate(chunks):
        chunk_id = _chunk_id(doc_id, i)
        embedding = await embed(chunk)

        ids.append(chunk_id)
        embeddings.append(embedding)
        documents.append(chunk)
        metadatas.append({
            "doc_id": doc_id,
            "chunk_index": i,
            "total_chunks": len(chunks),
            **{k: str(v) for k, v in metadata.items()},
        })

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    logger.info("Ingested doc '%s' → %d chunks", doc_id, len(chunks))
    return len(chunks)


# ──────────────────────────────────────────────
#  Search
# ──────────────────────────────────────────────

async def semantic_search(
    query: str,
    n_results: int | None = None,
    filter_doc_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve the most semantically similar chunks for a query.

    Args:
        query:         Natural language query.
        n_results:     Number of chunks to return.
        filter_doc_id: Optionally restrict to a single document.

    Returns:
        List of dicts with keys: text, doc_id, score, metadata.
    """
    n_results = n_results or settings.max_search_results
    collection = get_collection()
    query_embedding = await embed_query(query)

    where = {"doc_id": filter_doc_id} if filter_doc_id else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count() or 1),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits: list[dict] = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text": text,
            "doc_id": meta.get("doc_id"),
            "score": round(1 - dist, 4),   # cosine distance → similarity
            "metadata": meta,
        })

    return hits


async def get_collection_stats() -> dict:
    collection = get_collection()
    return {"total_chunks": collection.count()}
