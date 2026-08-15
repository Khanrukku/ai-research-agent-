"""ChromaDB vector store. Supports a centralized HTTP server for stateless workers."""
from __future__ import annotations

import os
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.core.llm import embed, embed_query

COLLECTION_NAME = "research_documents"
_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is not None:
        return _client
    if settings.chroma_host:
        _client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    else:
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir, settings=ChromaSettings(anonymized_telemetry=False))
    return _client


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        _collection = get_client().get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    return _collection


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        boundary = text.rfind(". ", start, end)
        if boundary > start + chunk_size // 2:
            end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _chunk_id(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}::chunk::{chunk_index}"


async def ingest_document(doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> int:
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("Document text cannot be empty")
    collection = get_collection()
    metadata = metadata or {}
    embeddings = [await embed(chunk) for chunk in chunks]
    ids = [_chunk_id(doc_id, i) for i in range(len(chunks))]
    metadatas = [{"doc_id": doc_id, "chunk_index": i, "total_chunks": len(chunks), **{k: str(v) for k, v in metadata.items()}} for i in range(len(chunks))]
    collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    return len(chunks)


async def semantic_search(query: str, n_results: int | None = None, filter_doc_id: str | None = None) -> list[dict[str, Any]]:
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []
    n_results = min(n_results or settings.max_search_results, count)
    kwargs: dict[str, Any] = {"query_embeddings": [await embed_query(query)], "n_results": n_results, "include": ["documents", "metadatas", "distances"]}
    if filter_doc_id:
        kwargs["where"] = {"doc_id": filter_doc_id}
    results = collection.query(**kwargs)
    hits: list[dict[str, Any]] = []
    for text, meta, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        hits.append({"text": text, "doc_id": meta.get("doc_id"), "score": round(max(0.0, 1 - float(distance)), 4), "metadata": meta})
    return hits


async def get_collection_stats() -> dict[str, Any]:
    return {"total_chunks": get_collection().count(), "distance_metric": "cosine", "index": "HNSW", "backend": "http" if settings.chroma_host else "persistent"}
