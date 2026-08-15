"""Reproducible retrieval evaluation: Precision@K, Recall@K and MRR.

Requires GEMINI_API_KEY because Chroma embeddings use Gemini embeddings.
Run: python -m evals.evaluate_retrieval
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.rag.vector_store import ingest_document, semantic_search

QUERIES = [
    ("How does retrieval augmented generation use external evidence?", {"rag"}),
    ("What architecture uses self attention without recurrence?", {"transformer"}),
    ("How does fine tuning adapt a pretrained model?", {"finetune"}),
    ("How can entities and relationships be expanded through graph traversal?", {"graph"}),
]


async def main() -> None:
    fixture = json.loads(Path(__file__).with_name("fixtures").joinpath("retrieval.json").read_text())
    for doc in fixture:
        await ingest_document(f"eval-{doc['id']}", doc["text"], {"title": doc["title"], "source": f"fixture://{doc['id']}"})

    precisions, recalls, reciprocal = [], [], []
    k = 3
    for query, relevant in QUERIES:
        hits = await semantic_search(query, n_results=k)
        ids = [h["doc_id"].removeprefix("eval-") for h in hits]
        found = set(ids) & relevant
        precisions.append(len(found) / k)
        recalls.append(len(found) / len(relevant))
        reciprocal.append(1 / (ids.index(next(iter(found))) + 1) if found else 0.0)
    print({"k": k, "precision_at_k": round(sum(precisions)/len(precisions), 4), "recall_at_k": round(sum(recalls)/len(recalls), 4), "mrr": round(sum(reciprocal)/len(reciprocal), 4)})


if __name__ == "__main__":
    asyncio.run(main())
