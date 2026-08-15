"""Benchmark sequential vs concurrent stateless workers.

This measures orchestration latency reduction, not human research time.
Use it as the reproducible engineering benchmark behind scalability claims.
"""
from __future__ import annotations

import asyncio
import time

from app.agents.workers import ArxivSearchAgent, KnowledgeGraphAgent, VectorRagAgent


async def run(sequential: bool) -> float:
    workers = (ArxivSearchAgent(), VectorRagAgent(), KnowledgeGraphAgent())
    start = time.perf_counter()
    if sequential:
        for worker in workers:
            try:
                await worker.run("retrieval augmented generation", 3)
            except Exception:
                pass
    else:
        await asyncio.gather(*(worker.run("retrieval augmented generation", 3) for worker in workers), return_exceptions=True)
    return time.perf_counter() - start


async def main() -> None:
    sequential = await run(True)
    concurrent = await run(False)
    reduction = (1 - concurrent / sequential) * 100 if sequential else 0
    print({"sequential_seconds": round(sequential, 3), "concurrent_seconds": round(concurrent, 3), "latency_reduction_percent": round(reduction, 2)})


if __name__ == "__main__":
    asyncio.run(main())
