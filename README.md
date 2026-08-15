# 🔬 AI Research Agent System (Open Source)

An evidence-grounded, multi-agent research platform built for academic research workflows. It combines **Python, Gemini API, RAG, ChromaDB, Neo4j, FastAPI, Model Context Protocol (MCP), Docker, and AWS**.

## Resume-aligned architecture

- **System Design:** Multi-agent architecture with concurrent tool execution — stateless academic-search, semantic-RAG, and knowledge-graph workers; a centralized ChromaDB vector service; Neo4j relationship storage; and an async FastAPI orchestration layer. AWS Terraform runs multiple ECS Fargate replicas for horizontal scaling.
- **DSA / Retrieval:** ChromaDB is configured for **cosine distance with an HNSW index**; retrieval evaluation reports Precision@K, Recall@K, and MRR. Neo4j relationship expansion exposes an explicit **breadth-first traversal (BFS)** frontier by graph depth.
- **MCP:** External AI clients can call paper search, semantic retrieval, graph search, and graph-BFS tools through the Model Context Protocol.
- **Open source:** Modular package structure, type-annotated Python, CI, architecture documentation, tests, contributor guidelines, MIT license, Docker Compose, and AWS Terraform.

> **Important:** The repository includes reproducible evaluation and benchmarking scripts. Numerical claims such as “85%+ retrieval relevance” or “70% reduction” should be treated as benchmark outputs only when the supplied evaluation/benchmark is run in the target environment; the project does not hard-code fabricated performance numbers.

## Architecture

```text
Question
   │
   ▼
┌──────────────────────────┐
│ FastAPI / MCP Interface  │
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ Stateless Orchestrator   │
│ asyncio.gather           │
└──────┬────────┬──────────┘
       │        │
       ▼        ▼
  arXiv Agent  Chroma RAG Agent
       │        │
       └────┬───┘
            ▼
       Neo4j Graph Agent
            │
            ▼
  Evidence Ranking + Verification
            │
            ▼
      Gemini Synthesis Agent
            │
            ▼
   Cited Research Report
```

### Why the workers are stateless

Workers do not keep per-user research history in process memory. Persistent evidence belongs in ChromaDB/Neo4j (and can be extended to object storage), which lets identical API tasks run behind a load balancer or ECS service without sticky sessions.

## Technology stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| API/orchestration | FastAPI, asyncio |
| LLM | Google Gemini API |
| RAG | ChromaDB + Gemini embeddings |
| Vector metric/index | Cosine distance + HNSW |
| Knowledge graph | Neo4j |
| Graph algorithm | BFS relationship expansion |
| Academic retrieval | arXiv API |
| Agent interoperability | Model Context Protocol |
| Packaging/quality | Pytest, Ruff, Black |
| CI | GitHub Actions |
| Containers | Docker, Docker Compose |
| AWS | ECR, ECS Fargate, S3, CloudWatch, Terraform |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

For a local full stack:

```bash
docker compose up --build
```

Open:

- `http://localhost:8000/docs` — FastAPI Swagger UI
- `http://localhost:8000/ui` — research dashboard
- `http://localhost:7474` — Neo4j browser
- `http://localhost:8001` — ChromaDB host port

## API

### `POST /api/v1/research`

Runs the concurrent multi-agent workflow.

```json
{
  "goal": "Compare retrieval-augmented generation with fine-tuning for domain-specific QA",
  "depth": "comprehensive"
}
```

The response contains the synthesized answer, source URLs, worker observations, timings, and total orchestration latency.

### `POST /api/v1/ingest`

Embeds a document into ChromaDB and extracts entities/relationships into Neo4j.

### `POST /api/v1/query`

Runs hybrid local RAG with optional graph evidence.

### `GET /api/v1/graph/entities?q=transformer`

Searches graph entities.

### `GET /api/v1/graph/bfs?q=transformer`

Returns the graph relationship frontier grouped by BFS depth.

### `GET /api/v1/health`

Returns vector-store and graph status.

## MCP server

Run the MCP server as a standalone tool provider:

```bash
python -m app.mcp.server
```

Tools exposed:

- `search_papers`
- `semantic_search_tool`
- `graph_search`
- `graph_bfs`

## Evaluation

The repository contains reproducible metrics instead of hard-coded performance claims:

```bash
python -m evals.evaluate_retrieval
```

It reports:

- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)

For concurrent orchestration:

```bash
python -m evals.benchmark_parallelism
```

This compares sequential and concurrent worker execution and reports latency reduction. It is an engineering benchmark, not a fabricated human-productivity study.

## AWS deployment

`infra/aws/` contains Terraform for:

- ECR image repository with scan-on-push
- ECS Fargate cluster/service
- multiple stateless task replicas
- CloudWatch log group
- S3 research-artifact bucket
- IAM execution/task roles

See `infra/aws/README.md`. For production, move secrets from Terraform variables into AWS Secrets Manager.

## Project structure

```text
ai-research-agent/
├── app/
│   ├── agents/
│   │   ├── base.py
│   │   ├── workers.py
│   │   └── research_agent.py
│   ├── api/
│   │   └── routes.py
│   ├── core/
│   │   ├── config.py
│   │   ├── llm.py
│   │   └── logging.py
│   ├── graph/
│   │   ├── client.py
│   │   └── extractor.py
│   ├── mcp/
│   │   └── server.py
│   ├── rag/
│   │   ├── vector_store.py
│   │   └── retriever.py
│   └── main.py
├── evals/
│   ├── fixtures/retrieval.json
│   ├── evaluate_retrieval.py
│   └── benchmark_parallelism.py
├── infra/aws/
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
├── docs/ARCHITECTURE.md
├── tests/
├── CONTRIBUTING.md
├── LICENSE
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/CI.yml
```

## Interview walkthrough

> “I designed the system as a stateless multi-agent research platform. FastAPI orchestrates independent academic, semantic-RAG, and graph workers concurrently. ChromaDB provides centralized vector retrieval using cosine distance and HNSW, while Neo4j stores entity relationships and exposes BFS-based relationship expansion. The evidence is ranked and passed to Gemini for cited synthesis. The same stateless worker image can be horizontally replicated with ECS Fargate, and MCP exposes the research tools to external AI clients.”

## Tests and quality

```bash
pytest -q
ruff check .
black --check .
```

GitHub Actions runs the test suite on pushes and pull requests to `main`.
