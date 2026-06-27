<img width="1920" height="1080" alt="Screenshot 2026-06-27 230435" src="https://github.com/user-attachments/assets/b58c9a7c-ae94-4562-83bf-796f44150972" />
<img width="1920" height="1080" alt="Screenshot 2026-06-27 230425" src="https://github.com/user-attachments/assets/d116ac2e-0d57-4736-88b3-3beda812ea77" />
<img width="1920" height="1080" alt="Screenshot 2026-06-27 230417" src="https://github.com/user-attachments/assets/08c876c4-8b22-4f27-ac7d-34865d77401e" />
<img width="1920" height="1080" alt="Screenshot 2026-06-27 230435" src="https://github.com/user-attachments/assets/c848ed29-710e-4890-b818-9e9b04953a0e" />
<img width="1920" height="1080" alt="Screenshot 2026-06-27 230425" src="https://github.com/user-attachments/assets/588aa476-b514-41ac-bab0-abb3e9b53eab" />
<img width="1920" height="1080" alt="Screenshot 2026-06-27 230417" src="https://github.com/user-attachments/assets/670c2240-6153-49c1-8fbb-a6bfa400feef" />
# 🔬 AI Research Agent

> **Multi-step agentic RAG system** combining vector search, knowledge graphs, and LLM orchestration — built for the Google SWE Internship portfolio.

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![Gemini](https://img.shields.io/badge/LLM-Gemini_1.5-orange)](https://aistudio.google.com)
[![Neo4j](https://img.shields.io/badge/Graph-Neo4j_Aura-teal)](https://neo4j.com/cloud/aura)
[![ChromaDB](https://img.shields.io/badge/Vector-ChromaDB-purple)](https://trychroma.com)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      FastAPI REST API                        │
│   /ingest    /query    /research    /graph/*                 │
└──────────────────┬───────────────────────────────────────────┘
                   │
       ┌───────────▼────────────┐
       │    Research Agent      │  ← ReAct loop (Reason→Act→Observe)
       │  (MCP-style tool use)  │     up to 8 autonomous steps
       └───┬───────────┬────────┘
           │           │
    ┌──────▼──┐   ┌────▼──────────┐
    │   RAG   │   │ Knowledge     │
    │ Pipeline│   │    Graph      │
    └──┬──────┘   └────┬──────────┘
       │               │
  ┌────▼────┐    ┌─────▼──────┐
  │ChromaDB │    │   Neo4j    │
  │(vectors)│    │  (Aura)    │
  └────┬────┘    └─────┬──────┘
       │               │
       └───────┬───────┘
               │
        ┌──────▼──────┐
        │  Gemini API │
        │  (LLM +     │
        │  Embeddings)│
        └─────────────┘
```

## Key Concepts Demonstrated

| Concept | Implementation |
|---|---|
| **RAG** | ChromaDB vector store + Gemini embeddings → grounded answers |
| **Knowledge Graphs** | Neo4j Aura — entities, relations, graph traversal |
| **LLM Orchestration** | Gemini 1.5 Flash for generation + `embedding-001` for vectors |
| **MCP-style Tool Use** | Agent exposes a tool manifest; LLM decides which tools to call |
| **Agentic Workflows** | ReAct loop (Reason→Act→Observe) with up to 8 autonomous steps |
| **Semantic Search** | Cosine similarity over 768-dim Gemini embeddings |

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/yourusername/ai-research-agent
cd ai-research-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up credentials

```bash
cp .env.example .env
# Edit .env — add your GEMINI_API_KEY and Neo4j credentials
```

**Gemini API key** (free): [aistudio.google.com](https://aistudio.google.com) → Get API Key

**Neo4j Aura** (free tier): [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura) → Create Instance

### 3. Start the server

```bash
uvicorn app.main:app --reload --port 8000
```

API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Load sample data

```bash
python scripts/ingest_samples.py
```

This seeds 5 AI research papers (Transformer, GPT-4, Gemini, Constitutional AI, RAG).

---

## API Reference

### `POST /api/v1/ingest`
Ingest a document into both vector store and knowledge graph.

```json
{
  "title": "Attention Is All You Need",
  "source": "https://arxiv.org/abs/1706.03762",
  "text": "The Transformer architecture..."
}
```

Response:
```json
{
  "doc_id": "doc-a3f2b1c4",
  "chunks": 6,
  "entities": 12,
  "relations": 8
}
```

---

### `POST /api/v1/query`
Single-turn RAG query (fast, no agentic loop).

```json
{
  "question": "What is Retrieval-Augmented Generation?",
  "n_results": 5,
  "include_graph": true
}
```

---

### `POST /api/v1/research` ⭐
**Multi-step agentic research** — the flagship endpoint.

The agent autonomously:
1. Chooses which tools to call (vector search, graph lookup)
2. Calls them and observes results
3. Iterates until it has enough context
4. Synthesises a final comprehensive answer

```json
{
  "goal": "Compare the training approaches of GPT-4 and Gemini, and explain how Constitutional AI differs from RLHF"
}
```

Response includes the full reasoning trace:
```json
{
  "goal": "...",
  "answer": "## Comparison of GPT-4 and Gemini...",
  "steps": [
    {
      "step": 1,
      "thought": "I should first search for GPT-4 training details",
      "action": "vector_search",
      "action_input": "GPT-4 training RLHF",
      "observation": "[doc-a3f2] GPT-4 was trained using RLHF...",
      "duration_ms": 312.4
    },
    ...
  ],
  "total_duration_ms": 4821.0,
  "sources": ["doc-a3f2", "doc-c9d1"]
}
```

---

### `GET /api/v1/graph/entities?q=Gemini`
Search entities in the knowledge graph.

### `GET /api/v1/health`
Check system status (vector store + graph stats).

---

## How the Agent Works (Interview Answer)

The `research` endpoint implements a **ReAct** (Reason + Act) agent loop:

```
LOOP (up to 8 steps):
  1. Give LLM: goal + tool manifest + history so far
  2. LLM responds: { thought, action, action_input }
  3. Execute tool → get observation
  4. Append observation to history
  5. If action == "synthesise": break and return answer

Tools available:
  vector_search(query)    → semantic search in ChromaDB
  graph_lookup(entity)    → Neo4j entity neighbourhood
  graph_documents(entity) → documents mentioning entity
  synthesise(context)     → terminal: generate final answer
```

This is structurally identical to how **MCP (Model Context Protocol)** works: a host exposes tools, the model calls them, the host executes and returns results.

---

## Project Structure

```
ai-research-agent/
├── app/
│   ├── main.py              # FastAPI app factory
│   ├── api/
│   │   └── routes.py        # All REST endpoints
│   ├── core/
│   │   ├── config.py        # Pydantic settings (reads .env)
│   │   ├── llm.py           # Gemini wrapper (generate + embed)
│   │   └── logging.py       # Rich structured logging
│   ├── rag/
│   │   ├── vector_store.py  # ChromaDB: chunk, embed, search
│   │   └── retriever.py     # RAG pipeline (vector + graph)
│   ├── graph/
│   │   ├── client.py        # Neo4j Cypher queries
│   │   └── extractor.py     # Entity/relation extraction → graph
│   └── agents/
│       └── research_agent.py # Multi-step ReAct agent loop
├── scripts/
│   └── ingest_samples.py    # Seeds 5 AI research papers
├── tests/
│   └── test_core.py         # Unit tests (mocked)
├── .env.example             # Credential template
├── .gitignore               # Excludes .env, data/, __pycache__
└── requirements.txt
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Tech Stack

- **LLM**: Google Gemini 1.5 Flash (generation) + `embedding-001` (768-dim vectors)
- **Vector DB**: ChromaDB (local persistent, cosine similarity)
- **Graph DB**: Neo4j Aura (managed cloud, free tier)
- **API**: FastAPI + Uvicorn (async, OpenAPI docs auto-generated)
- **Resilience**: Tenacity retry with exponential back-off on all API calls
- **Config**: Pydantic-settings (12-factor app pattern)
- **Logging**: Rich structured console logging

---

*Built as portfolio project for Google SWE Internship application.*
