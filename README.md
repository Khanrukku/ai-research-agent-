# 🤖 AI Research Agent

[![CI](https://github.com/Khanrukku/ai-research-agent-/actions/workflows/CI.yml/badge.svg)](https://github.com/Khanrukku/ai-research-agent-/actions/workflows/CI.yml)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-ECS%20Fargate-232F3E?logo=amazonwebservices&logoColor=white)
![Terraform](https://img.shields.io/badge/Infrastructure-Terraform-844FBA?logo=terraform&logoColor=white)

An asynchronous **AI-powered research platform** that combines academic retrieval, semantic search, knowledge graphs, and LLM-based synthesis to investigate research questions and generate evidence-grounded responses.

The project explores how multiple specialized research components can operate concurrently and combine their findings into a unified research workflow.

---

## ✨ Why I Built This

Traditional AI applications often rely on a single retrieval or generation step.

This project explores a more structured research workflow where specialized components independently gather evidence from different sources before the results are ranked, verified, and synthesized.

The system currently combines:

- 📚 Academic research retrieval from arXiv
- 🔎 Semantic retrieval using ChromaDB
- 🕸️ Knowledge-graph exploration using Neo4j
- 🧠 Gemini-powered synthesis
- ⚡ Asynchronous worker orchestration
- 📊 Retrieval evaluation
- 🧪 Automated testing
- 🔄 GitHub Actions CI
- 🐳 Docker-based development
- ☁️ AWS infrastructure using Terraform

---

## 🏗️ System Architecture

```text
                         ┌─────────────────────┐
                         │    Research Query   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Research Agent    │
                         │    Orchestrator     │
                         └──────────┬──────────┘
                                    │
                       asyncio.gather(...)
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
       ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
       │   arXiv     │       │ Semantic    │       │ Knowledge   │
       │   Worker    │       │ RAG Worker  │       │ Graph Worker│
       └──────┬──────┘       └──────┬──────┘       └──────┬──────┘
              │                     │                     │
              ▼                     ▼                     ▼
        Academic Papers          ChromaDB               Neo4j
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Evidence Aggregation│
                         │     & Ranking       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Evidence Validation │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Gemini Synthesis    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Research Report   │
                         └─────────────────────┘
```

The workers are executed concurrently using Python's asynchronous programming capabilities, allowing independent retrieval operations to proceed without unnecessary sequential blocking.

---

## 🚀 Core Features

### 🔬 Multi-Source Research

The research agent gathers evidence through multiple specialized workers rather than relying on a single information source.

This allows the system to combine different retrieval strategies before synthesis.

### 📚 arXiv Research Retrieval

The arXiv worker retrieves academic papers and extracts information including:

- Paper title
- Abstract
- Publication information
- Source URL
- Relevance information

This provides an academic evidence source for research queries.

### 🔎 Semantic Retrieval with ChromaDB

The RAG component supports semantic retrieval from indexed documents.

The retrieval pipeline is designed around:

```text
Documents
    ↓
Text Chunking
    ↓
Embeddings
    ↓
ChromaDB
    ↓
Semantic Retrieval
    ↓
Relevant Evidence
```

### 🕸️ Knowledge Graph Exploration

Neo4j is used for graph-based information exploration.

The graph component supports entity relationships and neighborhood exploration, including **Breadth-First Search (BFS)** concepts for traversing connected information.

### ⚡ Asynchronous Orchestration

Independent research workers are executed concurrently using:

```python
await asyncio.gather(...)
```

This architecture allows multiple retrieval operations to run in parallel while isolating individual worker failures.

### 🧠 Evidence Ranking

Evidence returned by the research workers is aggregated and ranked before synthesis.

The current ranking strategy combines retrieval confidence with lexical relevance to the original research query.

### 🤖 LLM-Based Synthesis

Ranked evidence is passed to Gemini for synthesis into a structured research response.

The goal is to keep generation connected to evidence gathered by the research pipeline rather than relying entirely on unconstrained generation.

---

## 🧰 Tech Stack

### Backend

- Python
- FastAPI
- AsyncIO
- Pydantic

### AI & Retrieval

- Google Gemini
- Retrieval-Augmented Generation (RAG)
- ChromaDB
- Embeddings
- Semantic Search

### Knowledge Graph

- Neo4j
- Entity relationships
- Graph traversal
- Breadth-First Search concepts

### Research Sources

- arXiv API

### Testing & Quality

- Pytest
- pytest-asyncio
- Mocking of external dependencies
- GitHub Actions

### Cloud & DevOps

- Docker
- Docker Compose
- GitHub Actions
- AWS ECS Fargate
- Amazon ECR
- Amazon S3
- Amazon CloudWatch
- Terraform

---

## 📁 Project Structure

```text
ai-research-agent/
│
├── app/
│   ├── agents/
│   │   ├── base.py
│   │   └── research_agent.py
│   │
│   ├── api/
│   │
│   ├── core/
│   │   └── config.py
│   │
│   ├── graph/
│   │   └── extractor.py
│   │
│   ├── rag/
│   │   └── vector_store.py
│   │
│   ├── static/
│   │
│   └── main.py
│
├── tests/
│   ├── test_core.py
│   └── test_research_agent.py
│
├── infra/
│   └── aws/
│
├── .github/
│   └── workflows/
│       └── CI.yml
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

> The exact repository structure may evolve as the project is refactored and additional components are introduced.

---

## 🧪 Automated Testing

The project includes automated tests for core application behavior and research-agent functionality.

Current tests cover areas such as:

- Text chunking
- Chunk validation
- Configuration loading
- Chunk ID generation
- LLM failure handling
- FastAPI application creation
- API route registration
- Evidence ranking
- Empty evidence handling
- Missing-score handling
- Lexical relevance
- Evidence verification
- Context formatting
- Research input validation
- Multi-worker research orchestration

External services are mocked where appropriate so that unit tests do not require live Gemini, arXiv, ChromaDB, or Neo4j access for basic logic validation.

Run the tests with:

```bash
pytest -q
```

---

## 🔄 Continuous Integration

The repository uses **GitHub Actions** to automatically execute the test suite when code changes are pushed or submitted through pull requests.

Current CI pipeline:

```text
Push / Pull Request
        ↓
Checkout Repository
        ↓
Configure Python
        ↓
Install Dependencies
        ↓
Run Pytest
        ↓
Pass / Fail
```

The live CI status is displayed at the top of this README.

---

## 📊 Retrieval Evaluation

The project includes retrieval-evaluation tooling for measuring information-retrieval quality.

Metrics explored include:

### Precision@K

Measures how many of the top-K retrieved documents are relevant.

### Recall@K

Measures how much of the relevant information is successfully retrieved.

### Mean Reciprocal Rank (MRR)

Measures how highly the first relevant result appears in the ranking.

These evaluation utilities are intended to support reproducible experimentation rather than unsupported performance claims.

---

## ⚡ Concurrency Benchmarking

The repository also contains benchmarking logic for comparing sequential and concurrent research-worker execution.

Conceptually:

```text
Sequential

Worker A ──────► Worker B ──────► Worker C


Concurrent

Worker A ───────────►
Worker B ───────────►     → Aggregate Results
Worker C ───────────►
```

Any numerical performance results should be reported together with the benchmark environment and methodology rather than treated as universal system guarantees.

---

## ☁️ AWS Deployment Architecture

Terraform infrastructure is included to explore deployment of the application using AWS services such as:

- ECS Fargate
- ECR
- CloudWatch
- S3
- IAM
- Security Groups

A simplified deployment model is:

```text
                     Internet / Client
                            │
                            ▼
                    ┌───────────────┐
                    │  FastAPI App  │
                    │  ECS Service  │
                    └───────┬───────┘
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
        ┌──────────────┐        ┌──────────────┐
        │ ECS Replica  │        │ ECS Replica  │
        │      #1      │        │      #2      │
        └──────┬───────┘        └──────┬───────┘
               │                       │
               └───────────┬───────────┘
                           │
              ┌────────────┼─────────────┐
              │            │             │
              ▼            ▼             ▼
          ChromaDB       Neo4j       Gemini API
```

The current infrastructure should be treated as a **cloud deployment demonstration**, not as a claim of a fully production-hardened environment.

Future infrastructure improvements include secret management, HTTPS/load balancing, private networking, health checks, and autoscaling policies.

---

## ⚙️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Khanrukku/ai-research-agent-.git
cd ai-research-agent-
```

### 2. Create a virtual environment

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create the required environment configuration for services used by the project.

Depending on the components being run, this may include credentials/configuration for:

```text
Gemini
Neo4j
ChromaDB
AWS
```

Never commit API keys, passwords, or cloud credentials to the repository.

### 5. Run the application

Depending on the current application configuration:

```bash
uvicorn app.main:app --reload
```

### 6. Open API documentation

After starting the application locally, FastAPI exposes interactive API documentation through the application's `/docs` route.

---

## 🐳 Running with Docker

Build the application image:

```bash
docker build -t ai-research-agent .
```

Run the container:

```bash
docker run -p 8000:8000 ai-research-agent
```

For the multi-service development environment, use Docker Compose according to the repository configuration.

---

## 🛣️ Engineering Roadmap

The project is actively being improved.

Planned engineering work includes:

- [x] FastAPI application architecture
- [x] Multi-worker research orchestration
- [x] arXiv retrieval
- [x] Semantic retrieval
- [x] Neo4j graph integration
- [x] Gemini synthesis
- [x] Automated Pytest suite
- [x] GitHub Actions CI
- [x] Docker support
- [x] Terraform-based AWS infrastructure
- [ ] Expand retrieval evaluation dataset
- [ ] Add structured application logging
- [ ] Improve worker-level failure reporting
- [ ] Add API integration tests
- [ ] Add end-to-end research workflow tests
- [ ] Add benchmark reports
- [ ] Add architecture diagrams
- [ ] Improve observability
- [ ] Introduce secure cloud secret management
- [ ] Add load balancing and health checks
- [ ] Explore autoscaling strategies

---

## 🧠 Engineering Concepts Explored

This project gives me hands-on experience with:

- Asynchronous programming
- Concurrent task execution
- Multi-agent orchestration
- Retrieval-Augmented Generation
- Information retrieval
- Semantic search
- Knowledge graphs
- Graph traversal
- Evidence ranking
- API development
- Failure handling
- Unit testing
- Dependency mocking
- Continuous Integration
- Containerization
- Infrastructure as Code
- Cloud deployment architecture

---

## 🎯 Current Focus

I'm currently improving this project with an emphasis on:

**Correctness → Testing → Reliability → Evaluation → Observability → Scalability**

Rather than adding unsupported performance claims, the goal is to progressively benchmark and document the system so that engineering decisions and results can be reproduced.

---

## 👩‍💻 Author

**Rukaiya Khan**

MCA Student | Software Engineering | Backend & Distributed Systems | Cloud | AI/ML

[![GitHub](https://img.shields.io/badge/GitHub-Khanrukku-181717?logo=github)](https://github.com/Khanrukku)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Rukaiya%20Khan-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rukaiyakhan/)

---

## 🤝 Contributions

Suggestions, issues, and technical feedback are welcome.

If you find a bug or have an idea for improving the architecture, feel free to open an issue or submit a pull request.

---

⭐ If you find this project useful or interesting, consider starring the repository.
