# Contributing

1. Create a branch from `main`.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run `pytest -q`, `ruff check .`, and `black --check .`.
4. Add tests for behavior changes.
5. Keep workers stateless: shared state belongs in ChromaDB/Neo4j/object storage, not process memory.
6. Open a pull request describing the architecture impact and evaluation results.

Security issues should not be reported publicly; contact the repository owner privately.
