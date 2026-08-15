import pytest
from unittest.mock import AsyncMock, patch


def test_chunk_text_short_and_empty():
    from app.rag.vector_store import chunk_text
    assert chunk_text("Short text") == ["Short text"]
    assert chunk_text("") == []


def test_chunk_text_long_has_overlap():
    from app.rag.vector_store import chunk_text
    text = "A " * 500
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    assert all(chunks)


def test_invalid_chunking_is_rejected():
    from app.rag.vector_store import chunk_text
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=10, overlap=10)


def test_settings_loads():
    from app.core.config import Settings
    settings = Settings()
    assert settings.gemini_model
    assert settings.embedding_model
    assert settings.max_search_results > 0


def test_chunk_id_format():
    from app.rag.vector_store import _chunk_id
    assert _chunk_id("doc-abc", 3) == "doc-abc::chunk::3"


@pytest.mark.asyncio
async def test_extract_and_store_handles_llm_error():
    with patch("app.graph.extractor.extract_json", new_callable=AsyncMock, side_effect=Exception("LLM error")):
        from app.graph.extractor import extract_and_store
        result = await extract_and_store("some text", "doc-1")
    assert result["entities"] == 0
    assert "error" in result


def test_app_creates():
    from app.main import app
    assert app.title == "AI Research Agent"
    assert app.version == "2.0.0"


def test_routes_registered():
    from app.main import app
    paths = {r.path for r in app.routes}
    assert "/api/v1/health" in paths
    assert "/api/v1/ingest" in paths
    assert "/api/v1/query" in paths
    assert "/api/v1/research" in paths
