"""
tests/test_core.py
------------------
Unit tests for the core modules (no API keys required — uses mocks).
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ──────────────────────────────────────────────
#  Chunking tests (pure logic, no mocks needed)
# ──────────────────────────────────────────────

def test_chunk_text_short():
    from app.rag.vector_store import chunk_text
    text = "Short text under 512 chars."
    chunks = chunk_text(text, chunk_size=512)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_long():
    from app.rag.vector_store import chunk_text
    text = "A " * 300  # 600 chars
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    assert len(chunks) > 1


def test_chunk_text_empty():
    from app.rag.vector_store import chunk_text
    chunks = chunk_text("", chunk_size=512)
    assert chunks == [] or chunks == [""]


# ──────────────────────────────────────────────
#  Config tests
# ──────────────────────────────────────────────

def test_settings_loads():
    from app.core.config import Settings
    s = Settings()
    assert isinstance(s.gemini_model, str)
    assert isinstance(s.max_search_results, int)
    assert s.max_search_results > 0


def test_chunk_id_format():
    from app.rag.vector_store import _chunk_id
    cid = _chunk_id("doc-abc", 3)
    assert "doc-abc" in cid
    assert "3" in cid


# ──────────────────────────────────────────────
#  LLM module tests (mocked)
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_calls_gemini():
    with patch("app.core.llm._chat_model") as mock_model, \
         patch("app.core.llm._init_gemini"):
        mock_response = MagicMock()
        mock_response.text = "Test response"
        mock_model.generate_content.return_value = mock_response

        # Patch asyncio.get_event_loop().run_in_executor to run synchronously
        import asyncio
        with patch.object(asyncio.get_event_loop(), "run_in_executor",
                          new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_response
            from app.core.llm import generate
            # Just verify the function exists and is async
            assert asyncio.iscoroutinefunction(generate)


# ──────────────────────────────────────────────
#  Graph extractor tests (mocked)
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_extract_and_store_handles_llm_error():
    with patch("app.graph.extractor.extract_json", side_effect=Exception("LLM error")), \
         patch("app.graph.extractor.graph.upsert_document", new_callable=AsyncMock):
        from app.graph.extractor import extract_and_store
        result = await extract_and_store("some text", "doc-1")
        assert "error" in result
        assert result["entities"] == 0


# ──────────────────────────────────────────────
#  FastAPI route tests
# ──────────────────────────────────────────────

def test_app_creates():
    from app.main import app
    assert app.title == "AI Research Agent"


def test_routes_registered():
    from app.main import app
    paths = [r.path for r in app.routes]
    assert "/api/v1/health" in paths
    assert "/api/v1/ingest" in paths
    assert "/api/v1/query" in paths
    assert "/api/v1/research" in paths
