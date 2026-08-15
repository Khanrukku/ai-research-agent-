import pytest

from research_agent import ResearchAgent, ResearchResult
 

def test_empty_query_is_rejected():
    agent = ResearchAgent()

    with pytest.raises(ValueError, match="cannot be empty"):
        agent.research("")


def test_invalid_depth_is_rejected():
    agent = ResearchAgent()

    with pytest.raises(ValueError, match="depth"):
        agent._create_research_plan(
            query="RAG",
            depth="invalid",
        )


def test_invalid_output_format_is_rejected():
    agent = ResearchAgent()

    with pytest.raises(ValueError, match="output_format"):
        agent.research(
            query="RAG",
            output_format="pdf",
        )


def test_tokenize_removes_common_stop_words():
    tokens = ResearchAgent._tokenize(
        "What is the latest research on RAG?"
    )

    assert "what" not in tokens
    assert "the" not in tokens
    assert "latest" not in tokens
    assert "research" in tokens
    assert "rag" in tokens


def test_relevance_score_prioritizes_title_matches():
    score = ResearchAgent._relevance_score(
        query_terms={"rag", "retrieval"},
        title="Retrieval Augmented Generation",
        content="A general discussion about language models.",
    )

    assert score > 0


def test_rag_context_contains_sources():
    sources = [
        ResearchResult(
            source="arXiv",
            title="RAG Research",
            content="Retrieval augmented generation.",
            url="https://arxiv.org/abs/example",
            relevance_score=0.9,
            timestamp="2026-01-01T00:00:00Z",
            metadata={"type": "academic_paper"},
        )
    ]

    context = ResearchAgent._build_rag_context(
        sources
    )

    assert "RAG Research" in context
    assert "https://arxiv.org/abs/example" in context
    assert "SOURCE 1" in context


def test_context_memory_is_limited():
    agent = ResearchAgent()

    for index in range(15):
        agent._update_context_memory(
            query=f"query {index}",
            report="test report",
        )

    assert len(agent.get_context_memory()) == 10
