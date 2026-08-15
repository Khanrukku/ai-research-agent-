from app.agents.research_agent import ResearchAgent, ResearchSource


def test_plan_depths():
    agent = ResearchAgent()
    assert agent._create_research_plan("RAG", "quick")["steps"][-1] == "synthesize"
    assert "cross_check" in agent._create_research_plan("RAG", "comprehensive")["steps"]


def test_invalid_depth_is_rejected():
    agent = ResearchAgent()
    try:
        agent._create_research_plan("RAG", "invalid")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "depth" in str(exc)


def test_tokenize_removes_stop_words():
    tokens = ResearchAgent._tokenize("What is the latest research on RAG?")
    assert "what" not in tokens
    assert "the" not in tokens
    assert "latest" not in tokens
    assert "rag" in tokens


def test_relevance_prioritizes_title_matches():
    score = ResearchAgent._relevance_score(
        {"rag", "retrieval"},
        "Retrieval Augmented Generation",
        "A general discussion about language models.",
    )
    assert score > 0.5


def test_ranker_adds_recency_signal():
    source = ResearchSource(
        source="arXiv",
        title="Retrieval Augmented Generation",
        content="retrieval augmented generation",
        url="https://arxiv.org/abs/2005.11401",
        relevance_score=0,
        timestamp="2026-01-01T00:00:00+00:00",
    )
    ranked = ResearchAgent()._rank_sources("RAG retrieval", [source])
    assert ranked[0].relevance_score > 0


def test_context_memory_is_limited():
    agent = ResearchAgent()
    for index in range(15):
        agent._update_context_memory(f"query {index}", "report", [f"url-{index}"])
    assert len(agent.get_context_memory()) == 10
