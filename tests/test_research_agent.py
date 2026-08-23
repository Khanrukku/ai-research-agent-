import pytest
from unittest.mock import AsyncMock, patch

from app.agents.base import AgentResult
from app.agents.research_agent import ResearchAgent


class DummyWorker:
    def __init__(self, result):
        self.result = result

    async def run(self, goal, max_sources):
        return self.result


@pytest.mark.asyncio
async def test_research_agent_runs_workers_concurrently():
    result_1 = AgentResult(
        name="worker_one",
        observation="Retrieved first source",
        evidence=[
            {
                "source": "test",
                "title": "Paper A",
                "text": "machine learning continual learning",
                "score": 0.8,
                "url": "https://example.com/a",
            }
        ],
        duration_ms=10.0,
    )

    result_2 = AgentResult(
        name="worker_two",
        observation="Retrieved second source",
        evidence=[
            {
                "source": "test",
                "title": "Paper B",
                "text": "catastrophic forgetting neural networks",
                "score": 0.7,
                "url": "https://example.com/b",
            }
        ],
        duration_ms=12.0,
    )

    agent = ResearchAgent(max_sources=5)

    # Replace the real external workers with deterministic test workers.
    agent.workers = (
        DummyWorker(result_1),
        DummyWorker(result_2),
    )

    with patch.object(
        agent,
        "_synthesize",
        new=AsyncMock(return_value="Generated research report"),
    ):
        result = await agent.research(
            "continual learning catastrophic forgetting"
        )

    assert result.goal == "continual learning catastrophic forgetting"
    assert result.answer == "Generated research report"
    assert len(result.steps) == 3
    assert "https://example.com/a" in result.sources
    assert "https://example.com/b" in result.sources


def test_rank_returns_evidence_in_descending_order():
    evidence = [
        {
            "title": "Low",
            "text": "unrelated database content",
            "score": 0.2,
        },
        {
            "title": "High",
            "text": "continual learning catastrophic forgetting",
            "score": 0.9,
        },
    ]

    ranked = ResearchAgent._rank(
        evidence,
        "continual learning catastrophic forgetting",
    )

    assert len(ranked) == 2
    assert ranked[0]["title"] == "High"
    assert ranked[0]["final_score"] >= ranked[1]["final_score"]


def test_rank_preserves_all_evidence():
    evidence = [
        {"title": "A", "text": "AI systems", "score": 0.5},
        {"title": "B", "text": "machine learning", "score": 0.6},
        {"title": "C", "text": "deep learning", "score": 0.4},
    ]

    ranked = ResearchAgent._rank(
        evidence,
        "machine learning",
    )

    assert len(ranked) == len(evidence)


def test_rank_handles_empty_evidence():
    ranked = ResearchAgent._rank([], "test query")

    assert ranked == []


def test_rank_handles_missing_score():
    evidence = [
        {
            "title": "No Score",
            "text": "machine learning research",
        }
    ]

    ranked = ResearchAgent._rank(
        evidence,
        "machine learning",
    )

    assert len(ranked) == 1
    assert "final_score" in ranked[0]


def test_rank_rewards_lexical_overlap():
    evidence = [
        {
            "title": "Relevant",
            "text": "continual learning catastrophic forgetting",
            "score": 0.5,
        },
        {
            "title": "Less Relevant",
            "text": "database indexing sql query",
            "score": 0.5,
        },
    ]

    ranked = ResearchAgent._rank(
        evidence,
        "continual learning catastrophic forgetting",
    )

    assert ranked[0]["title"] == "Relevant"


def test_verify_reports_evidence_and_source_counts():
    evidence = [
        {
            "source": "arXiv",
            "title": "Paper A",
        },
        {
            "source": "vector",
            "title": "Document B",
        },
        {
            "source": "arXiv",
            "title": "Paper C",
        },
    ]

    result = ResearchAgent._verify(evidence)

    assert "3 items" in result
    assert "2 source types" in result


def test_context_formats_evidence():
    evidence = [
        {
            "source": "arXiv",
            "title": "Continual Learning Paper",
            "text": "Research about catastrophic forgetting.",
            "url": "https://example.com/paper",
            "final_score": 0.91,
        }
    ]

    context = ResearchAgent._context(evidence)

    assert "Continual Learning Paper" in context
    assert "catastrophic forgetting" in context
    assert "https://example.com/paper" in context


@pytest.mark.asyncio
async def test_invalid_research_goal_is_rejected():
    agent = ResearchAgent()

    with pytest.raises(ValueError):
        await agent.research("")


@pytest.mark.asyncio
async def test_invalid_depth_is_rejected():
    agent = ResearchAgent()

    with pytest.raises(ValueError):
        await agent.research(
            "continual learning",
            depth="invalid",
        )
