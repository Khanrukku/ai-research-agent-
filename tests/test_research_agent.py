import pytest

from app.agents.research_agent import ResearchAgent


class DummyWorker:
    def __init__(self, result):
        self.result = result

    async def run(self, goal, max_sources):
        return self.result


class DummyResult:
    def __init__(self, evidence=None, metadata=None):
        self.evidence = evidence or []
        self.metadata = metadata or {}


@pytest.mark.asyncio
async def test_research_agent_runs_workers_concurrently():
    result_1 = DummyResult(
        evidence=[
            {
                "title": "Paper A",
                "text": "machine learning continual learning",
                "score": 0.8,
                "url": "https://example.com/a",
            }
        ]
    )

    result_2 = DummyResult(
        evidence=[
            {
                "title": "Paper B",
                "text": "catastrophic forgetting neural networks",
                "score": 0.7,
                "url": "https://example.com/b",
            }
        ]
    )

    workers = [
        DummyWorker(result_1),
        DummyWorker(result_2),
    ]

    agent = ResearchAgent(workers=workers)

    result = await agent.run("continual learning catastrophic forgetting")

    assert result is not None


def test_rank_returns_evidence_in_descending_order():
    agent = ResearchAgent(workers=[])

    evidence = [
        {
            "title": "Low",
            "text": "unrelated content",
            "score": 0.2,
        },
        {
            "title": "High",
            "text": "continual learning catastrophic forgetting",
            "score": 0.9,
        },
    ]

    ranked = agent._rank(
        evidence,
        "continual learning catastrophic forgetting",
    )

    assert len(ranked) == 2
    assert ranked[0]["title"] == "High"


def test_rank_preserves_all_evidence():
    agent = ResearchAgent(workers=[])

    evidence = [
        {"title": "A", "text": "AI systems", "score": 0.5},
        {"title": "B", "text": "machine learning", "score": 0.6},
        {"title": "C", "text": "deep learning", "score": 0.4},
    ]

    ranked = agent._rank(evidence, "machine learning")

    assert len(ranked) == len(evidence)


def test_rank_handles_empty_evidence():
    agent = ResearchAgent(workers=[])

    ranked = agent._rank([], "test query")

    assert ranked == []


def test_rank_handles_missing_score():
    agent = ResearchAgent(workers=[])

    evidence = [
        {
            "title": "No Score",
            "text": "machine learning research",
        }
    ]

    ranked = agent._rank(evidence, "machine learning")

    assert len(ranked) == 1


def test_rank_rewards_lexical_overlap():
    agent = ResearchAgent(workers=[])

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

    ranked = agent._rank(
        evidence,
        "continual learning catastrophic forgetting",
    )

    assert ranked[0]["title"] == "Relevant"
