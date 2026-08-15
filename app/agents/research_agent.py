"""
AI Research Agent
-----------------
A lightweight research agent that:

1. Creates a research plan.
2. Retrieves real papers from arXiv.
3. Ranks sources using lexical relevance.
4. Builds a retrieval-augmented context.
5. Uses an LLM to synthesize a cited report.
6. Maintains lightweight session memory.
"""

from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

ARXIV_API_URL = "https://export.arxiv.org/api/query"

DEFAULT_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4.1-mini",
)


@dataclass
class ResearchQuery:
    """Research query and execution settings."""

    query: str
    depth: str = "standard"
    max_sources: int = 5
    output_format: str = "markdown"


@dataclass
class ResearchResult:
    """A retrieved research source."""

    source: str
    title: str
    content: str
    url: str
    relevance_score: float
    timestamp: str
    metadata: dict[str, Any]


class ResearchAgent:
    """Research agent for retrieval and evidence-grounded synthesis."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_sources: int = 5,
        timeout: int = 20,
    ) -> None:

        if max_sources < 1:
            raise ValueError("max_sources must be at least 1.")

        self.model = model
        self.max_sources = max_sources
        self.timeout = timeout
        self.context_memory: list[dict[str, Any]] = []

        api_key = os.getenv("OPENAI_API_KEY")

        self.client: OpenAI | None = (
            OpenAI(api_key=api_key)
            if api_key
            else None
        )

    def research(
        self,
        query: str,
        output_format: str = "markdown",
        depth: str = "standard",
    ) -> str:
        """Execute the complete research workflow."""

        query = query.strip()

        if not query:
            raise ValueError("Research query cannot be empty.")

        if output_format not in {"markdown", "json"}:
            raise ValueError(
                "output_format must be 'markdown' or 'json'."
            )

        plan = self._create_research_plan(
            query=query,
            depth=depth,
        )

        sources = self._gather_information(
            query=query,
            plan=plan,
        )

        if not sources:
            raise RuntimeError(
                "No research sources were retrieved."
            )

        report = self._synthesize_report(
            query=query,
            sources=sources,
            plan=plan,
        )

        self._update_context_memory(
            query=query,
            report=report,
        )

        return self._format_output(
            report=report,
            output_format=output_format,
        )

    def _create_research_plan(
        self,
        query: str,
        depth: str,
    ) -> dict[str, Any]:

        steps_by_depth = {
            "quick": [
                "retrieve_sources",
                "rank_sources",
                "synthesize",
            ],
            "standard": [
                "retrieve_sources",
                "rank_sources",
                "build_rag_context",
                "synthesize",
            ],
            "comprehensive": [
                "retrieve_sources",
                "rank_sources",
                "build_rag_context",
                "cross_check_sources",
                "synthesize",
            ],
        }

        if depth not in steps_by_depth:
            raise ValueError(
                "depth must be one of: quick, standard, comprehensive"
            )

        return {
            "query": query,
            "depth": depth,
            "steps": steps_by_depth[depth],
            "max_sources": self.max_sources,
            "timestamp": self._timestamp(),
        }

    def _gather_information(
        self,
        query: str,
        plan: dict[str, Any],
    ) -> list[ResearchResult]:

        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": min(
                max(self.max_sources * 2, 5),
                20,
            ),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        response = requests.get(
            ARXIV_API_URL,
            params=params,
            timeout=self.timeout,
            headers={
                "User-Agent": "ai-research-agent/1.0",
            },
        )

        response.raise_for_status()

        root = ET.fromstring(response.text)

        namespace = {
            "atom": "http://www.w3.org/2005/Atom",
        }

        query_terms = self._tokenize(query)

        results: list[ResearchResult] = []

        for entry in root.findall("atom:entry", namespace):

            title = self._clean_text(
                entry.findtext(
                    "atom:title",
                    default="",
                    namespaces=namespace,
                )
            )

            summary = self._clean_text(
                entry.findtext(
                    "atom:summary",
                    default="",
                    namespaces=namespace,
                )
            )

            published = entry.findtext(
                "atom:published",
                default="",
                namespaces=namespace,
            )

            paper_id = entry.findtext(
                "atom:id",
                default="",
                namespaces=namespace,
            )

            if not title or not summary or not paper_id:
                continue

            score = self._relevance_score(
                query_terms=query_terms,
                title=title,
                content=summary,
            )

            results.append(
                ResearchResult(
                    source="arXiv",
                    title=title,
                    content=summary,
                    url=paper_id,
                    relevance_score=score,
                    timestamp=published or self._timestamp(),
                    metadata={
                        "type": "academic_paper",
                        "verified": True,
                    },
                )
            )

        results.sort(
            key=lambda result: result.relevance_score,
            reverse=True,
        )

        return results[: self.max_sources]

    @staticmethod
    def _relevance_score(
        query_terms: set[str],
        title: str,
        content: str,
    ) -> float:

        if not query_terms:
            return 0.0

        title_terms = ResearchAgent._tokenize(title)
        content_terms = ResearchAgent._tokenize(content)

        title_overlap = len(
            query_terms.intersection(title_terms)
        ) / len(query_terms)

        content_overlap = len(
            query_terms.intersection(content_terms)
        ) / len(query_terms)

        score = (
            0.7 * title_overlap
            + 0.3 * content_overlap
        )

        return round(
            min(score, 1.0),
            4,
        )

    @staticmethod
    def _tokenize(text: str) -> set[str]:

        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "of",
            "to",
            "in",
            "for",
            "on",
            "with",
            "is",
            "are",
            "what",
            "how",
            "why",
            "latest",
        }

        tokens = re.findall(
            r"[a-zA-Z0-9]+",
            text.lower(),
        )

        return {
            token
            for token in tokens
            if len(token) > 2
            and token not in stop_words
        }

    @staticmethod
    def _build_rag_context(
        sources: list[ResearchResult],
    ) -> str:

        context_parts = []

        for index, source in enumerate(
            sources,
            start=1,
        ):
            context_parts.append(
                f"""
SOURCE {index}
Title: {source.title}
URL: {source.url}
Relevance: {source.relevance_score:.4f}

Abstract:
{source.content}
""".strip()
            )

        return "\n\n".join(context_parts)

    def _synthesize_report(
        self,
        query: str,
        sources: list[ResearchResult],
        plan: dict[str, Any],
    ) -> str:

        context = self._build_rag_context(sources)

        if self.client is None:
            return self._fallback_report(
                query=query,
                sources=sources,
                plan=plan,
            )

        prompt = f"""
You are a research analyst.

Answer the user's research question using ONLY the supplied sources.

Research question:
{query}

Retrieved sources:
{context}

Requirements:

1. Write a concise executive summary.
2. Identify the most important findings.
3. Distinguish evidence from interpretation.
4. Do not invent facts that are not supported by the sources.
5. Include inline citations using [Source 1], [Source 2], etc.
6. Include a Sources section containing source titles and URLs.
7. If the sources are insufficient to answer a claim, explicitly say so.
8. Do not fabricate performance metrics.
9. Do not claim causation unless supported by the sources.

Return Markdown.
""".strip()

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        report = response.output_text.strip()

        if not report:
            raise RuntimeError(
                "The language model returned an empty report."
            )

        return report

    def _fallback_report(
        self,
        query: str,
        sources: list[ResearchResult],
        plan: dict[str, Any],
    ) -> str:

        lines = [
            f"# Research Report: {query}",
            "",
            "## Retrieved Evidence",
            "",
            (
                "No OPENAI_API_KEY was configured, so the agent "
                "returned retrieved evidence without LLM synthesis."
            ),
            "",
        ]

        for index, source in enumerate(
            sources,
            start=1,
        ):
            lines.extend(
                [
                    f"### Source {index}: {source.title}",
                    "",
                    source.content,
                    "",
                    f"**URL:** {source.url}",
                    "",
                    (
                        f"**Relevance score:** "
                        f"{source.relevance_score:.4f}"
                    ),
                    "",
                ]
            )

        return "\n".join(lines)

    @staticmethod
    def _format_output(
        report: str,
        output_format: str,
    ) -> str:

        if output_format == "markdown":
            return report

        if output_format == "json":
            return json.dumps(
                {
                    "report": report,
                    "generated_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "format": "json",
                },
                indent=2,
            )

        raise ValueError(
            "Unsupported output format."
        )

    def _update_context_memory(
        self,
        query: str,
        report: str,
    ) -> None:

        self.context_memory.append(
            {
                "query": query,
                "timestamp": self._timestamp(),
                "summary": report[:200],
                "report_length": len(report),
            }
        )

        self.context_memory = self.context_memory[-10:]

    def get_context_memory(self) -> list[dict[str, Any]]:
        """Return recent research session memory."""

        return list(self.context_memory)

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join(text.split())

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()


def main() -> None:
    """Run a sample research query."""

    agent = ResearchAgent(
        model=DEFAULT_MODEL,
        max_sources=5,
    )

    query = (
        "retrieval augmented generation "
        "large language models"
    )

    report = agent.research(
        query=query,
        depth="standard",
        output_format="markdown",
    )

    filename = (
        "research_report_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )

    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)

    print(report)
    print(f"\nReport saved to: {filename}")


if __name__ == "__main__":
    main()
