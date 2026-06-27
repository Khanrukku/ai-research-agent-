"""
app/agents/research_agent.py
-----------------------------
Multi-step agentic research loop implementing a ReAct-style
(Reason → Act → Observe) workflow with MCP-inspired tool definitions.

The agent autonomously decides which tools to call, calls them,
observes the results, then either calls more tools or writes a
final synthesis.

Tools available to the agent:
  - vector_search(query)       → semantic search in ChromaDB
  - graph_lookup(entity)       → entity neighbourhood in Neo4j
  - graph_documents(entity)    → documents mentioning an entity
  - synthesise(context, goal)  → final answer generation (terminal)

This mirrors how MCP (Model Context Protocol) works:
  a host (this agent loop) exposes a tool manifest to the LLM,
  the LLM replies with tool calls, the host executes them and feeds
  results back — repeating until the model emits a final answer.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.llm import extract_json, generate
from app.core.logging import get_logger
from app.graph import client as graph
from app.rag.vector_store import semantic_search

logger = get_logger(__name__)

MAX_STEPS = 8   # safety limit

# ──────────────────────────────────────────────
#  Tool manifest (MCP-inspired)
# ──────────────────────────────────────────────

TOOL_MANIFEST = """
You have access to the following research tools. Call them by responding
with a JSON object:
{
  "thought": "your internal reasoning",
  "action": "<tool_name>",
  "action_input": "<string argument>"
}

Tools:
  vector_search(query: str)
      Semantic search across ingested research documents.
      Use for: broad topic queries, finding relevant passages.

  graph_lookup(entity: str)
      Retrieve the knowledge-graph neighbourhood of a named entity.
      Use for: understanding relationships, connections, influence.

  graph_documents(entity: str)
      Find all documents that mention a given entity.
      Use for: tracing where an entity appears in the corpus.

  synthesise(context: str)
      Generate the final comprehensive answer.
      ONLY call this when you have gathered enough information.
      Pass ALL collected evidence as the context string.
      This terminates the research loop.

Rules:
  - Think step by step before every action.
  - Do not repeat the same tool call with the same input.
  - Call synthesise when you have at least 2-3 pieces of evidence.
  - If a tool returns no results, try a different query.
"""

_STEP_SYSTEM = (
    "You are a methodical AI research agent. "
    "Follow the ReAct loop: Reason, then Act using one tool at a time. "
    "Always respond with valid JSON only."
)

_SYNTH_SYSTEM = """
You are an expert research synthesiser.
Given the user's research goal and the evidence collected, produce a
well-structured, comprehensive answer.
Use headers, bullet points, and cite document IDs where relevant.
"""


# ──────────────────────────────────────────────
#  Data structures
# ──────────────────────────────────────────────

@dataclass
class AgentStep:
    step: int
    thought: str
    action: str
    action_input: str
    observation: str
    duration_ms: float


@dataclass
class AgentResult:
    goal: str
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_duration_ms: float = 0.0
    sources: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────
#  Tool implementations
# ──────────────────────────────────────────────

async def _tool_vector_search(query: str) -> str:
    hits = await semantic_search(query, n_results=4)
    if not hits:
        return "No results found."
    parts = []
    for h in hits:
        parts.append(f"[{h['doc_id']}] score={h['score']}\n{h['text'][:400]}")
    return "\n\n---\n".join(parts)


async def _tool_graph_lookup(entity: str) -> str:
    entities = await graph.search_entities(entity, limit=8)
    if not entities:
        return f"No entities matching '{entity}' found in the knowledge graph."
    lines = [f"Entities related to '{entity}':"]
    for e in entities:
        lines.append(f"  [{e['type']}] {e['name']}: {e.get('description', '')}")
    return "\n".join(lines)


async def _tool_graph_documents(entity: str) -> str:
    docs = await graph.get_entity_documents(entity)
    if not docs:
        return f"No documents mention '{entity}'."
    lines = [f"Documents mentioning '{entity}':"]
    for d in docs:
        lines.append(f"  [{d['id']}] {d['title']} — {d.get('snippet', '')[:200]}")
    return "\n".join(lines)


TOOLS: dict[str, Any] = {
    "vector_search": _tool_vector_search,
    "graph_lookup": _tool_graph_lookup,
    "graph_documents": _tool_graph_documents,
}


# ──────────────────────────────────────────────
#  Agent loop
# ──────────────────────────────────────────────

async def run_research_agent(goal: str) -> AgentResult:
    """
    Execute the multi-step research agent for a given research goal.

    Args:
        goal: Natural language research question or task.

    Returns:
        AgentResult with the final answer and a full trace of steps.
    """
    t_start = time.time()
    steps: list[AgentStep] = []
    evidence_log: list[str] = []
    all_sources: list[str] = []

    history = (
        f"{TOOL_MANIFEST}\n\n"
        f"=== RESEARCH GOAL ===\n{goal}\n\n"
        "Begin your research. Respond with JSON."
    )

    logger.info("Agent starting: %s", goal[:80])

    for step_num in range(1, MAX_STEPS + 1):
        t_step = time.time()

        # ── Ask the LLM what to do next ──
        try:
            raw = await generate(history, system=_STEP_SYSTEM, temperature=0.0)
            # Strip markdown fences if present
            import re
            raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
            action_dict = json.loads(raw)
        except Exception as exc:
            logger.warning("Step %d parse error: %s", step_num, exc)
            break

        thought = action_dict.get("thought", "")
        action = action_dict.get("action", "").strip().lower()
        action_input = str(action_dict.get("action_input", ""))

        logger.info("Step %d | action=%s | input=%s", step_num, action, action_input[:60])

        # ── Terminal action: synthesise ──
        if action == "synthesise":
            synth_prompt = (
                f"Research goal: {goal}\n\n"
                f"Collected evidence:\n{''.join(evidence_log)}\n\n"
                "Write the final answer."
            )
            answer = await generate(synth_prompt, system=_SYNTH_SYSTEM, temperature=0.2)

            steps.append(AgentStep(
                step=step_num,
                thought=thought,
                action=action,
                action_input=action_input,
                observation="[Final answer generated]",
                duration_ms=round((time.time() - t_step) * 1000, 1),
            ))

            return AgentResult(
                goal=goal,
                answer=answer,
                steps=steps,
                total_duration_ms=round((time.time() - t_start) * 1000, 1),
                sources=list(set(all_sources)),
            )

        # ── Execute tool ──
        tool_fn = TOOLS.get(action)
        if tool_fn is None:
            observation = f"Unknown tool: {action}"
        else:
            try:
                observation = await tool_fn(action_input)
            except Exception as exc:
                observation = f"Tool error: {exc}"

        # Track sources (doc IDs look like "doc-..." or contain "::")
        import re
        source_ids = re.findall(r'\[([^\]]+)\]', observation)
        all_sources.extend(source_ids)

        # Log evidence
        evidence_log.append(
            f"\n[Step {step_num} — {action}({action_input!r})]\n{observation}\n"
        )

        # Append to history for the LLM
        history += (
            f"\n\n=== STEP {step_num} RESULT ===\n"
            f"Action: {action}({action_input!r})\n"
            f"Observation:\n{observation}\n\n"
            "Continue your research. Respond with JSON."
        )

        steps.append(AgentStep(
            step=step_num,
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation[:500],
            duration_ms=round((time.time() - t_step) * 1000, 1),
        ))

    # Fallback if we hit MAX_STEPS without synthesise
    logger.warning("Agent hit MAX_STEPS without synthesising — generating fallback")
    fallback_prompt = (
        f"Research goal: {goal}\n\n"
        f"Collected evidence:\n{''.join(evidence_log)}\n\n"
        "Based on the above, provide the best answer you can."
    )
    answer = await generate(fallback_prompt, system=_SYNTH_SYSTEM, temperature=0.2)

    return AgentResult(
        goal=goal,
        answer=answer,
        steps=steps,
        total_duration_ms=round((time.time() - t_start) * 1000, 1),
        sources=list(set(all_sources)),
    )
