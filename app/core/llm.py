"""
app/core/llm.py
---------------
Gemini API wrapper using the new google-genai package.
"""
from __future__ import annotations

import asyncio
from typing import Any

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _get_client() -> genai.Client:
    if not settings.gemini_api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Get a free key at https://aistudio.google.com"
        )
    return genai.Client(api_key=settings.gemini_api_key)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def generate(
    prompt: str,
    system: str = "",
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    client = _get_client()
    full_prompt = f"{system}\n\n{prompt}" if system else prompt

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.models.generate_content(
            model=settings.gemini_model,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        ),
    )
    text = response.text
    logger.debug("Gemini generated %d chars", len(text))
    return text


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def embed(text: str) -> list[float]:
    client = _get_client()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=text,
        ),
    )
    # handle both old and new SDK response shapes safely
    if hasattr(result, "embeddings"):
        emb = result.embeddings
        if isinstance(emb, list):
            first = emb[0]
            return list(first.values) if hasattr(first, "values") else list(first)
        return list(emb.values) if hasattr(emb, "values") else list(emb)
    elif hasattr(result, "embedding"):
        return list(result.embedding.values)
    raise ValueError(f"Unexpected embed response shape: {type(result)}")


async def embed_query(text: str) -> list[float]:
    return await embed(text)


async def extract_json(prompt: str, schema_hint: str = "") -> Any:
    import json
    import re

    system = (
        "You are a JSON extraction engine. "
        "Respond ONLY with valid JSON — no prose, no markdown code fences. "
        f"{schema_hint}"
    )
    raw = await generate(prompt, system=system, temperature=0.0)
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failed: %s\nRaw: %s", exc, raw[:200])
        raise