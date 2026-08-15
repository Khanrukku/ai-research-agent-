"""Small async wrapper around the Google GenAI SDK."""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _get_client() -> genai.Client:
    if not settings.gemini_api_key:
        raise EnvironmentError("GEMINI_API_KEY is not configured.")
    return genai.Client(api_key=settings.gemini_api_key)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
async def generate(
    prompt: str,
    system: str = "",
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    client = _get_client()
    contents = f"{system}\n\n{prompt}" if system else prompt
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.models.generate_content(
            model=settings.gemini_model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        ),
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
async def embed(text: str) -> list[float]:
    client = _get_client()
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: client.models.embed_content(
            model=settings.embedding_model,
            contents=text,
        ),
    )
    embeddings = getattr(result, "embeddings", None)
    if embeddings:
        first = embeddings[0]
        values = getattr(first, "values", first)
        return list(values)
    embedding = getattr(result, "embedding", None)
    values = getattr(embedding, "values", embedding)
    if values:
        return list(values)
    raise ValueError(f"Unexpected embedding response: {type(result)!r}")


async def embed_query(text: str) -> list[float]:
    return await embed(text)


async def extract_json(prompt: str, schema_hint: str = "") -> Any:
    system = (
        "Return only valid JSON. No markdown fences or commentary. "
        + schema_hint
    )
    raw = await generate(prompt, system=system, temperature=0.0, max_tokens=2048)
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {raw[:300]}") from exc
