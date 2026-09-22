"""OpenAI-compatible chat + embedding access.

Only the OpenAI wire protocol is assumed, so a self-hosted vLLM/Ollama server,
an internal gateway or OpenAI itself is a ``.env`` change and never a code
change. Credentials and model names live in the environment (``NEWS__LLM__*``).

Structured output is requested as a JSON object and validated with Pydantic
rather than through a provider-specific schema endpoint - JSON mode is the
common denominator across compatible servers.
"""

from __future__ import annotations

import json
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from yoda.common.logger import logger
from yoda.config.research import NewsLLMConfig

Model = TypeVar("Model", bound=BaseModel)


def _json_slice(text: str) -> str:
    """Pull the outermost JSON object out of a chatty completion."""
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if 0 <= start < end else text


class LLMClient:
    """Thin, deterministic wrapper: temperature 0 and a fixed seed where honoured."""

    def __init__(self, config: NewsLLMConfig):
        if not config.base_url or not config.model:
            raise ValueError(
                "News LLM backend needs NEWS__LLM__BASE_URL and NEWS__LLM__MODEL"
            )
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key or "not-needed",
            timeout=config.timeout,
        )

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=20))
    def _complete(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            seed=self.config.seed,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or "{}"

    def structured(self, system: str, user: str, schema: type[Model]) -> Model:
        """Complete and validate, falling back to the schema's defaults."""
        raw = self._complete(system, user)
        try:
            return schema.model_validate(json.loads(_json_slice(raw)))
        except (json.JSONDecodeError, ValidationError) as error:
            logger.warning(
                "news_llm_unparsed schema=%s error=%s", schema.__name__, error
            )
            return schema()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=20))
    def embed(self, text: str) -> list[float]:
        if not self.config.embed_model or not text.strip():
            return []
        response = self.client.embeddings.create(
            model=self.config.embed_model, input=text
        )
        return list(response.data[0].embedding)
