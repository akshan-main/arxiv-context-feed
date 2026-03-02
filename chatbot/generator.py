"""Text generator for the chatbot.

Uses same LLM endpoint as the pipeline (Gemini/MiniMax/local Qwen).
Swap providers by changing LLM_BASE_URL and LLM_API_KEY env vars.
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)


class Generator:
    """Text generator using OpenAI-compatible API."""

    def __init__(
        self,
        api_base: str = "",
        api_key: str = "",
        model: str = "",
    ):
        self._api_base = api_base or os.getenv(
            "LLM_BASE_URL", "http://127.0.0.1:8080/v1"
        )
        self._api_key = api_key or os.getenv("LLM_API_KEY", "not-needed")
        self._model = model or os.getenv("LLM_MODEL", "gemini-2.5-flash")
        self._client = httpx.Client(timeout=120.0)

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate text from prompt."""
        response = self._client.post(
            f"{self._api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def close(self) -> None:
        self._client.close()


def create_generator(
    api_base: str = "",
    api_key: str = "",
    model: str = "",
) -> Generator:
    """Create a generator with the given or default settings."""
    return Generator(api_base=api_base, api_key=api_key, model=model)
