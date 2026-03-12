"""Discovery Agent for Stage 1.5 semantic matching.

Runs on papers that FAILED Stage 1 keyword matching to catch papers
that discuss relevant topics using different terminology.

Uses Cerebras with key rotation (same as judge).
Uses DISCOVERY_* env vars if set, else falls back to LLM_* vars.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field

import httpx

from contextual_arxiv_feed.config import TopicConfig
from contextual_arxiv_feed.keys.rotator import KeyPool

logger = logging.getLogger(__name__)

DISCOVERY_PROMPT = """You are a research paper classifier for an LLM systems and applied AI engineering research feed. Determine if this paper should be included.

## Topics
{topic_list}

## Paper
**Title:** {title}
**Abstract snippet:** {abstract_snippet}

## Instructions
Mark as relevant if ANY of these apply:
1. **Topic match**: the paper discusses concepts related to any topic above, even using different terminology
   - Examples: "attention bottleneck" = context window, "grounded dialogue" = RAG, "cognitive scaffolding" = agent architecture
2. **Novel contribution**: the paper introduces a genuinely new concept, technique, or approach for LLM systems or applied AI engineering — even if it doesn't match any predefined topic. If it's something practitioners building LLM systems need to know about, include it.

When in doubt, mark as relevant — the judge will do deeper evaluation next.

Respond with ONLY a valid JSON object:

```json
{{
  "is_relevant": true or false,
  "matched_topics": ["topic_key_1"],
  "reason": "<one sentence>"
}}
```
"""


@dataclass
class DiscoveryResult:
    """Result from Discovery Agent check."""

    is_relevant: bool = False
    matched_topics: list[str] = field(default_factory=list)
    reason: str = ""
    error: str = ""


def _build_discovery_key_pool() -> KeyPool:
    """Build primary key pool for discovery agent.

    Priority: DISCOVERY_API_KEYS > DISCOVERY_API_KEY > LLM_API_KEYS > LLM_API_KEY
    """
    keys_str = os.getenv("DISCOVERY_API_KEYS", "")
    if keys_str:
        return KeyPool(keys_str.split(","), cooldown_seconds=300)

    single = os.getenv("DISCOVERY_API_KEY", "")
    if single:
        return KeyPool([single], cooldown_seconds=300)

    keys_str = os.getenv("LLM_API_KEYS", "")
    if keys_str:
        return KeyPool(keys_str.split(","), cooldown_seconds=300)

    single = os.getenv("LLM_API_KEY", "")
    if single:
        return KeyPool([single], cooldown_seconds=300)

    return KeyPool([], cooldown_seconds=300)


class DiscoveryAgent:
    """Smart agent for semantic topic matching.

    Stage 1.5: runs on papers that failed keyword matching.
    Uses Cerebras with key rotation (same as judge).
    Uses DISCOVERY_* env vars if set, else falls back to LLM_* vars.
    """

    def __init__(
        self,
        topics: list[TopicConfig],
        base_url: str = "",
        model_id: str = "",
        key_pool: KeyPool | None = None,
    ):
        self._topics = topics

        # Primary (Cerebras): DISCOVERY_* env vars take priority over LLM_*
        self._base_url = (
            base_url
            or os.getenv("DISCOVERY_BASE_URL", "")
            or os.getenv("LLM_BASE_URL", "https://api.cerebras.ai/v1")
        )
        self._model_id = (
            model_id
            or os.getenv("DISCOVERY_MODEL_ID", "")
            or os.getenv("LLM_MODEL_ID", "gpt-oss-120b")
        )
        if key_pool is not None:
            self._key_pool = key_pool
        else:
            self._key_pool = _build_discovery_key_pool()

        self._client = httpx.Client(timeout=30.0)
        self._topic_list = self._build_topic_list()

        logger.info(
            f"Discovery Agent initialized: model={self._model_id}, "
            f"base_url={self._base_url}, keys={self._key_pool.size}"
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> DiscoveryAgent:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _build_topic_list(self) -> str:
        lines = []
        for topic in self._topics:
            lines.append(f"- **{topic.key}**: {topic.name} — {topic.description.strip()}")
        return "\n".join(lines)

    def check(self, title: str, abstract_snippet: str) -> DiscoveryResult:
        """Check if a paper is relevant to any topic."""
        prompt = DISCOVERY_PROMPT.format(
            topic_list=self._topic_list,
            title=title,
            abstract_snippet=abstract_snippet[:500],
        )

        try:
            content = self._call_llm(prompt)
            return self._parse_response(content)
        except Exception as e:
            logger.error(f"Discovery Agent error: {e}")
            return DiscoveryResult(error=str(e))

    def _call_llm(self, prompt: str) -> str:
        """Call Cerebras LLM with key rotation."""
        result = self._try_tier(self._base_url, self._model_id, self._key_pool, prompt)
        if result is not None:
            return result

        raise RuntimeError("All Cerebras API keys exhausted or unavailable")

    def _try_tier(
        self, base_url: str, model_id: str, key_pool: KeyPool, prompt: str,
    ) -> str | None:
        """Try a provider tier, rotating through all its keys. Returns None if all exhausted."""
        while True:
            api_key = key_pool.get_key()
            if api_key is None:
                return None
            try:
                result = self._do_request(base_url, api_key, model_id, prompt)
                key_pool.report_success(api_key)
                return result
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    key_pool.report_rate_limit(api_key)
                    continue
                raise

    def _do_request(
        self, base_url: str, api_key: str, model_id: str, prompt: str
    ) -> str:
        response = self._client.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_id,
                "max_tokens": 256,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""

    def _parse_response(self, response: str) -> DiscoveryResult:
        try:
            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    return DiscoveryResult(error="No JSON in response")

            data = json.loads(json_str)

            valid_keys = {t.key for t in self._topics}
            matched = [t for t in data.get("matched_topics", []) if t in valid_keys]

            return DiscoveryResult(
                is_relevant=bool(data.get("is_relevant", False)),
                matched_topics=matched,
                reason=str(data.get("reason", "")),
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Discovery Agent parse error: {e}")
            return DiscoveryResult(error=str(e))
