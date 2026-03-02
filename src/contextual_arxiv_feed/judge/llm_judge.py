"""LLM Judge for Stage 2 abstract evaluation.

Fallback chain: Cerebras → Gemini → Local Qwen.
See config/judge.yaml for provider setup.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx

from contextual_arxiv_feed.config import JudgeConfig, TopicConfig
from contextual_arxiv_feed.judge.judge import TEMPLATES_DIR, JudgeResult
from contextual_arxiv_feed.judge.schema import JudgeOutput, parse_judge_output, truncate_rationale
from contextual_arxiv_feed.keys.rotator import KeyPool

logger = logging.getLogger(__name__)

# Fallback LLM config (local Qwen on Oracle Cloud)
FALLBACK_BASE_URL = "http://127.0.0.1:8080/v1"
FALLBACK_API_KEY = "not-needed"
FALLBACK_MODEL_ID = "qwen2.5-14b-instruct-q4_k_m"


def _build_llm_key_pool() -> KeyPool:
    """Build primary LLM key pool from environment.

    Reads LLM_API_KEYS (comma-separated) or LLM_API_KEY (single).
    """
    keys_str = os.getenv("LLM_API_KEYS", "")
    if keys_str:
        keys = keys_str.split(",")
    else:
        single = os.getenv("LLM_API_KEY", "")
        keys = [single] if single else []
    return KeyPool(keys, cooldown_seconds=300)


def _build_secondary_key_pool() -> KeyPool:
    """Build secondary (Gemini) key pool from environment.

    Reads LLM_SECONDARY_API_KEYS (comma-separated) or LLM_SECONDARY_API_KEY.
    """
    keys_str = os.getenv("LLM_SECONDARY_API_KEYS", "")
    if keys_str:
        keys = keys_str.split(",")
    else:
        single = os.getenv("LLM_SECONDARY_API_KEY", "")
        keys = [single] if single else []
    return KeyPool(keys, cooldown_seconds=300)


class LLMJudge:
    """LLM judge for evaluating paper abstracts.

    3-tier fallback: Cerebras → Gemini → Local Qwen.
    Each tier rotates keys round-robin. On 429, exhausts all keys
    in current tier before falling to the next.

    Interface: judge(title, abstract) -> JudgeResult
    """

    def __init__(
        self,
        config: JudgeConfig,
        topics: list[TopicConfig],
        base_url: str = "",
        key_pool: KeyPool | None = None,
    ):
        self._config = config
        self._topics = topics

        # Primary LLM endpoint (Cerebras — best model)
        self._base_url = base_url or os.getenv("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
        self._model_id = config.model_id
        self._key_pool = key_pool or _build_llm_key_pool()

        # Secondary LLM endpoint (Gemini — fallback)
        self._secondary_base_url = os.getenv("LLM_SECONDARY_BASE_URL", "")
        self._secondary_model_id = os.getenv("LLM_SECONDARY_MODEL_ID", "")
        self._secondary_key_pool = _build_secondary_key_pool()

        # Final fallback (local Qwen, always available)
        self._fallback_base_url = os.getenv("LLM_FALLBACK_BASE_URL", FALLBACK_BASE_URL)
        self._fallback_api_key = os.getenv("LLM_FALLBACK_API_KEY", FALLBACK_API_KEY)
        self._fallback_model_id = os.getenv("LLM_FALLBACK_MODEL_ID", FALLBACK_MODEL_ID)

        self._prompt_template = self._load_prompt_template()
        self._client = httpx.Client(timeout=120.0)

        logger.info(
            f"Judge initialized: model={self._model_id}, "
            f"base_url={self._base_url}, keys={self._key_pool.size}, "
            f"secondary={'yes' if self._secondary_base_url else 'none'}"
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LLMJudge:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _load_prompt_template(self) -> str:
        template_path = TEMPLATES_DIR / "base_prompt.txt"
        with open(template_path) as f:
            return f.read()

    def _build_prompt(self, title: str, abstract: str) -> str:
        thresholds = self._config.get_thresholds()
        return self._prompt_template.format(
            title=title,
            abstract=abstract,
            strictness=self._config.strictness,
            min_quality_i=thresholds.min_quality_i,
            prompt_version=self._config.prompt_version,
            model_id=self._model_id,
            max_rationale_length=self._config.max_rationale_length,
        )

    def judge(self, title: str, abstract: str) -> JudgeResult:
        """Judge a paper's abstract for quality."""
        prompt = self._build_prompt(title, abstract)
        logger.debug(f"Judging paper: {title[:50]}...")

        try:
            raw_response = self._call_llm(prompt)

            output = self._parse_response(raw_response)
            if output is None:
                return JudgeResult(
                    success=False,
                    error="Failed to parse judge response as JSON",
                    raw_response=raw_response,
                )

            # Truncate rationale if needed
            output_dict = output.to_dict()
            output_dict["rationale"] = truncate_rationale(
                output_dict["rationale"], self._config.max_rationale_length
            )
            output = parse_judge_output(output_dict)

            logger.info(
                f"Judge verdict: quality={output.quality_verdict}, quality_i={output.quality_i}"
            )

            return JudgeResult(
                success=True,
                output=output,
                raw_response=raw_response,
            )

        except Exception as e:
            logger.error(f"Error during judge call: {e}")
            return JudgeResult(success=False, error=str(e))

    def _call_llm(self, prompt: str) -> str:
        """Call LLM with 3-tier fallback: primary → secondary → local.

        1. Try primary (Cerebras) with key rotation
        2. On 429 / all keys exhausted: try secondary (Gemini) with key rotation
        3. On 429 / all keys exhausted: fall back to local Qwen
        """
        # Tier 1: Primary (Cerebras)
        result = self._try_tier(self._base_url, self._model_id, self._key_pool, prompt)
        if result is not None:
            return result

        # Tier 2: Secondary (Gemini)
        if self._secondary_base_url and self._secondary_key_pool.size > 0:
            logger.info("Primary exhausted, trying secondary provider")
            result = self._try_tier(
                self._secondary_base_url, self._secondary_model_id,
                self._secondary_key_pool, prompt,
            )
            if result is not None:
                return result

        # Tier 3: Local Qwen
        logger.warning("All API keys exhausted, using local Qwen fallback")
        return self._do_llm_request(
            self._fallback_base_url,
            self._fallback_api_key,
            self._fallback_model_id,
            prompt,
        )

    def _try_tier(
        self, base_url: str, model_id: str, key_pool: KeyPool, prompt: str,
    ) -> str | None:
        """Try a provider tier, rotating through all its keys. Returns None if all exhausted."""
        while True:
            api_key = key_pool.get_key()
            if api_key is None:
                return None
            try:
                result = self._do_llm_request(base_url, api_key, model_id, prompt)
                key_pool.report_success(api_key)
                return result
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    key_pool.report_rate_limit(api_key)
                    continue  # try next key in this tier
                raise

    def _do_llm_request(
        self, base_url: str, api_key: str, model_id: str, prompt: str
    ) -> str:
        """Make the actual HTTP request to the LLM."""
        response = self._client.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_id,
                "max_tokens": 1024,
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

    def _parse_response(self, response: str) -> JudgeOutput | None:
        """Parse LLM response to extract JSON and cast numerics to int."""
        try:
            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    logger.error("No JSON found in response")
                    return None

            data = json.loads(json_str)
            data = self._cast_numerics_to_int(data)
            return parse_judge_output(data)

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return None
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            return None

    def _cast_numerics_to_int(self, data: dict[str, Any]) -> dict[str, Any]:
        """Cast all numeric fields to int to satisfy INT-ONLY validation."""
        int_fields = [
            "prompt_version",
            "quality_i",
            "confidence_i",
        ]
        for field in int_fields:
            if field in data and isinstance(data[field], (int, float)):
                data[field] = int(round(data[field]))

        breakdown = data.get("quality_breakdown_i", {})
        if isinstance(breakdown, dict):
            for key in ["novelty_i", "relevance_i", "technical_depth_i"]:
                if key in breakdown and isinstance(breakdown[key], (int, float)):
                    breakdown[key] = int(round(breakdown[key]))

        return data
