"""LLM Judge for Stage 2 abstract evaluation.

Fallback chain: Cerebras -> Gemini -> Local Qwen.
See config/judge.yaml for provider setup.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import time

import httpx

from contextual_arxiv_feed.config import JudgeConfig, TopicConfig
from contextual_arxiv_feed.judge.judge import TEMPLATES_DIR, JudgeResult
from contextual_arxiv_feed.judge.schema import JudgeOutput, parse_judge_output, truncate_rationale
from contextual_arxiv_feed.keys.rotator import KeyPool

logger = logging.getLogger(__name__)

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
    return KeyPool(keys, cooldown_seconds=60)


def _build_secondary_key_pool() -> KeyPool:
    """Build secondary (Cerebras) key pool from environment.

    Reads LLM_SECONDARY_API_KEYS (comma-separated) or LLM_SECONDARY_API_KEY.
    """
    keys_str = os.getenv("LLM_SECONDARY_API_KEYS", "")
    if keys_str:
        keys = keys_str.split(",")
    else:
        single = os.getenv("LLM_SECONDARY_API_KEY", "")
        keys = [single] if single else []
    return KeyPool(keys, cooldown_seconds=60)


class LLMJudge:
    """LLM judge for evaluating paper abstracts.

    3-tier fallback: Cerebras -> Gemini -> Local Qwen.
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

        self._base_url = base_url or os.getenv("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
        self._model_id = config.model_id
        self._key_pool = key_pool or _build_llm_key_pool()

        self._secondary_base_url = os.getenv("LLM_SECONDARY_BASE_URL", "")
        self._secondary_model_id = os.getenv("LLM_SECONDARY_MODEL_ID", "")
        self._secondary_key_pool = _build_secondary_key_pool()

        self._fallback_base_url = os.getenv("LLM_FALLBACK_BASE_URL", FALLBACK_BASE_URL)
        self._fallback_api_key = os.getenv("LLM_FALLBACK_API_KEY", FALLBACK_API_KEY)
        self._fallback_model_id = os.getenv("LLM_FALLBACK_MODEL_ID", FALLBACK_MODEL_ID)

        self._prompt_template = self._load_prompt_template()
        self._client = httpx.Client(timeout=120.0)
        self._last_llm_call = 0.0  # monotonic timestamp for rate limit throttle
        self._llm_min_interval = 2.0  # 4 keys x 10 RPM = 40 RPM; 2s = 30 RPM -> 7.5 RPM/key

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
        """Call LLM with fallback chain: primary -> secondary -> local.

        With 1 req/sec throttle + 4 keys round-robin, primary should handle
        everything. Secondary/local are safety nets for outages, not rate limits.
        """
        # Primary (Gemini) — always wait for cooldown, should never need fallback
        result = self._try_tier(
            self._base_url, self._model_id, self._key_pool, prompt,
        )
        if result is not None:
            return result

        # Secondary (Cerebras) — only if Gemini has a real outage
        if self._secondary_base_url and self._secondary_key_pool.size > 0:
            logger.warning("Primary unavailable, trying secondary provider")
            result = self._try_tier(
                self._secondary_base_url, self._secondary_model_id,
                self._secondary_key_pool, prompt,
            )
            if result is not None:
                return result

        # Local Qwen — last resort
        logger.warning("All API providers unavailable, using local Qwen")
        return self._do_llm_request(
            self._fallback_base_url,
            self._fallback_api_key,
            self._fallback_model_id,
            prompt,
        )

    def _try_tier(
        self, base_url: str, model_id: str, key_pool: KeyPool, prompt: str,
    ) -> str | None:
        """Try a provider tier with round-robin keys.

        On 429: put the key in cooldown with server's retry delay,
        then immediately try the next key. If ALL keys are in cooldown,
        sleep until the soonest one is available.
        Never gives up — daily quota is sufficient, just respect RPM.
        Only returns None if pool is empty (no keys configured).
        """
        while True:
            api_key = key_pool.get_key()
            if api_key is None:
                # All keys in cooldown — wait for the soonest one
                wait_secs = key_pool.seconds_until_next_available()
                if wait_secs is None:
                    return None  # no keys configured at all
                logger.info(f"All keys in cooldown, waiting {wait_secs:.0f}s")
                time.sleep(wait_secs + 1.0)
                continue
            try:
                result = self._do_llm_request(base_url, api_key, model_id, prompt)
                key_pool.report_success(api_key)
                return result
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    retry_secs = self._parse_retry_delay(e.response)
                    delay = retry_secs if retry_secs and retry_secs > 0 else 60
                    delay += 5  # buffer to ensure we're past the window boundary
                    logger.info(f"429 on key ...{api_key[-4:]}, cooldown {delay:.0f}s — rotating to next key")
                    key_pool.report_rate_limit_with_delay(api_key, delay)
                    continue  # immediately try next key (no sleep)
                raise

    def _do_llm_request(
        self, base_url: str, api_key: str, model_id: str, prompt: str
    ) -> str:
        """Make the actual HTTP request to the LLM (rate-limited)."""
        # Throttle to stay under RPM limit (all keys share same quota)
        now = time.monotonic()
        elapsed = now - self._last_llm_call
        if elapsed < self._llm_min_interval:
            time.sleep(self._llm_min_interval - elapsed)
        self._last_llm_call = time.monotonic()

        response = self._client.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_id,
                "max_tokens": 4096,
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

    def _parse_retry_delay(self, response: httpx.Response) -> float | None:
        """Parse 'retry in Xs' from 429 response body."""
        try:
            text = response.text
            match = re.search(r"retry in (\d+(?:\.\d+)?)s", text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        except Exception:
            pass
        return None

    def _parse_response(self, response: str) -> JudgeOutput | None:
        """Parse LLM response to extract JSON and cast numerics to int."""
        try:
            if not response or not response.strip():
                logger.error("Empty response from LLM")
                return None

            # Strip thinking tags (Gemini 2.5 Flash thinking mode)
            cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            if not cleaned:
                cleaned = response  # fallback to original if stripping removed everything

            json_match = re.search(r"```json\s*(.*?)\s*```", cleaned, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    logger.error(f"No JSON found in response (first 200 chars): {response[:200]}")
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
