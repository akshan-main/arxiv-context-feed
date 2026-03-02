"""Round-robin API key rotation with cooldown on rate limits.

Supports multiple API keys per service (e.g., OpenAlex, LLM providers).
Keys from a team of 5 are loaded via comma-separated environment variables.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class KeyState:
    """State of a single API key."""

    key: str
    exhausted_until: float = 0.0  # monotonic timestamp
    request_count: int = 0
    error_count: int = 0


class KeyPool:
    """Round-robin pool of API keys for a single service.

    Rotates through available keys. When a key hits a rate limit,
    it enters cooldown for `cooldown_seconds` and the next key is used.
    """

    def __init__(self, keys: list[str], cooldown_seconds: int = 60):
        """Initialize key pool.

        Args:
            keys: List of API keys (empty strings are filtered out).
            cooldown_seconds: Seconds to wait after rate limit before retrying key.
        """
        # Filter out empty/whitespace keys
        valid_keys = [k.strip() for k in keys if k.strip()]
        self._keys = [KeyState(key=k) for k in valid_keys]
        self._cooldown = cooldown_seconds
        self._index = 0

    @property
    def size(self) -> int:
        """Number of keys in the pool."""
        return len(self._keys)

    def get_key(self) -> str | None:
        """Get the next available API key (round-robin).

        Skips keys that are in cooldown. Returns None if all keys
        are exhausted.

        Returns:
            API key string, or None if no keys available.
        """
        if not self._keys:
            return None

        now = time.monotonic()
        checked = 0

        while checked < len(self._keys):
            state = self._keys[self._index]
            self._index = (self._index + 1) % len(self._keys)

            if state.exhausted_until <= now:
                state.request_count += 1
                return state.key

            checked += 1

        # All keys exhausted
        logger.warning("All API keys are in cooldown")
        return None

    def report_success(self, key: str) -> None:
        """Report successful use of a key.

        Args:
            key: The API key that succeeded.
        """
        for state in self._keys:
            if state.key == key:
                state.exhausted_until = 0.0
                break

    def report_rate_limit(self, key: str) -> None:
        """Report rate limit hit for a key, putting it in cooldown.

        Args:
            key: The API key that was rate limited.
        """
        for state in self._keys:
            if state.key == key:
                state.exhausted_until = time.monotonic() + self._cooldown
                state.error_count += 1
                logger.info(
                    f"Key ending ...{key[-4:]} rate limited, cooldown {self._cooldown}s"
                )
                break


class KeyRotator:
    """Manages key pools for multiple services."""

    def __init__(self, cooldown_seconds: int = 60):
        """Initialize rotator.

        Args:
            cooldown_seconds: Default cooldown for all pools.
        """
        self._cooldown = cooldown_seconds
        self._pools: dict[str, KeyPool] = {}

    def add_pool(self, service: str, keys: list[str]) -> None:
        """Add a key pool for a service.

        Args:
            service: Service name (e.g., "openalex", "llm").
            keys: List of API keys.
        """
        self._pools[service] = KeyPool(keys, self._cooldown)
        logger.info(f"Key pool '{service}': {self._pools[service].size} keys")

    def get_pool(self, service: str) -> KeyPool | None:
        """Get the key pool for a service.

        Args:
            service: Service name.

        Returns:
            KeyPool or None if service not registered.
        """
        return self._pools.get(service)

    @classmethod
    def from_environment(cls, cooldown_seconds: int = 60) -> KeyRotator:
        """Create a KeyRotator from environment variables.

        Looks for:
        - OPENALEX_API_KEYS (comma-separated) or OPENALEX_API_KEY (single)
        - LLM_API_KEYS (comma-separated) or LLM_API_KEY (single)

        Args:
            cooldown_seconds: Cooldown period.

        Returns:
            Configured KeyRotator.
        """
        rotator = cls(cooldown_seconds)

        # OpenAlex keys
        openalex_keys = os.getenv("OPENALEX_API_KEYS", "")
        if openalex_keys:
            rotator.add_pool("openalex", openalex_keys.split(","))
        else:
            single = os.getenv("OPENALEX_API_KEY", "")
            if single:
                rotator.add_pool("openalex", [single])

        # LLM keys (Cerebras / Gemini / any provider)
        llm_keys = os.getenv("LLM_API_KEYS", "")
        if llm_keys:
            rotator.add_pool("llm", llm_keys.split(","))
        else:
            single = os.getenv("LLM_API_KEY", "")
            if single:
                rotator.add_pool("llm", [single])

        return rotator
