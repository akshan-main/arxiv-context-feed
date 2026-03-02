"""Tests for API key rotation."""

from unittest.mock import patch

from contextual_arxiv_feed.keys.rotator import KeyPool, KeyRotator


class TestKeyPool:
    """Test KeyPool round-robin rotation."""

    def test_round_robin(self):
        pool = KeyPool(["key1", "key2", "key3"])
        assert pool.size == 3
        assert pool.get_key() == "key1"
        assert pool.get_key() == "key2"
        assert pool.get_key() == "key3"
        assert pool.get_key() == "key1"  # wraps around

    def test_empty_keys_filtered(self):
        pool = KeyPool(["key1", "", "  ", "key2"])
        assert pool.size == 2
        assert pool.get_key() == "key1"
        assert pool.get_key() == "key2"

    def test_empty_pool_returns_none(self):
        pool = KeyPool([])
        assert pool.size == 0
        assert pool.get_key() is None

    def test_all_empty_strings_returns_none(self):
        pool = KeyPool(["", "", ""])
        assert pool.size == 0
        assert pool.get_key() is None

    def test_rate_limit_skips_key(self):
        pool = KeyPool(["key1", "key2"], cooldown_seconds=60)

        # Rate limit key1
        pool.report_rate_limit("key1")

        # Should skip key1 and return key2
        assert pool.get_key() == "key2"

    def test_all_exhausted_returns_none(self):
        pool = KeyPool(["key1", "key2"], cooldown_seconds=60)
        pool.report_rate_limit("key1")
        pool.report_rate_limit("key2")

        assert pool.get_key() is None

    def test_success_resets_cooldown(self):
        pool = KeyPool(["key1"], cooldown_seconds=60)
        pool.report_rate_limit("key1")
        assert pool.get_key() is None

        pool.report_success("key1")
        assert pool.get_key() == "key1"

    def test_cooldown_expires(self):
        pool = KeyPool(["key1"], cooldown_seconds=0)  # 0 second cooldown
        pool.report_rate_limit("key1")
        # With 0 cooldown, should be immediately available
        assert pool.get_key() == "key1"


class TestKeyRotator:
    """Test KeyRotator multi-service management."""

    def test_add_and_get_pool(self):
        rotator = KeyRotator()
        rotator.add_pool("openalex", ["key1", "key2"])
        rotator.add_pool("llm", ["key3"])

        pool = rotator.get_pool("openalex")
        assert pool is not None
        assert pool.size == 2

        pool = rotator.get_pool("llm")
        assert pool is not None
        assert pool.size == 1

    def test_missing_pool_returns_none(self):
        rotator = KeyRotator()
        assert rotator.get_pool("nonexistent") is None

    @patch.dict("os.environ", {
        "OPENALEX_API_KEYS": "k1,k2,k3",
    })
    def test_from_environment_multi_keys(self):
        rotator = KeyRotator.from_environment()
        pool = rotator.get_pool("openalex")
        assert pool is not None
        assert pool.size == 3

    @patch.dict("os.environ", {
        "OPENALEX_API_KEY": "single-key",
    }, clear=True)
    def test_from_environment_single_key_fallback(self):
        rotator = KeyRotator.from_environment()
        pool = rotator.get_pool("openalex")
        assert pool is not None
        assert pool.size == 1
        assert pool.get_key() == "single-key"

    @patch.dict("os.environ", {}, clear=True)
    def test_from_environment_no_keys(self):
        rotator = KeyRotator.from_environment()
        assert rotator.get_pool("openalex") is None
