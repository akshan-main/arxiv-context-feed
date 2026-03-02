"""Tests for configuration loading and validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from contextual_arxiv_feed.config import (
    CategoriesConfig,
    JudgeConfig,
    StrictnessPreset,
    TopicConfig,
    TopicsConfig,
    load_config,
    validate_topic_against_categories,
)


class TestTopicConfig:
    """Tests for TopicConfig validation."""

    def test_valid_topic_key(self):
        """Test valid topic key patterns."""
        valid_keys = ["test", "test-topic", "test_topic", "llm-inference", "ml_ops"]
        for key in valid_keys:
            topic = TopicConfig(key=key, name="Test", keywords=["test"])
            assert topic.key == key

    def test_invalid_topic_key(self):
        """Test invalid topic key patterns."""
        invalid_keys = ["Test", "TEST", "test topic", "test.topic", "123test", "-test"]
        for key in invalid_keys:
            with pytest.raises(ValidationError, match="must match pattern"):
                TopicConfig(key=key, name="Test", keywords=["test"])

    def test_requires_keywords_or_phrases(self):
        """Test that topic requires at least one keyword or phrase."""
        with pytest.raises(ValidationError, match="at least one keyword or phrase"):
            TopicConfig(key="test", name="Test")

    def test_accepts_keywords_only(self):
        """Test topic with only keywords."""
        topic = TopicConfig(key="test", name="Test", keywords=["word"])
        assert topic.keywords == ["word"]
        assert topic.phrases == []

    def test_accepts_phrases_only(self):
        """Test topic with only phrases."""
        topic = TopicConfig(key="test", name="Test", phrases=["a phrase"])
        assert topic.keywords == []
        assert topic.phrases == ["a phrase"]


class TestStrictnessPreset:
    """Tests for StrictnessPreset int-only enforcement."""

    def test_accepts_int_values(self):
        """Test that int values are accepted."""
        preset = StrictnessPreset(min_quality_i=60)
        assert preset.min_quality_i == 60

    def test_rejects_float_values(self):
        """Test that float values are rejected."""
        with pytest.raises(ValidationError, match="Float values not allowed"):
            StrictnessPreset(min_quality_i=60.5)

    def test_rejects_out_of_range(self):
        """Test that out-of-range values are rejected."""
        with pytest.raises(ValidationError):
            StrictnessPreset(min_quality_i=-1)

        with pytest.raises(ValidationError):
            StrictnessPreset(min_quality_i=101)


class TestJudgeConfig:
    """Tests for JudgeConfig validation."""

    def test_default_values(self):
        """Test default configuration values."""
        config = JudgeConfig()
        assert config.provider == "gemini"
        assert config.model_id == "gemini-2.5-flash"
        assert config.strictness == "medium"
        assert config.prompt_version == 1

    def test_get_thresholds(self):
        """Test getting thresholds for different strictness levels."""
        config = JudgeConfig(strictness="low")
        thresholds = config.get_thresholds()
        assert thresholds.min_quality_i == 40

        config = JudgeConfig(strictness="high")
        thresholds = config.get_thresholds()
        assert thresholds.min_quality_i == 80

    def test_rejects_float_prompt_version(self):
        """Test that float prompt_version is rejected."""
        with pytest.raises(ValidationError, match="Float values not allowed"):
            JudgeConfig(prompt_version=1.5)


class TestTopicsConfig:
    """Tests for TopicsConfig."""

    def test_get_enabled_topics(self):
        """Test filtering enabled topics."""
        topics = TopicsConfig(
            topics=[
                TopicConfig(key="enabled", name="Enabled", keywords=["test"], enabled=True),
                TopicConfig(key="disabled", name="Disabled", keywords=["test"], enabled=False),
            ]
        )
        enabled = topics.get_enabled_topics()
        assert len(enabled) == 1
        assert enabled[0].key == "enabled"

    def test_get_topic_by_key(self):
        """Test getting topic by key."""
        topic = TopicConfig(key="test", name="Test", keywords=["test"])
        topics = TopicsConfig(topics=[topic])

        assert topics.get_topic("test") == topic
        assert topics.get_topic("nonexistent") is None


class TestCategoriesConfig:
    """Tests for CategoriesConfig."""

    def test_is_valid(self):
        """Test category validation."""
        config = CategoriesConfig(categories=["cs.LG", "cs.AI"])

        assert config.is_valid("cs.LG") is True
        assert config.is_valid("cs.AI") is True
        assert config.is_valid("cs.CV") is False


class TestValidateTopicAgainstCategories:
    """Tests for topic category validation."""

    def test_valid_categories(self):
        """Test topic with valid categories."""
        topic = TopicConfig(
            key="test",
            name="Test",
            keywords=["test"],
            arxiv_categories=["cs.LG", "cs.AI"],
        )
        categories = CategoriesConfig(categories=["cs.LG", "cs.AI", "cs.CL"])

        errors = validate_topic_against_categories(topic, categories)
        assert errors == []

    def test_invalid_categories(self):
        """Test topic with invalid categories."""
        topic = TopicConfig(
            key="test",
            name="Test",
            keywords=["test"],
            arxiv_categories=["cs.LG", "invalid.CAT"],
        )
        categories = CategoriesConfig(categories=["cs.LG", "cs.AI"])

        errors = validate_topic_against_categories(topic, categories)
        assert len(errors) == 1
        assert "invalid.CAT" in errors[0]


class TestLoadConfig:
    """Tests for configuration loading."""

    def test_load_from_temp_dir(self, temp_config_dir: Path):
        """Test loading config from temporary directory."""
        config = load_config(temp_config_dir)

        assert len(config.topics.topics) == 1
        assert config.topics.topics[0].key == "test-topic"
        assert config.judge.provider == "local"
        assert config.sources.enable_openalex is True
        assert "cs.LG" in config.categories.categories
