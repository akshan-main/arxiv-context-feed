"""Tests for issue payload validation (Google Form -> Issue -> PR)."""

import pytest

from contextual_arxiv_feed.config import CategoriesConfig
from contextual_arxiv_feed.pipeline.apply_config_change import (
    IssuePayload,
    parse_issue_payload,
    validate_payload,
)


@pytest.fixture
def categories() -> CategoriesConfig:
    """Sample categories config."""
    return CategoriesConfig(categories=["cs.LG", "cs.AI", "cs.CL", "stat.ML"])


class TestParseIssuePayload:
    """Tests for issue payload parsing."""

    def test_parse_json_in_code_block(self):
        """Test parsing JSON in markdown code block."""
        body = """
        Some text before

        ```json
        {
            "target_repo": "contextual-arxiv-feed",
            "change_type": "add",
            "target_type": "topic",
            "topic": {"key": "test-topic", "name": "Test"}
        }
        ```

        Some text after
        """
        payload = parse_issue_payload(body)
        assert payload is not None
        assert payload.target_repo == "contextual-arxiv-feed"
        assert payload.change_type == "add"

    def test_parse_raw_json(self):
        """Test parsing raw JSON without code block."""
        body = '{"target_repo": "contextual-arxiv-feed", "change_type": "update", "target_type": "judge", "judge": {"strictness": "high"}}'
        payload = parse_issue_payload(body)
        assert payload is not None
        assert payload.change_type == "update"

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON."""
        body = "This is not JSON"
        payload = parse_issue_payload(body)
        assert payload is None


class TestValidatePayload:
    """Tests for payload validation."""

    def test_valid_topic_add(self, categories: CategoriesConfig):
        """Test valid topic addition."""
        payload = IssuePayload({
            "target_repo": "contextual-arxiv-feed",
            "change_type": "add",
            "target_type": "topic",
            "topic": {
                "key": "new-topic",
                "name": "New Topic",
                "arxiv_categories": ["cs.LG"],
                "keywords": ["test"],
            },
        })
        result = validate_payload(payload, categories)
        assert result.success is True

    def test_invalid_target_repo(self, categories: CategoriesConfig):
        """Test invalid target repo."""
        payload = IssuePayload({
            "target_repo": "wrong-repo",
            "change_type": "add",
            "target_type": "topic",
            "topic": {"key": "test", "name": "Test", "keywords": ["test"]},
        })
        result = validate_payload(payload, categories)
        assert result.success is False
        assert any(e.field == "target_repo" for e in result.errors)

    def test_invalid_change_type(self, categories: CategoriesConfig):
        """Test invalid change type."""
        payload = IssuePayload({
            "target_repo": "contextual-arxiv-feed",
            "change_type": "invalid",
            "target_type": "topic",
            "topic": {"key": "test", "name": "Test", "keywords": ["test"]},
        })
        result = validate_payload(payload, categories)
        assert result.success is False
        assert any(e.field == "change_type" for e in result.errors)

    def test_invalid_topic_key(self, categories: CategoriesConfig):
        """Test invalid topic key pattern."""
        payload = IssuePayload({
            "target_repo": "contextual-arxiv-feed",
            "change_type": "add",
            "target_type": "topic",
            "topic": {
                "key": "Invalid Key",  # Has space and capital
                "name": "Test",
                "keywords": ["test"],
            },
        })
        result = validate_payload(payload, categories)
        assert result.success is False
        assert any("topic.key" in e.field for e in result.errors)

    def test_invalid_arxiv_category(self, categories: CategoriesConfig):
        """Test invalid arXiv category."""
        payload = IssuePayload({
            "target_repo": "contextual-arxiv-feed",
            "change_type": "add",
            "target_type": "topic",
            "topic": {
                "key": "test-topic",
                "name": "Test",
                "arxiv_categories": ["invalid.CAT"],
                "keywords": ["test"],
            },
        })
        result = validate_payload(payload, categories)
        assert result.success is False
        assert any("invalid.CAT" in e.message for e in result.errors)

    def test_missing_keywords_and_phrases(self, categories: CategoriesConfig):
        """Test topic without keywords or phrases."""
        payload = IssuePayload({
            "target_repo": "contextual-arxiv-feed",
            "change_type": "add",
            "target_type": "topic",
            "topic": {
                "key": "test-topic",
                "name": "Test",
                "arxiv_categories": ["cs.LG"],
                # No keywords or phrases!
            },
        })
        result = validate_payload(payload, categories)
        assert result.success is False
        assert any("keyword or phrase" in e.message for e in result.errors)

    def test_judge_change_whitelisted_fields(self, categories: CategoriesConfig):
        """Test judge change with whitelisted fields."""
        payload = IssuePayload({
            "target_repo": "contextual-arxiv-feed",
            "change_type": "update",
            "target_type": "judge",
            "judge": {
                "strictness": "high",
                "model_id": "claude-3-opus-20240229",
            },
        })
        result = validate_payload(payload, categories)
        assert result.success is True

    def test_judge_change_non_whitelisted_field(self, categories: CategoriesConfig):
        """Test judge change with non-whitelisted field."""
        payload = IssuePayload({
            "target_repo": "contextual-arxiv-feed",
            "change_type": "update",
            "target_type": "judge",
            "judge": {
                "strictness": "high",
                "prompt_version": 99,  # Not editable!
            },
        })
        result = validate_payload(payload, categories)
        assert result.success is False
        assert any("not editable" in e.message for e in result.errors)

    def test_invalid_strictness_value(self, categories: CategoriesConfig):
        """Test invalid strictness value."""
        payload = IssuePayload({
            "target_repo": "contextual-arxiv-feed",
            "change_type": "update",
            "target_type": "judge",
            "judge": {
                "strictness": "extreme",  # Not valid!
            },
        })
        result = validate_payload(payload, categories)
        assert result.success is False
        assert any("strictness" in e.field for e in result.errors)
