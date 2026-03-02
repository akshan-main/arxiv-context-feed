"""Tests for keyword matcher (Stage 1 filter)."""

import pytest

from contextual_arxiv_feed.config import TopicConfig, TopicsConfig
from contextual_arxiv_feed.matcher import KeywordMatcher


@pytest.fixture
def matcher_config() -> TopicsConfig:
    """Create topics config for matcher tests."""
    return TopicsConfig(
        topics=[
            TopicConfig(
                key="llm-inference",
                name="LLM Inference",
                keywords=["inference", "optimization", "quantization"],
                phrases=["large language model", "model serving"],
                negatives=["survey", "literature review"],
                enabled=True,
            ),
            TopicConfig(
                key="rag",
                name="RAG",
                keywords=["retrieval", "rag"],
                phrases=["retrieval augmented"],
                negatives=[],
                enabled=True,
            ),
            TopicConfig(
                key="disabled",
                name="Disabled Topic",
                keywords=["disabled"],
                phrases=[],
                negatives=[],
                enabled=False,
            ),
        ]
    )


@pytest.fixture
def matcher(matcher_config: TopicsConfig) -> KeywordMatcher:
    """Create KeywordMatcher for tests."""
    return KeywordMatcher(matcher_config)


class TestKeywordMatcher:
    """Tests for KeywordMatcher."""

    def test_keyword_match_stemming(self, matcher: KeywordMatcher):
        """Test that stemming works for keyword matching."""
        # "optimizing" should match "optimization" due to stemming
        result = matcher.match("Optimizing LLM Inference", "")
        assert result.passes_stage1
        assert "llm-inference" in result.matched_topics

    def test_phrase_match(self, matcher: KeywordMatcher):
        """Test multi-word phrase matching."""
        result = matcher.match(
            "Efficient Model Serving for Large Language Models",
            "We present a system for large language model deployment."
        )
        assert result.passes_stage1
        assert "llm-inference" in result.matched_topics
        assert "large language model" in result.matched_phrases.get("llm-inference", [])

    def test_negative_keyword_exclusion(self, matcher: KeywordMatcher):
        """Test that negative keywords exclude topics."""
        result = matcher.match(
            "A Survey of LLM Inference Optimization",
            "This survey covers recent advances in optimization."
        )
        # Should be excluded due to "survey"
        assert "llm-inference" in result.excluded_by

    def test_negative_phrase_exclusion(self, matcher: KeywordMatcher):
        """Test that negative phrases exclude topics."""
        result = matcher.match(
            "Optimization Techniques: A Literature Review",
            "This literature review covers optimization methods."
        )
        # Should be excluded due to "literature review"
        assert "llm-inference" in result.excluded_by

    def test_multiple_topic_match(self, matcher: KeywordMatcher):
        """Test matching multiple topics."""
        result = matcher.match(
            "RAG Optimization for Large Language Models",
            "We optimize retrieval augmented generation for LLMs."
        )
        assert result.passes_stage1
        # Could match both llm-inference (optimization) and rag (retrieval augmented)
        assert len(result.matched_topics) >= 1

    def test_no_match(self, matcher: KeywordMatcher):
        """Test when no topics match."""
        result = matcher.match(
            "Understanding Quantum Computing",
            "This paper explores quantum algorithms."
        )
        assert not result.passes_stage1
        assert result.matched_topics == []

    def test_disabled_topic_not_matched(self, matcher: KeywordMatcher):
        """Test that disabled topics are not matched."""
        result = matcher.match(
            "This paper is about disabled topics",
            "Testing disabled functionality."
        )
        # "disabled" keyword should not match because topic is disabled
        assert "disabled" not in result.matched_topics

    def test_case_insensitive_keyword(self, matcher: KeywordMatcher):
        """Test case insensitive keyword matching."""
        result = matcher.match("INFERENCE OPTIMIZATION", "")
        assert result.passes_stage1

    def test_case_insensitive_phrase(self, matcher: KeywordMatcher):
        """Test case insensitive phrase matching."""
        result = matcher.match("LARGE LANGUAGE MODEL serving", "")
        assert result.passes_stage1

    def test_combined_title_and_abstract(self, matcher: KeywordMatcher):
        """Test matching against combined title and abstract."""
        # Keyword only in abstract
        result = matcher.match(
            "A Novel Approach to Neural Networks",
            "We apply quantization techniques to improve efficiency."
        )
        assert result.passes_stage1
        assert "llm-inference" in result.matched_topics

    def test_get_topic_names(self, matcher: KeywordMatcher):
        """Test getting human-readable topic names."""
        names = matcher.get_topic_names(["llm-inference", "rag"])
        assert "LLM Inference" in names
        assert "RAG" in names

    def test_empty_input(self, matcher: KeywordMatcher):
        """Test with empty inputs."""
        result = matcher.match("", "")
        assert not result.passes_stage1
