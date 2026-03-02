"""Stage 1 keyword/phrase matching for cost-efficient filtering.

This is a COST GATE ONLY - it exists to reduce LLM API calls, not to
judge paper quality or relevance. Papers that fail Stage 1 are simply
not sent to the expensive LLM judge.

Features:
- Single-word keyword matching with stemming
- Multi-word phrase matching (substring)
- Optional negative keywords/phrases for exclusion
- Regex tokenization (no NLTK data downloads needed)
- NLTK PorterStemmer (algorithmic, no data files)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from nltk.stem import PorterStemmer

from contextual_arxiv_feed.config import TopicsConfig

logger = logging.getLogger(__name__)

# Regex for tokenization: extract alphabetic words
WORD_PATTERN = re.compile(r"\b[a-zA-Z]+\b")


@dataclass
class MatchResult:
    """Result of keyword matching for a paper."""

    matched: bool
    matched_topics: list[str] = field(default_factory=list)
    matched_keywords: dict[str, list[str]] = field(default_factory=dict)
    matched_phrases: dict[str, list[str]] = field(default_factory=dict)
    excluded_by: dict[str, list[str]] = field(default_factory=dict)

    @property
    def passes_stage1(self) -> bool:
        """Whether paper passes Stage 1 filter (has at least one topic match)."""
        return self.matched and len(self.matched_topics) > 0


class KeywordMatcher:
    """Matches paper titles/abstracts against topic keywords and phrases.

    This is Stage 1 filtering - a cost gate to reduce LLM calls.
    It is NOT a quality or relevance judgment.
    """

    def __init__(self, topics_config: TopicsConfig):
        """Initialize matcher with topic configuration.

        Args:
            topics_config: Topics configuration with keywords/phrases.
        """
        self._topics = topics_config.get_enabled_topics()
        self._stemmer = PorterStemmer()
        self._preprocess_topics()

    def _preprocess_topics(self) -> None:
        """Preprocess topic keywords for efficient matching."""
        self._topic_data: dict[str, dict] = {}

        for topic in self._topics:
            # Stem keywords for matching
            stemmed_keywords = set()
            for kw in topic.keywords:
                stemmed = self._stem_word(kw.lower())
                stemmed_keywords.add(stemmed)

            # Store phrases as lowercase for substring matching
            phrases = [p.lower() for p in topic.phrases]

            # Process negatives (both words and phrases)
            stemmed_negatives = set()
            negative_phrases = []
            for neg in topic.negatives:
                neg_lower = neg.lower()
                if " " in neg_lower:
                    # Multi-word negative phrase
                    negative_phrases.append(neg_lower)
                else:
                    # Single-word negative
                    stemmed_negatives.add(self._stem_word(neg_lower))

            self._topic_data[topic.key] = {
                "topic": topic,
                "stemmed_keywords": stemmed_keywords,
                "phrases": phrases,
                "stemmed_negatives": stemmed_negatives,
                "negative_phrases": negative_phrases,
            }

        logger.info(f"Preprocessed {len(self._topic_data)} topics for matching")

    def _stem_word(self, word: str) -> str:
        """Stem a single word using PorterStemmer."""
        return self._stemmer.stem(word)

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase words using regex.

        No NLTK data downloads needed - pure regex.
        """
        return WORD_PATTERN.findall(text.lower())

    def _stem_tokens(self, tokens: list[str]) -> set[str]:
        """Stem a list of tokens."""
        return {self._stem_word(token) for token in tokens}

    def match(self, title: str, abstract_snippet: str = "") -> MatchResult:
        """Match paper against all enabled topics.

        Args:
            title: Paper title.
            abstract_snippet: RSS abstract snippet (may be partial).

        Returns:
            MatchResult indicating which topics matched (if any).
        """
        # Combine title and abstract for matching
        combined_text = f"{title} {abstract_snippet}".lower()

        # Tokenize and stem for keyword matching
        tokens = self._tokenize(combined_text)
        stemmed_tokens = self._stem_tokens(tokens)

        result = MatchResult(matched=False)

        for topic_key, data in self._topic_data.items():
            topic_match = self._match_topic(
                topic_key, data, combined_text, stemmed_tokens
            )

            if topic_match["matched"] and not topic_match["excluded"]:
                result.matched = True
                result.matched_topics.append(topic_key)

                if topic_match["keywords"]:
                    result.matched_keywords[topic_key] = topic_match["keywords"]
                if topic_match["phrases"]:
                    result.matched_phrases[topic_key] = topic_match["phrases"]

            if topic_match["excluded"]:
                result.excluded_by[topic_key] = topic_match["excluded_by"]

        return result

    def _match_topic(
        self,
        topic_key: str,
        data: dict,
        combined_text: str,
        stemmed_tokens: set[str],
    ) -> dict:
        """Match against a single topic.

        Returns dict with:
            matched: bool - whether any keyword/phrase matched
            excluded: bool - whether excluded by negative
            keywords: list - matched keywords
            phrases: list - matched phrases
            excluded_by: list - negatives that triggered exclusion
        """
        result = {
            "matched": False,
            "excluded": False,
            "keywords": [],
            "phrases": [],
            "excluded_by": [],
        }

        # Check for negative matches first
        for neg in data["stemmed_negatives"]:
            if neg in stemmed_tokens:
                result["excluded"] = True
                result["excluded_by"].append(neg)

        for neg_phrase in data["negative_phrases"]:
            if neg_phrase in combined_text:
                result["excluded"] = True
                result["excluded_by"].append(neg_phrase)

        if result["excluded"]:
            logger.debug(f"Paper excluded from {topic_key} by: {result['excluded_by']}")
            return result

        # Check keyword matches (stemmed)
        for kw in data["stemmed_keywords"]:
            if kw in stemmed_tokens:
                result["matched"] = True
                result["keywords"].append(kw)

        # Check phrase matches (substring)
        for phrase in data["phrases"]:
            if phrase in combined_text:
                result["matched"] = True
                result["phrases"].append(phrase)

        return result

    def get_topic_names(self, topic_keys: list[str]) -> list[str]:
        """Get human-readable topic names from keys."""
        names = []
        for key in topic_keys:
            if key in self._topic_data:
                names.append(self._topic_data[key]["topic"].name)
        return names


def create_matcher_from_config(topics_config: TopicsConfig) -> KeywordMatcher:
    """Factory function to create KeywordMatcher from config.

    Args:
        topics_config: Topics configuration.

    Returns:
        Configured KeywordMatcher instance.
    """
    return KeywordMatcher(topics_config)
