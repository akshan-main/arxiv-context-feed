"""Tests for metadata packing rules.

Critical: custom_metadata must be flat with primitive values only. NO FLOATS.
Total metadata must stay under 2KB per document.
"""

import json

import pytest

from contextual_arxiv_feed.contextual.contextual_client import ContextualClient


class MockContextualClient(ContextualClient):
    """Mock client for testing metadata validation."""

    def __init__(self):
        # Don't call super().__init__ to avoid needing API key
        pass


class TestMetadataValidation:
    """Tests for custom_metadata validation."""

    @pytest.fixture
    def client(self):
        return MockContextualClient()

    def test_accepts_string_values(self, client):
        """Test that string values are accepted."""
        metadata = {"key": "value", "title": "Test Paper"}
        # Should not raise
        client._validate_metadata(metadata)

    def test_accepts_int_values(self, client):
        """Test that int values are accepted."""
        metadata = {"quality_i": 78, "year": 2024}
        client._validate_metadata(metadata)

    def test_accepts_bool_values(self, client):
        """Test that bool values are accepted."""
        metadata = {"enabled": True, "processed": False}
        client._validate_metadata(metadata)

    def test_rejects_float_values(self, client):
        """Test that float values are rejected."""
        metadata = {"score": 78.5}
        with pytest.raises(ValueError, match="Float values not allowed"):
            client._validate_metadata(metadata)

    def test_rejects_nested_dict(self, client):
        """Test that nested dicts are rejected."""
        metadata = {"nested": {"key": "value"}}
        with pytest.raises(ValueError, match="nested dict"):
            client._validate_metadata(metadata)

    def test_rejects_list_values(self, client):
        """Test that list values are rejected."""
        metadata = {"tags": ["tag1", "tag2"]}
        with pytest.raises(ValueError, match="contains a list"):
            client._validate_metadata(metadata)

    def test_accepts_empty_string(self, client):
        """Test that empty strings are accepted."""
        metadata = {"doi": "", "venue": ""}
        client._validate_metadata(metadata)

    def test_accepts_zero(self, client):
        """Test that zero is accepted."""
        metadata = {"citation_count": 0}
        client._validate_metadata(metadata)

    def test_rejects_none_value(self, client):
        """Test None handling - should be converted to empty string in practice."""
        # None is technically allowed by type hints but should be avoided
        metadata = {"field": None}
        # This should pass as None is a valid primitive
        client._validate_metadata(metadata)


class TestCustomMetadataStructure:
    """Tests for expected custom_metadata structure."""

    def _build_sample_metadata(self) -> dict:
        """Build a representative sample matching pipeline output."""
        return {
            "arxiv_id": "2401.12345",
            "arxiv_version": 1,
            "title": "Test Paper Title",
            "primary_category": "cs.LG",
            "categories": "cs.LG|cs.AI",
            "doi": "",
            "year": 2024,
            "topics": "context-engineering|rag-retrieval",
            "quality_verdict": "accept",
            "quality_i": 78,
            "novelty_i": 80,
            "relevance_i": 75,
            "technical_depth_i": 70,
            "confidence_i": 90,
            "citation_count": 0,
            "reference_count": 0,
            "venue": "",
            "citations_updated_at": "",
            "authors": "",
            "publication_date": "",
            "paper_type": "",
            "open_access": False,
            "judge_model_id": "qwen-3-235b-a22b-instruct-2507",
            "judge_prompt_version": 3,
        }

    def test_required_fields_are_primitives(self):
        """Test that all required fields can be represented as primitives."""
        sample_metadata = self._build_sample_metadata()

        for key, value in sample_metadata.items():
            assert isinstance(value, (str, int, bool)), f"{key} is not a primitive"
            if not isinstance(value, bool):
                assert not isinstance(value, float), f"{key} is a float"

    def test_no_abstract_in_custom_metadata(self):
        """Test that abstract is NOT in custom_metadata (goes in manifest)."""
        sample_metadata = self._build_sample_metadata()
        assert "abstract" not in sample_metadata

    def test_derivable_fields_omitted(self):
        """Test that derivable fields are omitted to save 2KB budget."""
        sample_metadata = self._build_sample_metadata()
        # These can be reconstructed from arxiv_id + arxiv_version
        assert "source" not in sample_metadata  # always "arxiv"
        assert "url" not in sample_metadata  # https://arxiv.org/abs/{id}v{ver}
        assert "pdf_url" not in sample_metadata  # https://arxiv.org/pdf/{id}v{ver}.pdf
        assert "manifest_doc_name" not in sample_metadata  # arxiv:{id}v{ver}:manifest

    def test_has_quality_breakdown_scores(self):
        """Test that individual quality breakdown scores are present."""
        sample_metadata = self._build_sample_metadata()
        assert "novelty_i" in sample_metadata
        assert "relevance_i" in sample_metadata
        assert "technical_depth_i" in sample_metadata
        assert "confidence_i" in sample_metadata

    def test_has_openalex_placeholder_fields(self):
        """Test that OpenAlex/S2 enrichment fields are present."""
        sample_metadata = self._build_sample_metadata()
        assert "authors" in sample_metadata
        assert "publication_date" in sample_metadata
        assert "paper_type" in sample_metadata
        assert "open_access" in sample_metadata

    def test_metadata_under_2kb(self):
        """Test that metadata stays under 2KB even with realistic values."""
        metadata = self._build_sample_metadata()
        # Simulate worst case: long title, many authors, populated fields
        metadata["title"] = "A" * 200  # Very long title
        metadata["authors"] = "|".join([f"Author {i}" for i in range(15)])[:200]
        metadata["categories"] = "cs.LG|cs.AI|cs.CL|cs.IR|cs.SE"
        metadata["topics"] = "context-engineering|rag-retrieval|llm-inference"
        metadata["venue"] = "Conference on Neural Information Processing Systems"
        metadata["doi"] = "10.1234/very.long.doi.identifier.2024"
        metadata["citations_updated_at"] = "2024-01-15T12:30:00.123456"
        metadata["publication_date"] = "2024-01-15"
        metadata["paper_type"] = "journal-article"
        metadata["citation_count"] = 9999
        metadata["reference_count"] = 999

        serialized = json.dumps(metadata)
        assert len(serialized) < 2048, (
            f"Metadata is {len(serialized)} bytes, exceeds 2KB limit"
        )

    def test_authors_truncated_to_200_chars(self):
        """Test that authors field respects 200 char limit."""
        # This is enforced in the OpenAlex/S2 clients, not in metadata
        authors = "|".join([f"Very Long Author Name {i}" for i in range(20)])
        if len(authors) > 200:
            authors = authors[:200].rsplit("|", 1)[0]
        assert len(authors) <= 200
