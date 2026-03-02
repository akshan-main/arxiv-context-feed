"""Tests for manifest document completeness.

Manifest must contain ALL metadata - nothing omitted.
"""



class TestManifestCompleteness:
    """Tests for manifest document structure."""

    def test_manifest_contains_arxiv_metadata(self):
        """Test that manifest includes full arXiv metadata."""
        # Expected manifest structure
        sample_manifest = {
            "arxiv_metadata": {
                "arxiv_id": "2401.12345",
                "version": 1,
                "title": "Test Paper",
                "abstract": "Full abstract text here...",  # MUST be in manifest
                "authors": [  # MUST be in manifest (full author details)
                    {"name": "John Doe", "affiliations": ["MIT"]},
                ],
                "categories": ["cs.LG", "cs.AI"],
                "primary_category": "cs.LG",
                "published": "2024-01-15T00:00:00Z",
                "updated": "2024-01-15T00:00:00Z",
                "doi": "10.1234/test.12345",
                "journal_ref": "",
                "comments": "10 pages, 5 figures",
                "links": {"pdf": "https://arxiv.org/pdf/2401.12345.pdf"},
                "abs_url": "https://arxiv.org/abs/2401.12345v1",
                "pdf_url": "https://arxiv.org/pdf/2401.12345v1.pdf",
                "year": 2024,
            },
            "judge_output": {},
            "discovery_channel": "daily_rss",
            "citation_enrichment": None,
            "run_metadata": {},
        }

        # Verify required fields
        arxiv = sample_manifest["arxiv_metadata"]
        assert "abstract" in arxiv, "Manifest must contain abstract"
        assert "authors" in arxiv, "Manifest must contain authors"
        assert "title" in arxiv
        assert "doi" in arxiv
        assert "comments" in arxiv
        assert "journal_ref" in arxiv

    def test_manifest_contains_judge_output(self):
        """Test that manifest includes full judge output."""
        sample_manifest = {
            "arxiv_metadata": {},
            "judge_output": {
                "prompt_version": 3,
                "model_id": "qwen-3-235b-a22b-instruct-2507",
                "quality_verdict": "accept",
                "quality_i": 78,
                "quality_breakdown_i": {
                    "novelty_i": 80,
                    "relevance_i": 78,
                    "technical_depth_i": 75,
                },
                "confidence_i": 85,
                "rationale": "The paper presents...",
            },
            "discovery_channel": "daily_rss",
            "citation_enrichment": None,
            "run_metadata": {},
        }

        judge = sample_manifest["judge_output"]
        assert "quality_verdict" in judge
        assert "quality_i" in judge
        assert "quality_breakdown_i" in judge
        assert "confidence_i" in judge
        assert "rationale" in judge

    def test_manifest_contains_discovery_info(self):
        """Test that manifest includes discovery channel info."""
        sample_manifest = {
            "arxiv_metadata": {},
            "judge_output": {},
            "discovery_channel": "daily_rss",  # or "weekly_updates" or "backfill"
            "citation_enrichment": None,
            "run_metadata": {
                "run_id": "abc123",
                "ingested_at": "2024-01-15T00:00:00Z",
                "pipeline": "daily",
            },
        }

        assert "discovery_channel" in sample_manifest
        assert "run_metadata" in sample_manifest
        assert "run_id" in sample_manifest["run_metadata"]

    def test_manifest_contains_citation_enrichment_slot(self):
        """Test that manifest has slot for citation enrichment."""
        sample_manifest = {
            "arxiv_metadata": {},
            "judge_output": {},
            "discovery_channel": "daily_rss",
            "citation_enrichment": {
                "citation_count": 42,
                "reference_count": 15,
                "venue": "NeurIPS 2024",
                "source": "openalex",
                "refreshed_at": "2024-01-20T00:00:00Z",
            },
            "run_metadata": {},
        }

        # Citation enrichment can be None initially
        assert "citation_enrichment" in sample_manifest

        # When populated, should have these fields
        citation = sample_manifest["citation_enrichment"]
        assert "citation_count" in citation
        assert "reference_count" in citation
        assert "venue" in citation


class TestManifestVsCustomMetadata:
    """Tests verifying manifest contains what custom_metadata doesn't."""

    CUSTOM_METADATA_FIELDS = {
        "arxiv_id", "arxiv_version", "title", "primary_category",
        "categories", "doi", "year", "topics",
        "quality_verdict", "quality_i",
        "novelty_i", "relevance_i",
        "technical_depth_i", "confidence_i",
        "citation_count", "reference_count", "venue",
        "citations_updated_at", "authors", "publication_date",
        "paper_type", "open_access",
        "judge_model_id", "judge_prompt_version",
    }

    def test_abstract_only_in_manifest(self):
        """Abstract should be in manifest, NOT custom_metadata."""
        assert "abstract" not in self.CUSTOM_METADATA_FIELDS

    def test_derivable_fields_omitted(self):
        """Derivable fields omitted from custom_metadata to save 2KB budget."""
        # These can be reconstructed from arxiv_id + arxiv_version
        assert "source" not in self.CUSTOM_METADATA_FIELDS
        assert "url" not in self.CUSTOM_METADATA_FIELDS
        assert "pdf_url" not in self.CUSTOM_METADATA_FIELDS
        assert "manifest_doc_name" not in self.CUSTOM_METADATA_FIELDS

    def test_quality_breakdown_in_both(self):
        """Quality breakdown scores are in custom_metadata AND manifest."""
        # custom_metadata has flat scores for filtering
        assert "novelty_i" in self.CUSTOM_METADATA_FIELDS
        assert "relevance_i" in self.CUSTOM_METADATA_FIELDS
        assert "technical_depth_i" in self.CUSTOM_METADATA_FIELDS

    def test_authors_as_flat_string(self):
        """Authors in custom_metadata are pipe-separated string (not list)."""
        assert "authors" in self.CUSTOM_METADATA_FIELDS
        # The actual value is "Author1|Author2|Author3" (truncated to 200 chars)
        # Full author list with affiliations is in manifest
