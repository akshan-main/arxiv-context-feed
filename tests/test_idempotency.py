"""Tests for idempotency via prefix-based document checks.

Idempotency rules:
- Check prefix "arxiv:{arxiv_id}v{version}" before ingest
- v1, v2, v3 are separate documents
- v2 can coexist with v1
"""



from contextual_arxiv_feed.contextual.naming import (
    build_document_name,
    build_prefix,
    document_exists_in_list,
)


class MockContextualClient:
    """Mock Contextual client for testing."""

    def __init__(self, existing_docs: list[str]):
        self._existing_docs = existing_docs

    def list_documents(self, prefix: str = "", limit: int = 1000) -> list[str]:
        """Return documents matching prefix."""
        if not prefix:
            return self._existing_docs
        return [d for d in self._existing_docs if d.startswith(prefix)]

    def document_exists(self, arxiv_id: str, version: int) -> bool:
        """Check if document exists."""
        prefix = build_prefix(arxiv_id, version)
        docs = self.list_documents(prefix=prefix, limit=1)
        target = build_document_name(arxiv_id, version)
        return target in docs


class TestIdempotencyPrefixCheck:
    """Tests for prefix-based idempotency."""

    def test_skip_existing_v1(self):
        """Test that existing v1 is skipped."""
        client = MockContextualClient([
            "arxiv:2401.12345v1",
            "arxiv:2401.12345v1:manifest",
        ])

        assert client.document_exists("2401.12345", 1) is True

    def test_allow_new_paper(self):
        """Test that new paper is allowed."""
        client = MockContextualClient([
            "arxiv:2401.11111v1",  # Different paper
        ])

        assert client.document_exists("2401.12345", 1) is False

    def test_allow_v2_when_v1_exists(self):
        """Test that v2 can be ingested when v1 exists."""
        client = MockContextualClient([
            "arxiv:2401.12345v1",
            "arxiv:2401.12345v1:manifest",
        ])

        # v2 should NOT exist
        assert client.document_exists("2401.12345", 2) is False

    def test_skip_existing_v2(self):
        """Test that existing v2 is skipped."""
        client = MockContextualClient([
            "arxiv:2401.12345v1",
            "arxiv:2401.12345v2",
        ])

        assert client.document_exists("2401.12345", 2) is True

    def test_multiple_versions_coexist(self):
        """Test that multiple versions can coexist."""
        client = MockContextualClient([
            "arxiv:2401.12345v1",
            "arxiv:2401.12345v1:manifest",
            "arxiv:2401.12345v2",
            "arxiv:2401.12345v2:manifest",
            "arxiv:2401.12345v3",
            "arxiv:2401.12345v3:manifest",
        ])

        # All versions exist
        assert client.document_exists("2401.12345", 1) is True
        assert client.document_exists("2401.12345", 2) is True
        assert client.document_exists("2401.12345", 3) is True

        # v4 doesn't exist
        assert client.document_exists("2401.12345", 4) is False


class TestPrefixMatching:
    """Tests for prefix matching behavior."""

    def test_exact_prefix_match(self):
        """Test that prefix matching is exact."""
        existing = [
            "arxiv:2401.12345v1",
            "arxiv:2401.12345v10",  # v10, not v1!
            "arxiv:2401.123456v1",  # Different ID
        ]

        # Check v1 specifically
        assert document_exists_in_list(existing, "2401.12345", 1) is True

        # v10 is different from v1
        assert document_exists_in_list(existing, "2401.12345", 10) is True

        # Different paper
        assert document_exists_in_list(existing, "2401.123456", 1) is True
        assert document_exists_in_list(existing, "2401.12345", 2) is False

    def test_manifest_not_confused_with_pdf(self):
        """Test that manifest docs don't confuse PDF existence check."""
        existing = [
            "arxiv:2401.12345v1:manifest",  # Only manifest, no PDF
        ]

        # PDF document name doesn't match manifest
        assert document_exists_in_list(existing, "2401.12345", 1) is False


class TestVersionAwareness:
    """Tests for version-aware behavior."""

    def test_v1_v2_are_separate(self):
        """Test that v1 and v2 are treated as separate documents."""
        name_v1 = build_document_name("2401.12345", 1)
        name_v2 = build_document_name("2401.12345", 2)

        assert name_v1 != name_v2
        assert name_v1 == "arxiv:2401.12345v1"
        assert name_v2 == "arxiv:2401.12345v2"

    def test_prefix_can_match_all_versions(self):
        """Test that prefix without version matches all versions."""
        prefix_all = build_prefix("2401.12345")  # No version
        prefix_v1 = build_prefix("2401.12345", 1)

        # prefix_all should be shorter
        assert len(prefix_all) < len(prefix_v1)
        assert prefix_v1.startswith(prefix_all)


class TestIdempotencyFlow:
    """Integration-style tests for idempotency flow."""

    def test_ingest_flow_checks_before_download(self):
        """Test that we check existence before downloading PDF."""
        # This is a logical test - actual implementation does this
        existing = ["arxiv:2401.12345v1"]

        # Simulate the check that happens before download
        should_ingest = not document_exists_in_list(existing, "2401.12345", 1)
        assert should_ingest is False  # Should skip

    def test_ingest_flow_allows_new_version(self):
        """Test that new version can be ingested."""
        existing = ["arxiv:2401.12345v1"]

        # v2 should be allowed
        should_ingest = not document_exists_in_list(existing, "2401.12345", 2)
        assert should_ingest is True
