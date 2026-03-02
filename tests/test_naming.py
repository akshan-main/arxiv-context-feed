"""Tests for document naming and idempotency."""


from contextual_arxiv_feed.contextual.naming import (
    build_document_name,
    build_manifest_name,
    build_prefix,
    document_exists_in_list,
    extract_versions_from_names,
    get_all_versions_prefix,
    parse_document_name,
)


class TestBuildDocumentName:
    """Tests for document name building."""

    def test_basic_name(self):
        """Test basic document name."""
        name = build_document_name("2401.12345", 1)
        assert name == "arxiv:2401.12345v1"

    def test_version_2(self):
        """Test document name with version 2."""
        name = build_document_name("2401.12345", 2)
        assert name == "arxiv:2401.12345v2"

    def test_five_digit_id(self):
        """Test document name with 5-digit ID."""
        name = build_document_name("2401.12345", 1)
        assert name == "arxiv:2401.12345v1"


class TestBuildManifestName:
    """Tests for manifest name building."""

    def test_basic_manifest(self):
        """Test basic manifest name."""
        name = build_manifest_name("2401.12345", 1)
        assert name == "arxiv:2401.12345v1:manifest"

    def test_manifest_version_2(self):
        """Test manifest name with version 2."""
        name = build_manifest_name("2401.12345", 2)
        assert name == "arxiv:2401.12345v2:manifest"


class TestBuildPrefix:
    """Tests for prefix building."""

    def test_with_version(self):
        """Test prefix with specific version."""
        prefix = build_prefix("2401.12345", 1)
        assert prefix == "arxiv:2401.12345v1"

    def test_without_version(self):
        """Test prefix without version (matches all versions)."""
        prefix = build_prefix("2401.12345")
        assert prefix == "arxiv:2401.12345"


class TestParseDocumentName:
    """Tests for document name parsing."""

    def test_parse_pdf_name(self):
        """Test parsing PDF document name."""
        info = parse_document_name("arxiv:2401.12345v1")
        assert info is not None
        assert info.arxiv_id == "2401.12345"
        assert info.version == 1
        assert info.is_manifest is False

    def test_parse_manifest_name(self):
        """Test parsing manifest document name."""
        info = parse_document_name("arxiv:2401.12345v1:manifest")
        assert info is not None
        assert info.arxiv_id == "2401.12345"
        assert info.version == 1
        assert info.is_manifest is True

    def test_parse_version_2(self):
        """Test parsing version 2 document."""
        info = parse_document_name("arxiv:2401.12345v2")
        assert info is not None
        assert info.version == 2

    def test_parse_invalid_name(self):
        """Test parsing invalid document name."""
        info = parse_document_name("invalid-name")
        assert info is None

    def test_parse_no_version(self):
        """Test parsing name without version."""
        info = parse_document_name("arxiv:2401.12345")
        assert info is None

    def test_document_info_properties(self):
        """Test DocumentNameInfo properties."""
        info = parse_document_name("arxiv:2401.12345v1")
        assert info.base_name == "arxiv:2401.12345v1"
        assert info.pdf_name == "arxiv:2401.12345v1"
        assert info.manifest_name == "arxiv:2401.12345v1:manifest"


class TestGetAllVersionsPrefix:
    """Tests for all-versions prefix."""

    def test_prefix(self):
        """Test all-versions prefix."""
        prefix = get_all_versions_prefix("2401.12345")
        assert prefix == "arxiv:2401.12345v"


class TestExtractVersionsFromNames:
    """Tests for version extraction."""

    def test_single_version(self):
        """Test extracting single version."""
        names = ["arxiv:2401.12345v1", "arxiv:2401.12345v1:manifest"]
        versions = extract_versions_from_names(names, "2401.12345")
        assert versions == [1]

    def test_multiple_versions(self):
        """Test extracting multiple versions."""
        names = [
            "arxiv:2401.12345v1",
            "arxiv:2401.12345v2",
            "arxiv:2401.12345v3",
            "arxiv:2401.12345v1:manifest",
        ]
        versions = extract_versions_from_names(names, "2401.12345")
        assert versions == [1, 2, 3]

    def test_different_papers(self):
        """Test filtering by arxiv_id."""
        names = [
            "arxiv:2401.12345v1",
            "arxiv:2401.99999v1",
            "arxiv:2401.12345v2",
        ]
        versions = extract_versions_from_names(names, "2401.12345")
        assert versions == [1, 2]

    def test_no_versions(self):
        """Test when no versions exist."""
        names = ["arxiv:2401.99999v1"]
        versions = extract_versions_from_names(names, "2401.12345")
        assert versions == []


class TestDocumentExistsInList:
    """Tests for document existence check."""

    def test_exists(self):
        """Test document exists."""
        names = ["arxiv:2401.12345v1", "arxiv:2401.12345v2"]
        assert document_exists_in_list(names, "2401.12345", 1) is True
        assert document_exists_in_list(names, "2401.12345", 2) is True

    def test_not_exists(self):
        """Test document not exists."""
        names = ["arxiv:2401.12345v1"]
        assert document_exists_in_list(names, "2401.12345", 2) is False
        assert document_exists_in_list(names, "2401.99999", 1) is False

    def test_empty_list(self):
        """Test with empty list."""
        assert document_exists_in_list([], "2401.12345", 1) is False
