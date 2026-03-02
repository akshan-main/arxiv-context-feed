"""Document naming conventions for Contextual AI Datastore.

Naming format:
- PDF document: "arxiv:{arxiv_id}v{version}"
- Manifest document: "arxiv:{arxiv_id}v{version}:manifest"

Idempotency:
- Prefix-based: check if "arxiv:{arxiv_id}v{version}" exists before ingest
- Version-aware: v1, v2, v3 are separate documents (v2 can coexist with v1)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Document name patterns
PDF_DOC_PATTERN = re.compile(r"^arxiv:(\d{4}\.\d{4,5})v(\d+)$")
MANIFEST_DOC_PATTERN = re.compile(r"^arxiv:(\d{4}\.\d{4,5})v(\d+):manifest$")


@dataclass
class DocumentNameInfo:
    """Parsed document name information."""

    arxiv_id: str
    version: int
    is_manifest: bool

    @property
    def base_name(self) -> str:
        """Base name without manifest suffix."""
        return f"arxiv:{self.arxiv_id}v{self.version}"

    @property
    def pdf_name(self) -> str:
        """PDF document name."""
        return self.base_name

    @property
    def manifest_name(self) -> str:
        """Manifest document name."""
        return f"{self.base_name}:manifest"


def build_document_name(arxiv_id: str, version: int) -> str:
    """Build PDF document name.

    Args:
        arxiv_id: arXiv paper ID (e.g., "2401.12345").
        version: Paper version number.

    Returns:
        Document name in format "arxiv:{arxiv_id}v{version}".
    """
    return f"arxiv:{arxiv_id}v{version}"


def build_manifest_name(arxiv_id: str, version: int) -> str:
    """Build manifest document name.

    Args:
        arxiv_id: arXiv paper ID (e.g., "2401.12345").
        version: Paper version number.

    Returns:
        Document name in format "arxiv:{arxiv_id}v{version}:manifest".
    """
    return f"arxiv:{arxiv_id}v{version}:manifest"


def build_prefix(arxiv_id: str, version: int | None = None) -> str:
    """Build prefix for document listing.

    Args:
        arxiv_id: arXiv paper ID.
        version: Optional version number. If None, matches all versions.

    Returns:
        Prefix string for filtering documents.
    """
    if version is not None:
        return f"arxiv:{arxiv_id}v{version}"
    return f"arxiv:{arxiv_id}"


def parse_document_name(name: str) -> DocumentNameInfo | None:
    """Parse a document name to extract components.

    Args:
        name: Document name.

    Returns:
        DocumentNameInfo or None if name doesn't match expected format.
    """
    # Try manifest pattern first
    match = MANIFEST_DOC_PATTERN.match(name)
    if match:
        return DocumentNameInfo(
            arxiv_id=match.group(1),
            version=int(match.group(2)),
            is_manifest=True,
        )

    # Try PDF pattern
    match = PDF_DOC_PATTERN.match(name)
    if match:
        return DocumentNameInfo(
            arxiv_id=match.group(1),
            version=int(match.group(2)),
            is_manifest=False,
        )

    return None


def get_all_versions_prefix(arxiv_id: str) -> str:
    """Get prefix that matches all versions of a paper.

    Args:
        arxiv_id: arXiv paper ID.

    Returns:
        Prefix that matches all versions.
    """
    return f"arxiv:{arxiv_id}v"


def extract_versions_from_names(names: list[str], arxiv_id: str) -> list[int]:
    """Extract version numbers for a given arxiv_id from document names.

    Args:
        names: List of document names.
        arxiv_id: arXiv paper ID to filter for.

    Returns:
        Sorted list of version numbers found.
    """
    versions = set()
    for name in names:
        info = parse_document_name(name)
        if info and info.arxiv_id == arxiv_id and not info.is_manifest:
            versions.add(info.version)
    return sorted(versions)


def document_exists_in_list(names: list[str], arxiv_id: str, version: int) -> bool:
    """Check if a specific document exists in a list of names.

    Args:
        names: List of document names.
        arxiv_id: arXiv paper ID.
        version: Version number.

    Returns:
        True if document exists.
    """
    target = build_document_name(arxiv_id, version)
    return target in names
