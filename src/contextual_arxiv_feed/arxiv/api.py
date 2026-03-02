"""arXiv API client for fetching full paper metadata.

Uses the arXiv API (https://arxiv.org/help/api) to retrieve complete
metadata including full abstracts, all authors, DOI, etc.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import urlencode

import httpx

from contextual_arxiv_feed.arxiv.throttle import ArxivThrottle, arxiv_retry, check_response_status

logger = logging.getLogger(__name__)

# arXiv API endpoint
ARXIV_API_URL = "http://export.arxiv.org/api/query"

# XML namespaces used by arXiv API
NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

# Regex to extract arXiv ID from various formats
ARXIV_ID_PATTERN = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


@dataclass
class Author:
    """Paper author information."""

    name: str
    affiliations: list[str] = field(default_factory=list)


@dataclass
class ArxivMetadata:
    """Complete arXiv paper metadata."""

    arxiv_id: str  # e.g., "2401.12345"
    version: int  # e.g., 1
    title: str
    abstract: str
    authors: list[Author]
    categories: list[str]
    primary_category: str
    published: datetime | None
    updated: datetime | None
    doi: str  # Empty string if not available
    journal_ref: str
    comments: str
    links: dict[str, str]  # type -> URL mapping

    @property
    def id_with_version(self) -> str:
        """Return arxiv_id with version."""
        return f"{self.arxiv_id}v{self.version}"

    @property
    def abs_url(self) -> str:
        """Return arXiv abstract page URL."""
        return f"https://arxiv.org/abs/{self.id_with_version}"

    @property
    def pdf_url(self) -> str:
        """Return PDF download URL."""
        return f"https://arxiv.org/pdf/{self.id_with_version}.pdf"

    @property
    def year(self) -> int:
        """Extract publication year. Returns 0 if unknown."""
        if self.published:
            return self.published.year
        # Try to extract from arxiv_id (format: YYMM.NNNNN)
        try:
            year_prefix = int(self.arxiv_id[:2])
            # Handle 2-digit year: 00-99 -> 2000-2099
            return 2000 + year_prefix
        except (ValueError, IndexError):
            return 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "arxiv_id": self.arxiv_id,
            "version": self.version,
            "title": self.title,
            "abstract": self.abstract,
            "authors": [
                {"name": a.name, "affiliations": a.affiliations} for a in self.authors
            ],
            "categories": self.categories,
            "primary_category": self.primary_category,
            "published": self.published.isoformat() if self.published else None,
            "updated": self.updated.isoformat() if self.updated else None,
            "doi": self.doi,
            "journal_ref": self.journal_ref,
            "comments": self.comments,
            "links": self.links,
            "abs_url": self.abs_url,
            "pdf_url": self.pdf_url,
            "year": self.year,
        }


class ArxivAPI:
    """Client for arXiv API."""

    def __init__(self, throttle: ArxivThrottle | None = None):
        """Initialize API client.

        Args:
            throttle: Rate limiter. Creates default if None.
        """
        self._throttle = throttle or ArxivThrottle()
        self._client = httpx.Client(timeout=60.0)

    def close(self) -> None:
        """Close HTTP client."""
        self._client.close()

    def __enter__(self) -> ArxivAPI:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @arxiv_retry
    def fetch_by_id(self, arxiv_id: str) -> ArxivMetadata | None:
        """Fetch metadata for a single paper by ID.

        Args:
            arxiv_id: arXiv ID (with or without version).

        Returns:
            ArxivMetadata or None if not found.
        """
        # Normalize ID (remove version if present for query)
        match = ARXIV_ID_PATTERN.search(arxiv_id)
        if not match:
            logger.error(f"Invalid arXiv ID format: {arxiv_id}")
            return None

        query_params = {
            "id_list": arxiv_id,
            "max_results": 1,
        }

        results = self._query(query_params)
        return results[0] if results else None

    @arxiv_retry
    def fetch_by_ids(self, arxiv_ids: list[str]) -> list[ArxivMetadata]:
        """Fetch metadata for multiple papers by ID.

        Args:
            arxiv_ids: List of arXiv IDs.

        Returns:
            List of ArxivMetadata objects (may be fewer than requested).
        """
        if not arxiv_ids:
            return []

        # API accepts comma-separated IDs
        query_params = {
            "id_list": ",".join(arxiv_ids),
            "max_results": len(arxiv_ids),
        }

        return self._query(query_params)

    @arxiv_retry
    def search_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        categories: list[str] | None = None,
        max_results: int = 1000,
    ) -> list[ArxivMetadata]:
        """Search for papers updated within a date range.

        Args:
            start_date: Start of date range.
            end_date: End of date range.
            categories: Optional category filter.
            max_results: Maximum results to return.

        Returns:
            List of ArxivMetadata objects.
        """
        # Build search query
        date_query = f"lastUpdatedDate:[{start_date.strftime('%Y%m%d')} TO {end_date.strftime('%Y%m%d')}]"

        if categories:
            cat_query = " OR ".join(f"cat:{cat}" for cat in categories)
            search_query = f"({cat_query}) AND {date_query}"
        else:
            search_query = date_query

        query_params = {
            "search_query": search_query,
            "sortBy": "lastUpdatedDate",
            "sortOrder": "descending",
            "max_results": max_results,
        }

        return self._query(query_params)

    def _query(self, params: dict[str, Any]) -> list[ArxivMetadata]:
        """Execute API query.

        Args:
            params: Query parameters.

        Returns:
            List of ArxivMetadata objects.
        """
        url = f"{ARXIV_API_URL}?{urlencode(params)}"
        logger.debug(f"arXiv API query: {url}")

        self._throttle.sync_wait()

        response = self._client.get(url)
        check_response_status(response.status_code, url)

        if response.status_code != 200:
            logger.error(f"arXiv API error: {response.status_code}")
            return []

        return self._parse_response(response.text)

    def _parse_response(self, content: str) -> list[ArxivMetadata]:
        """Parse API XML response.

        Args:
            content: XML response content.

        Returns:
            List of ArxivMetadata objects.
        """
        try:
            root = ET.fromstring(content)
        except ET.ParseError as e:
            logger.error(f"Failed to parse arXiv API response: {e}")
            return []

        results = []
        for entry in root.findall("atom:entry", NAMESPACES):
            metadata = self._parse_entry(entry)
            if metadata:
                results.append(metadata)

        return results

    def _parse_entry(self, entry: ET.Element) -> ArxivMetadata | None:
        """Parse a single entry from API response.

        Args:
            entry: XML entry element.

        Returns:
            ArxivMetadata or None if parsing fails.
        """
        try:
            # Extract ID and version
            id_elem = entry.find("atom:id", NAMESPACES)
            if id_elem is None or id_elem.text is None:
                return None

            match = ARXIV_ID_PATTERN.search(id_elem.text)
            if not match:
                return None

            arxiv_id = match.group(1)
            version_str = match.group(2)
            version = int(version_str[1:]) if version_str else 1

            # Extract title
            title_elem = entry.find("atom:title", NAMESPACES)
            title = self._clean_text((title_elem.text or "") if title_elem is not None else "")

            # Extract abstract
            summary_elem = entry.find("atom:summary", NAMESPACES)
            abstract = self._clean_text(
                (summary_elem.text or "") if summary_elem is not None else ""
            )

            # Extract authors
            authors = []
            for author_elem in entry.findall("atom:author", NAMESPACES):
                name_elem = author_elem.find("atom:name", NAMESPACES)
                name = name_elem.text if name_elem is not None else ""

                affiliations = []
                for aff_elem in author_elem.findall("arxiv:affiliation", NAMESPACES):
                    if aff_elem.text:
                        affiliations.append(aff_elem.text)

                if name:
                    authors.append(Author(name=name, affiliations=affiliations))

            # Extract categories
            categories = []
            primary_category = ""
            for cat_elem in entry.findall("atom:category", NAMESPACES):
                term = cat_elem.get("term", "")
                if term:
                    categories.append(term)
                    if cat_elem.get("{http://arxiv.org/schemas/atom}primary") == "yes":
                        primary_category = term

            # If no explicit primary, use first category
            if not primary_category and categories:
                primary_category = categories[0]

            # Also check for arxiv:primary_category element
            primary_elem = entry.find("arxiv:primary_category", NAMESPACES)
            if primary_elem is not None:
                term = primary_elem.get("term", "")
                if term:
                    primary_category = term
                    if term not in categories:
                        categories.insert(0, term)

            # Extract dates
            published = None
            updated = None
            pub_elem = entry.find("atom:published", NAMESPACES)
            if pub_elem is not None and pub_elem.text:
                published = self._parse_datetime(pub_elem.text)
            upd_elem = entry.find("atom:updated", NAMESPACES)
            if upd_elem is not None and upd_elem.text:
                updated = self._parse_datetime(upd_elem.text)

            # Extract DOI
            doi = ""
            doi_elem = entry.find("arxiv:doi", NAMESPACES)
            if doi_elem is not None and doi_elem.text:
                doi = doi_elem.text.strip()

            # Extract journal reference
            journal_ref = ""
            jr_elem = entry.find("arxiv:journal_ref", NAMESPACES)
            if jr_elem is not None and jr_elem.text:
                journal_ref = jr_elem.text.strip()

            # Extract comments
            comments = ""
            cm_elem = entry.find("arxiv:comment", NAMESPACES)
            if cm_elem is not None and cm_elem.text:
                comments = cm_elem.text.strip()

            # Extract links
            links = {}
            for link_elem in entry.findall("atom:link", NAMESPACES):
                rel = link_elem.get("rel", "")
                href = link_elem.get("href", "")
                link_type = link_elem.get("type", "")
                if href:
                    key = rel or link_type or "alternate"
                    links[key] = href

            return ArxivMetadata(
                arxiv_id=arxiv_id,
                version=version,
                title=title,
                abstract=abstract,
                authors=authors,
                categories=categories,
                primary_category=primary_category,
                published=published,
                updated=updated,
                doi=doi,
                journal_ref=journal_ref,
                comments=comments,
                links=links,
            )

        except Exception as e:
            logger.warning(f"Error parsing arXiv entry: {e}")
            return None

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        # Normalize whitespace
        return " ".join(text.split())

    def _parse_datetime(self, dt_str: str) -> datetime | None:
        """Parse ISO datetime string."""
        try:
            # Handle various formats
            dt_str = dt_str.strip()
            if dt_str.endswith("Z"):
                dt_str = dt_str[:-1] + "+00:00"
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except ValueError:
            return None
