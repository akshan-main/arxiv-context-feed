"""Citations refresh pipeline using OpenAlex.

Only refreshes citations for papers that have a DOI.
Uses free OpenAlex API with optional API keys for higher rate limits.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

from contextual_arxiv_feed.config import AppConfig, SourcesConfig
from contextual_arxiv_feed.contextual import ContextualClient
from contextual_arxiv_feed.contextual.naming import parse_document_name

logger = logging.getLogger(__name__)


@dataclass
class CitationData:
    """Citation data from external sources."""

    citation_count: int = 0
    reference_count: int = 0
    venue: str = ""
    year: int = 0
    source: str = ""  # "openalex"
    authors: str = ""  # pipe-separated author names, truncated to 200 chars
    publication_date: str = ""  # ISO date e.g. "2024-01-15"
    paper_type: str = ""  # "article", "preprint", "review", etc.
    open_access: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict with int-only values."""
        return {
            "citation_count": self.citation_count,
            "reference_count": self.reference_count,
            "venue": self.venue,
            "year": self.year,
            "source": self.source,
            "authors": self.authors,
            "publication_date": self.publication_date,
            "paper_type": self.paper_type,
            "open_access": self.open_access,
        }


@dataclass
class RefreshResult:
    """Result for a single citation refresh."""

    arxiv_id: str
    version: int
    doi: str
    success: bool = False
    citation_data: CitationData | None = None
    error: str = ""


@dataclass
class CitationsStats:
    """Statistics for citations refresh run."""

    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    total_documents: int = 0
    documents_with_doi: int = 0
    refreshed: int = 0
    failed: int = 0
    skipped_no_doi: int = 0
    results: list[RefreshResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "total_documents": self.total_documents,
            "documents_with_doi": self.documents_with_doi,
            "refreshed": self.refreshed,
            "failed": self.failed,
            "skipped_no_doi": self.skipped_no_doi,
        }


class OpenAlexClient:
    """Client for OpenAlex API (free, preferred)."""

    BASE_URL = "https://api.openalex.org"

    def __init__(self, api_key: str = "", rate_limit: int = 10, key_pool=None):
        """Initialize client.

        Args:
            api_key: Optional API key for higher rate limits.
            rate_limit: Requests per second.
            key_pool: Optional KeyPool for multi-key rotation.
        """
        self._api_key = api_key
        self._key_pool = key_pool
        self._rate_limit = rate_limit
        self._last_request = 0.0
        self._client = httpx.Client(timeout=30.0)

    def close(self) -> None:
        """Close client."""
        self._client.close()

    def _wait_for_rate_limit(self) -> None:
        """Wait for rate limit."""
        now = time.monotonic()
        elapsed = now - self._last_request
        min_interval = 1.0 / self._rate_limit
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request = time.monotonic()

    def _build_url(self, base_url: str, api_key: str) -> str:
        """Build URL with api_key query parameter (OpenAlex auth method)."""
        if api_key:
            sep = "&" if "?" in base_url else "?"
            return f"{base_url}{sep}api_key={api_key}"
        return base_url

    def _request_with_key_rotation(self, base_url: str) -> httpx.Response | None:
        """Make request, rotating through all API keys on 429.

        Exhausts every key in the pool before giving up (like LLM judge).

        Args:
            base_url: URL without api_key param.

        Returns:
            Successful response or None if all keys exhausted.
        """
        if not self._key_pool or self._key_pool.size == 0:
            # No pool, use single key
            self._wait_for_rate_limit()
            url = self._build_url(base_url, self._api_key)
            return self._client.get(url)

        # Try all keys in pool
        while True:
            api_key = self._key_pool.get_key()
            if api_key is None:
                # All keys exhausted, try without key as last resort
                logger.warning("All OpenAlex API keys exhausted, trying without key")
                self._wait_for_rate_limit()
                return self._client.get(base_url)

            self._wait_for_rate_limit()
            url = self._build_url(base_url, api_key)
            response = self._client.get(url)

            if response.status_code == 429:
                self._key_pool.report_rate_limit(api_key)
                logger.info(f"OpenAlex 429, rotating key ...{api_key[-4:]}")
                continue

            self._key_pool.report_success(api_key)
            return response

    def get_by_doi(self, doi: str) -> CitationData | None:
        """Get citation data by DOI.

        Uses api_key query param (OpenAlex auth). Rotates through all
        API keys on 429 before giving up.

        Args:
            doi: Paper DOI.

        Returns:
            CitationData or None if not found.
        """
        base_url = f"{self.BASE_URL}/works/doi:{doi}"

        try:
            response = self._request_with_key_rotation(base_url)
            if response is None:
                return None

            if response.status_code == 404:
                logger.debug(f"DOI not found in OpenAlex: {doi}")
                return None

            if response.status_code != 200:
                logger.warning(f"OpenAlex error: {response.status_code}")
                return None

            data = response.json()

            return CitationData(
                citation_count=data.get("cited_by_count", 0),
                reference_count=len(data.get("referenced_works", [])),
                venue=self._extract_venue(data),
                year=data.get("publication_year", 0) or 0,
                source="openalex",
                authors=self._extract_authors(data),
                publication_date=data.get("publication_date", "") or "",
                paper_type=data.get("type", "") or "",
                open_access=self._extract_open_access(data),
            )

        except httpx.HTTPError as e:
            logger.error(f"OpenAlex HTTP error: {e}")
            return None

    def _extract_venue(self, data: dict) -> str:
        """Extract venue name from OpenAlex data."""
        # Try primary location first
        primary = data.get("primary_location", {})
        if primary:
            source = primary.get("source", {})
            if source:
                return source.get("display_name", "")

        # Try host venue
        host = data.get("host_venue", {})
        if host:
            return host.get("display_name", "")

        return ""

    def _extract_authors(self, data: dict) -> str:
        """Extract pipe-separated author names, truncated to 200 chars."""
        authorships = data.get("authorships", [])
        names = []
        for a in authorships:
            author = a.get("author", {})
            name = author.get("display_name", "")
            if name:
                names.append(name)
        result = "|".join(names)
        if len(result) > 200:
            # Truncate to last complete name within 200 chars
            result = result[:200].rsplit("|", 1)[0]
        return result

    def _extract_open_access(self, data: dict) -> bool:
        """Extract open access status."""
        oa = data.get("open_access", {})
        return bool(oa.get("is_oa", False))


def refresh_citations_for_doi(
    doi: str,
    sources_config: SourcesConfig,
    openalex_key: str = "",
) -> dict[str, Any] | None:
    """Refresh citations for a single DOI.

    Args:
        doi: Paper DOI.
        sources_config: Sources configuration.
        openalex_key: OpenAlex API key.

    Returns:
        Citation data dict or None if failed.
    """
    if not doi:
        return None

    if sources_config.enable_openalex:
        client = OpenAlexClient(
            openalex_key,
            sources_config.openalex_rate_limit_per_second,
        )
        try:
            data = client.get_by_doi(doi)
            if data:
                return data.to_dict()
        finally:
            client.close()

    return None


class CitationsRefresh:
    """Pipeline for refreshing citation data via OpenAlex."""

    def __init__(
        self,
        config: AppConfig,
        openalex_key: str = "",
        dry_run: bool = False,
    ):
        """Initialize pipeline.

        Args:
            config: Application configuration.
            openalex_key: OpenAlex API key.
            dry_run: If True, skip actual updates.
        """
        self._config = config
        self._openalex_key = openalex_key
        self._dry_run = dry_run or config.dry_run

        self._contextual: ContextualClient | None = None
        if not self._dry_run:
            self._contextual = ContextualClient(
                config.contextual_api_key,
                config.contextual_datastore_id,
                config.contextual_base_url,
            )

        from contextual_arxiv_feed.keys.rotator import KeyRotator

        self._rotator = KeyRotator.from_environment(
            cooldown_seconds=config.sources.key_cooldown_seconds
        )

        self._openalex: OpenAlexClient | None = None
        if config.sources.enable_openalex:
            self._openalex = OpenAlexClient(
                openalex_key,
                config.sources.openalex_rate_limit_per_second,
                key_pool=self._rotator.get_pool("openalex"),
            )

    def close(self) -> None:
        """Close resources."""
        if self._contextual:
            self._contextual.close()
        if self._openalex:
            self._openalex.close()

    def __enter__(self) -> CitationsRefresh:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def run(self) -> CitationsStats:
        """Run citations refresh.

        Returns:
            CitationsStats with run results.
        """
        stats = CitationsStats(
            run_id=str(uuid.uuid4())[:8],
            started_at=datetime.utcnow(),
        )

        logger.info(f"Starting citations refresh run {stats.run_id}")

        if not self._contextual:
            logger.info("[DRY RUN] Would refresh citations")
            stats.finished_at = datetime.utcnow()
            return stats

        doc_names = self._contextual.list_documents(prefix="arxiv:")
        stats.total_documents = len(doc_names)

        for name in doc_names:
            info = parse_document_name(name)
            if not info or info.is_manifest:
                continue

            result = self._refresh_document(name, info.arxiv_id, info.version, stats)
            stats.results.append(result)

        stats.finished_at = datetime.utcnow()
        logger.info(f"Citations refresh complete: {stats.refreshed} refreshed, {stats.failed} failed")

        return stats

    def _refresh_document(
        self,
        doc_name: str,
        arxiv_id: str,
        version: int,
        stats: CitationsStats,
    ) -> RefreshResult:
        """Refresh citations for a single document.

        Args:
            doc_name: Document name.
            arxiv_id: arXiv ID.
            version: Version.
            stats: Stats to update.

        Returns:
            RefreshResult.
        """
        result = RefreshResult(arxiv_id=arxiv_id, version=version, doi="")

        doc = self._contextual.get_document(doc_name)
        if not doc:
            result.error = "Document not found"
            stats.failed += 1
            return result

        doi = doc.metadata.get("doi", "")
        result.doi = doi

        if not doi:
            stats.skipped_no_doi += 1
            return result

        stats.documents_with_doi += 1

        citation_data = self._get_citations(doi)
        if not citation_data:
            result.error = "Failed to fetch citations"
            stats.failed += 1
            return result

        result.citation_data = citation_data

        if self._dry_run:
            logger.info(f"[DRY RUN] Would update citations for {arxiv_id}")
            result.success = True
            stats.refreshed += 1
            return result

        # Update packed fields in our 15-field metadata schema
        cite_parts = [
            f"cited={citation_data.citation_count}",
            f"refs={citation_data.reference_count}",
            f"updated={datetime.utcnow().strftime('%Y-%m-%d')}",
        ]
        venue_parts = []
        if citation_data.venue:
            venue_parts.append(f"venue={citation_data.venue}")
        if citation_data.paper_type:
            venue_parts.append(f"type={citation_data.paper_type}")
        venue_parts.append(f"oa={'true' if citation_data.open_access else 'false'}")

        updates = {
            "citations": "|".join(cite_parts),
            "venue_info": "|".join(venue_parts),
            "authors": (citation_data.authors[:200] if citation_data.authors else ""),
            "publication_date": citation_data.publication_date,
        }

        if self._contextual.update_metadata(doc_name, updates):
            result.success = True
            stats.refreshed += 1
        else:
            result.error = "Failed to update metadata"
            stats.failed += 1

        return result

    def _get_citations(self, doi: str) -> CitationData | None:
        """Get citations from OpenAlex.

        Args:
            doi: Paper DOI.

        Returns:
            CitationData or None.
        """
        if self._openalex:
            return self._openalex.get_by_doi(doi)
        return None
