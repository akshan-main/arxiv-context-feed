"""Weekly updates pipeline for processing arXiv paper updates.

Handles:
1. New versions (v2+): Full two-stage filtering, ingest alongside v1
2. DOI enrichment: If stored DOI is empty but new DOI exists, update metadata + refresh citations

DOI enrichment is a CONVERGENT process: papers become richer over time automatically.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from contextual_arxiv_feed.arxiv import ArxivAPI, ArxivMetadata, PDFDownloader
from contextual_arxiv_feed.arxiv.throttle import ArxivThrottle
from contextual_arxiv_feed.config import AppConfig
from contextual_arxiv_feed.contextual import (
    ContextualClient,
    build_document_name,
)
from contextual_arxiv_feed.contextual.naming import parse_document_name
from contextual_arxiv_feed.judge import JudgeOutput, create_judge
from contextual_arxiv_feed.matcher import KeywordMatcher
from contextual_arxiv_feed.pipeline.citations import refresh_citations_for_doi

logger = logging.getLogger(__name__)


@dataclass
class UpdateResult:
    """Result for a single update processing."""

    arxiv_id: str
    version: int
    is_new_version: bool = False
    is_doi_update: bool = False
    stage1_passed: bool = False
    stage2_passed: bool = False
    ingested: bool = False
    metadata_updated: bool = False
    error: str = ""


@dataclass
class UpdatesStats:
    """Statistics for updates pipeline run."""

    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    lookback_days: int = 7
    candidates_total: int = 0
    new_versions_found: int = 0
    new_versions_ingested: int = 0
    doi_updates_found: int = 0
    doi_updates_applied: int = 0
    already_exists: int = 0
    stage1_failed: int = 0
    stage2_failed: int = 0
    rejected: int = 0
    download_failed: int = 0
    ingest_failed: int = 0
    results: list[UpdateResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "lookback_days": self.lookback_days,
            "candidates_total": self.candidates_total,
            "new_versions_found": self.new_versions_found,
            "new_versions_ingested": self.new_versions_ingested,
            "doi_updates_found": self.doi_updates_found,
            "doi_updates_applied": self.doi_updates_applied,
        }


class UpdatesPipeline:
    """Weekly pipeline for processing arXiv updates.

    Handles:
    - New paper versions (v2, v3, etc.)
    - DOI enrichment for already-ingested papers
    """

    def __init__(
        self,
        config: AppConfig,
        lookback_days: int = 7,
        dry_run: bool = False,
    ):
        """Initialize pipeline.

        Args:
            config: Application configuration.
            lookback_days: Days to look back for updates.
            dry_run: If True, skip actual operations.
        """
        self._config = config
        self._lookback_days = lookback_days
        self._dry_run = dry_run or config.dry_run

        # Initialize components
        self._throttle = ArxivThrottle(config.arxiv_throttle_seconds)
        self._api = ArxivAPI(self._throttle)
        self._pdf_downloader = PDFDownloader(self._throttle, config.max_download_mb)
        self._matcher = KeywordMatcher(config.topics)
        self._judge = create_judge(config.judge, config.topics.get_enabled_topics())

        self._contextual: ContextualClient | None = None
        if not self._dry_run:
            self._contextual = ContextualClient(
                config.contextual_api_key,
                config.contextual_datastore_id,
                config.contextual_base_url,
            )
            self._contextual.configure_text_only_ingestion()

    def close(self) -> None:
        """Close all resources."""
        self._api.close()
        self._pdf_downloader.close()
        self._judge.close()
        if self._contextual:
            self._contextual.close()

    def __enter__(self) -> UpdatesPipeline:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def run(self) -> UpdatesStats:
        """Run the updates pipeline.

        Returns:
            UpdatesStats with run results.
        """
        stats = UpdatesStats(
            run_id=str(uuid.uuid4())[:8],
            started_at=datetime.utcnow(),
            lookback_days=self._lookback_days,
        )

        logger.info(f"Starting updates pipeline run {stats.run_id}")

        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=self._lookback_days)

        # Get categories from enabled topics
        categories = self._get_categories()

        # Query arXiv API for updates
        logger.info(f"Querying arXiv for updates from {start_date.date()} to {end_date.date()}")
        updated_papers = self._api.search_by_date_range(start_date, end_date, categories)
        stats.candidates_total = len(updated_papers)
        logger.info(f"Found {len(updated_papers)} updated papers")

        # Get existing documents for comparison
        existing_docs = self._get_existing_documents() if self._contextual else {}

        # Process each update
        for metadata in updated_papers:
            result = self._process_update(metadata, existing_docs, stats)
            stats.results.append(result)

        stats.finished_at = datetime.utcnow()
        logger.info(
            f"Updates complete: {stats.new_versions_ingested} new versions, "
            f"{stats.doi_updates_applied} DOI updates"
        )

        return stats

    def _get_categories(self) -> list[str]:
        """Get unique arXiv categories from enabled topics."""
        categories = set()
        for topic in self._config.topics.get_enabled_topics():
            categories.update(topic.arxiv_categories)
        return list(categories)

    def _get_existing_documents(self) -> dict[str, dict[int, Any]]:
        """Get existing documents and their metadata.

        Returns:
            Dict mapping arxiv_id to {version: metadata} dicts.
        """
        existing: dict[str, dict[int, Any]] = {}

        # List all arxiv documents
        doc_names = self._contextual.list_documents(prefix="arxiv:")

        for name in doc_names:
            info = parse_document_name(name)
            if info and not info.is_manifest:
                if info.arxiv_id not in existing:
                    existing[info.arxiv_id] = {}

                # Get document metadata
                doc = self._contextual.get_document(name)
                if doc:
                    existing[info.arxiv_id][info.version] = doc.metadata

        return existing

    def _process_update(
        self,
        metadata: ArxivMetadata,
        existing_docs: dict[str, dict[int, Any]],
        stats: UpdatesStats,
    ) -> UpdateResult:
        """Process a single update.

        Args:
            metadata: Updated paper metadata.
            existing_docs: Existing documents.
            stats: Stats to update.

        Returns:
            UpdateResult for this paper.
        """
        result = UpdateResult(
            arxiv_id=metadata.arxiv_id,
            version=metadata.version,
        )

        existing_versions = existing_docs.get(metadata.arxiv_id, {})

        # Check if this exact version already exists
        if metadata.version in existing_versions:
            # Check for DOI enrichment opportunity
            stored_metadata = existing_versions[metadata.version]
            stored_doi = stored_metadata.get("doi", "")
            new_doi = metadata.doi or ""

            if (not stored_doi and new_doi) or (stored_doi and new_doi and stored_doi != new_doi):
                # DOI has appeared or changed - update metadata
                result.is_doi_update = True
                stats.doi_updates_found += 1

                if self._update_doi_metadata(metadata, stored_metadata, stats):
                    result.metadata_updated = True
                    stats.doi_updates_applied += 1

            return result

        # This is a new version - check if we have any version of this paper
        if existing_versions:
            result.is_new_version = True
            stats.new_versions_found += 1
        else:
            # Paper not in datastore at all - could be new or just not ingested before
            pass

        # Run two-stage filtering for new versions
        return self._process_new_version(metadata, result, stats)

    def _process_new_version(
        self,
        metadata: ArxivMetadata,
        result: UpdateResult,
        stats: UpdatesStats,
    ) -> UpdateResult:
        """Process a potentially new version through two-stage filtering.

        Args:
            metadata: Paper metadata.
            result: Result object to update.
            stats: Stats to update.

        Returns:
            Updated result.
        """
        # Stage 1: Keyword match
        match_result = self._matcher.match(metadata.title, metadata.abstract[:500])
        if not match_result.passes_stage1:
            stats.stage1_failed += 1
            return result

        result.stage1_passed = True

        # Stage 2: LLM judge
        judge_result = self._judge.judge(metadata.title, metadata.abstract)
        if not judge_result.success:
            result.error = f"Judge error: {judge_result.error}"
            stats.stage2_failed += 1
            return result

        result.stage2_passed = True

        if not judge_result.is_accepted:
            stats.rejected += 1
            return result

        # Download and ingest
        if self._dry_run:
            logger.info(f"[DRY RUN] Would ingest: {metadata.arxiv_id}v{metadata.version}")
            result.ingested = True
            stats.new_versions_ingested += 1
            return result

        # Download PDF
        pdf_result = self._pdf_downloader.download(metadata.pdf_url)
        if not pdf_result.success:
            result.error = f"Download failed: {pdf_result.error_message}"
            stats.download_failed += 1
            return result

        # Ingest
        if self._ingest_paper(metadata, judge_result.output, pdf_result.pdf_bytes, stats.run_id, match_result.matched_topics):
            result.ingested = True
            stats.new_versions_ingested += 1
        else:
            stats.ingest_failed += 1

        return result

    def _update_doi_metadata(
        self,
        metadata: ArxivMetadata,
        stored_metadata: dict[str, Any],
        stats: UpdatesStats,
    ) -> bool:
        """Update metadata when DOI appears or changes.

        Args:
            metadata: New metadata with DOI.
            stored_metadata: Currently stored metadata.
            stats: Stats for run_id.

        Returns:
            True if update succeeded.
        """
        if self._dry_run:
            logger.info(f"[DRY RUN] Would update DOI for: {metadata.arxiv_id}")
            return True

        doc_name = build_document_name(metadata.arxiv_id, metadata.version)
        new_doi = metadata.doi or ""

        logger.info(f"Updating DOI for {doc_name}: '{stored_metadata.get('doi', '')}' -> '{new_doi}'")

        # Update metadata
        updates = {"doi": new_doi}

        # Also try to refresh citations if DOI is now available
        if new_doi:
            citation_data = refresh_citations_for_doi(new_doi, self._config.sources)
            if citation_data:
                updates.update({
                    "citation_count": citation_data.get("citation_count", 0),
                    "reference_count": citation_data.get("reference_count", 0),
                    "venue": citation_data.get("venue", ""),
                    "citations_updated_at": datetime.utcnow().isoformat(),
                })

        return self._contextual.update_metadata(doc_name, updates)

    def _ingest_paper(
        self,
        metadata: ArxivMetadata,
        judge_output: JudgeOutput,
        pdf_bytes: bytes,
        run_id: str,
        topics: list[str] | None = None,
    ) -> bool:
        """Ingest a new version.

        Args:
            metadata: Paper metadata.
            judge_output: Judge result.
            pdf_bytes: PDF content.
            run_id: Run ID.
            topics: Stage 1 matched topics.

        Returns:
            True if ingest succeeded.
        """
        # Build custom_metadata (derivable fields omitted for 2KB budget)
        breakdown = judge_output.quality_breakdown_i
        custom_metadata = {
            "arxiv_id": metadata.arxiv_id,
            "arxiv_version": metadata.version,
            "title": metadata.title,
            "primary_category": metadata.primary_category,
            "categories": "|".join(metadata.categories),
            "doi": metadata.doi or "",
            "year": metadata.year,
            "topics": "|".join(topics or []),
            "quality_verdict": judge_output.quality_verdict,
            "quality_i": judge_output.quality_i,
            "novelty_i": breakdown.novelty_i,
            "relevance_i": breakdown.relevance_i,
            "technical_depth_i": breakdown.technical_depth_i,
            "confidence_i": judge_output.confidence_i,
            "citation_count": 0,
            "reference_count": 0,
            "venue": "",
            "citations_updated_at": "",
            "authors": "",
            "publication_date": "",
            "paper_type": "",
            "open_access": False,
            "judge_model_id": judge_output.model_id,
            "judge_prompt_version": judge_output.prompt_version,
        }

        # Ingest PDF
        pdf_result = self._contextual.ingest_pdf(
            metadata.arxiv_id,
            metadata.version,
            pdf_bytes,
            custom_metadata,
        )
        if not pdf_result.success:
            logger.error(f"PDF ingest failed: {pdf_result.error}")
            return False

        # Build and ingest manifest
        manifest_content = {
            "arxiv_metadata": metadata.to_dict(),
            "judge_output": judge_output.to_dict(),
            "discovery_channel": "weekly_updates",
            "citation_enrichment": None,
            "run_metadata": {
                "run_id": run_id,
                "ingested_at": datetime.utcnow().isoformat(),
                "pipeline": "updates",
            },
        }

        manifest_result = self._contextual.ingest_manifest(
            metadata.arxiv_id,
            metadata.version,
            manifest_content,
            custom_metadata,
        )
        if not manifest_result.success:
            logger.error(f"Manifest ingest failed: {manifest_result.error}")
            return False

        return True
