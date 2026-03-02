"""Backfill pipeline for historical arXiv papers.

For one-time local runs to ingest papers from a date range.
Uses arXiv API search instead of RSS feeds.

Run locally: contextual-arxiv-feed backfill --start 2024-01-01 --end 2024-06-01
"""

from __future__ import annotations

# Flip to True when you need to run backfill. Keep False for normal operation.
BACKFILL_ENABLED = False

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from contextual_arxiv_feed.arxiv import ArxivAPI, ArxivMetadata, PDFDownloader
from contextual_arxiv_feed.arxiv.throttle import ArxivThrottle
from contextual_arxiv_feed.config import AppConfig
from contextual_arxiv_feed.contextual import (
    ContextualClient,
)
from contextual_arxiv_feed.judge import JudgeOutput, create_judge
from contextual_arxiv_feed.judge.schema import QualityBreakdown
from contextual_arxiv_feed.matcher import KeywordMatcher
from contextual_arxiv_feed.pipeline.venue import detect_top_venue

logger = logging.getLogger(__name__)

# arXiv API max results per query
MAX_RESULTS_PER_QUERY = 1000


@dataclass
class BackfillResult:
    """Result for a single paper processing."""

    arxiv_id: str
    version: int
    title: str
    stage1_passed: bool = False
    stage1_topics: list[str] = field(default_factory=list)
    auto_ingest_reason: str = ""
    auto_ingest_venue: str = ""
    stage2_passed: bool = False
    judge_output: JudgeOutput | None = None
    ingested: bool = False
    skipped_exists: bool = False
    download_failed: bool = False
    ingest_failed: bool = False
    error: str = ""


@dataclass
class BackfillStats:
    """Statistics for backfill pipeline run."""

    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    candidates_total: int = 0
    stage1_passed: int = 0
    stage1_failed: int = 0
    already_exists: int = 0
    auto_ingest_revised: int = 0
    auto_ingest_venue: int = 0
    stage2_passed: int = 0
    stage2_failed: int = 0
    accepted: int = 0
    accepted_low_confidence: int = 0
    rejected_topicality: int = 0
    rejected_quality: int = 0
    download_failed: int = 0
    ingest_failed: int = 0
    ingested: int = 0
    results: list[BackfillResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "candidates_total": self.candidates_total,
            "stage1_passed": self.stage1_passed,
            "stage1_failed": self.stage1_failed,
            "already_exists": self.already_exists,
            "auto_ingest_revised": self.auto_ingest_revised,
            "auto_ingest_venue": self.auto_ingest_venue,
            "stage2_passed": self.stage2_passed,
            "stage2_failed": self.stage2_failed,
            "accepted": self.accepted,
            "accepted_low_confidence": self.accepted_low_confidence,
            "rejected_topicality": self.rejected_topicality,
            "rejected_quality": self.rejected_quality,
            "download_failed": self.download_failed,
            "ingest_failed": self.ingest_failed,
            "ingested": self.ingested,
        }


class BackfillPipeline:
    """Pipeline for backfilling historical arXiv papers.

    Uses arXiv API search_by_date_range instead of RSS feeds.
    Same two-stage filtering as daily pipeline.
    """

    def __init__(
        self,
        config: AppConfig,
        start_date: datetime,
        end_date: datetime,
        dry_run: bool = False,
    ):
        """Initialize pipeline.

        Args:
            config: Application configuration.
            start_date: Start of date range (inclusive).
            end_date: End of date range (inclusive).
            dry_run: If True, skip actual operations.
        """
        self._config = config
        self._start_date = start_date
        self._end_date = end_date
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

    def __enter__(self) -> BackfillPipeline:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def run(self) -> BackfillStats:
        """Run the backfill pipeline.

        Returns:
            BackfillStats with run results.
        """
        stats = BackfillStats(
            run_id=str(uuid.uuid4())[:8],
            started_at=datetime.utcnow(),
            start_date=self._start_date,
            end_date=self._end_date,
        )

        logger.info(
            f"Starting backfill run {stats.run_id}: "
            f"{self._start_date.date()} to {self._end_date.date()}"
        )

        # Get categories from enabled topics
        categories = self._get_categories()
        logger.info(f"Searching categories: {categories}")

        # Query arXiv API for papers in date range
        papers = self._api.search_by_date_range(
            self._start_date,
            self._end_date,
            categories,
            max_results=MAX_RESULTS_PER_QUERY,
        )
        stats.candidates_total = len(papers)
        logger.info(f"Found {len(papers)} papers in date range")

        if len(papers) >= MAX_RESULTS_PER_QUERY:
            logger.warning(
                f"Hit max results limit ({MAX_RESULTS_PER_QUERY}). "
                "Consider splitting into smaller date ranges."
            )

        # Process each paper
        for i, metadata in enumerate(papers):
            if i > 0 and i % 50 == 0:
                logger.info(f"Progress: {i}/{len(papers)} papers processed")

            result = self._process_paper(metadata, stats)
            stats.results.append(result)

        stats.finished_at = datetime.utcnow()
        logger.info(
            f"Backfill complete: {stats.ingested} ingested, "
            f"{stats.rejected_quality} rejected"
        )

        return stats

    def _get_categories(self) -> list[str]:
        """Get unique arXiv categories from enabled topics."""
        categories = set()
        for topic in self._config.topics.get_enabled_topics():
            categories.update(topic.arxiv_categories)
        return list(categories)

    def _process_paper(
        self,
        metadata: ArxivMetadata,
        stats: BackfillStats,
    ) -> BackfillResult:
        """Process a single paper through the pipeline.

        Args:
            metadata: Paper metadata from arXiv API.
            stats: Stats to update.

        Returns:
            BackfillResult for this paper.
        """
        result = BackfillResult(
            arxiv_id=metadata.arxiv_id,
            version=metadata.version,
            title=metadata.title,
        )

        # Check if already exists
        if self._check_exists(metadata.arxiv_id, metadata.version):
            result.skipped_exists = True
            stats.already_exists += 1
            return result

        # Stage 1: Keyword match on title + abstract (use first 500 chars for snippet)
        abstract_snippet = metadata.abstract[:500] if metadata.abstract else ""
        match_result = self._matcher.match(metadata.title, abstract_snippet)

        if not match_result.passes_stage1:
            stats.stage1_failed += 1
            return result

        result.stage1_passed = True
        result.stage1_topics = match_result.matched_topics
        stats.stage1_passed += 1

        # Auto-ingest: Revised version (v2+)
        if metadata.version >= 2:
            result.auto_ingest_reason = "revised_version"
            stats.auto_ingest_revised += 1
            logger.info(f"Auto-ingest (revised v{metadata.version}): {metadata.arxiv_id}")
            return self._download_and_ingest(metadata, result, stats, auto_ingest=True)

        # Auto-ingest: Top venue accepted
        venue_result = detect_top_venue(metadata.comments, metadata.journal_ref)
        if venue_result and venue_result.detected:
            result.auto_ingest_reason = "top_venue"
            result.auto_ingest_venue = venue_result.venue_display
            stats.auto_ingest_venue += 1
            logger.info(f"Auto-ingest (venue: {venue_result.venue_display}): {metadata.arxiv_id}")
            return self._download_and_ingest(metadata, result, stats, auto_ingest=True)

        # Stage 2: LLM judge
        judge_result = self._judge.judge(metadata.title, metadata.abstract)
        if not judge_result.success:
            result.error = f"Judge error: {judge_result.error}"
            stats.stage2_failed += 1
            logger.warning(f"Judge failed for {metadata.arxiv_id}: {judge_result.error}")
            # Fallback: ingest on LLM failure
            logger.info(f"Fallback ingest (LLM failed): {metadata.arxiv_id}")
            return self._download_and_ingest(metadata, result, stats, auto_ingest=True)

        result.judge_output = judge_result.output
        result.stage2_passed = True
        stats.stage2_passed += 1

        output = judge_result.output

        # Confidence-based quality check
        quality_ok = output.quality_i >= 65
        low_confidence = output.confidence_i < 80

        if quality_ok:
            stats.accepted += 1
        elif low_confidence:
            stats.accepted += 1
            stats.accepted_low_confidence += 1
            logger.info(
                f"Accepted (low confidence) {metadata.arxiv_id}: "
                f"quality={output.quality_i}, confidence={output.confidence_i}"
            )
        else:
            stats.rejected_quality += 1
            return result

        return self._download_and_ingest(metadata, result, stats, auto_ingest=False)

    def _download_and_ingest(
        self,
        metadata: ArxivMetadata,
        result: BackfillResult,
        stats: BackfillStats,
        auto_ingest: bool = False,
    ) -> BackfillResult:
        """Download PDF and ingest paper.

        Args:
            metadata: Paper metadata.
            result: Result to update.
            stats: Stats to update.
            auto_ingest: Whether this is auto-ingest (no judge output).

        Returns:
            Updated BackfillResult.
        """
        if self._dry_run:
            reason = result.auto_ingest_reason or "judge_accepted"
            logger.info(f"[DRY RUN] Would ingest ({reason}): {metadata.arxiv_id}v{metadata.version}")
            result.ingested = True
            stats.ingested += 1
            return result

        # Download PDF
        pdf_result = self._pdf_downloader.download(metadata.pdf_url)
        if not pdf_result.success:
            result.download_failed = True
            result.error = f"Download failed: {pdf_result.error_message}"
            stats.download_failed += 1
            return result

        # Create judge output for auto-ingest if needed
        judge_output: JudgeOutput
        if auto_ingest and result.judge_output is None:
            judge_output = self._create_auto_ingest_judge_output(result)
        else:
            assert result.judge_output is not None
            judge_output = result.judge_output

        # Ingest
        success = self._ingest_paper(metadata, judge_output, pdf_result.pdf_bytes, stats.run_id, result)
        if success:
            result.ingested = True
            stats.ingested += 1
        else:
            result.ingest_failed = True
            stats.ingest_failed += 1

        return result

    def _create_auto_ingest_judge_output(self, result: BackfillResult) -> JudgeOutput:
        """Create placeholder JudgeOutput for auto-ingested papers."""
        reason = result.auto_ingest_reason
        venue = result.auto_ingest_venue

        return JudgeOutput(
            prompt_version=0,
            model_id=f"auto_ingest:{reason}",
            quality_verdict="accept",
            quality_i=100 if reason == "top_venue" else 80,
            quality_breakdown_i=QualityBreakdown(
                novelty_i=80,
                relevance_i=80,
                technical_depth_i=80,
            ),
            confidence_i=100,
            rationale=f"Auto-ingested: {reason}" + (f" ({venue})" if venue else ""),
        )

    def _check_exists(self, arxiv_id: str, version: int) -> bool:
        """Check if paper already exists in datastore."""
        if self._dry_run or not self._contextual:
            return False
        return self._contextual.document_exists(arxiv_id, version)

    def _ingest_paper(
        self,
        metadata: ArxivMetadata,
        judge_output: JudgeOutput,
        pdf_bytes: bytes,
        run_id: str,
        result: BackfillResult | None = None,
    ) -> bool:
        """Ingest paper to datastore.

        Args:
            metadata: Paper metadata.
            judge_output: Judge result.
            pdf_bytes: PDF content.
            run_id: Run ID.
            result: BackfillResult for stage1 topics.

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
            "topics": "|".join(result.stage1_topics if result else []),
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
            "discovery_channel": "backfill",
            "citation_enrichment": None,
            "run_metadata": {
                "run_id": run_id,
                "ingested_at": datetime.utcnow().isoformat(),
                "pipeline": "backfill",
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
