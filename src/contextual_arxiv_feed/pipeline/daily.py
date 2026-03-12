"""Daily pipeline for ingesting new arXiv papers.

Flow:
1. Fetch RSS feeds for configured categories
2. Deduplicate entries
3. Stage 1: Keyword match on title + abstract snippet
4. Check idempotency (skip already ingested)
5. Fetch full metadata from arXiv API
6. Auto-ingest checks (skip LLM judge if passed):
   - v2+ papers (revised versions)
   - Top venue accepted papers (NeurIPS, ICML, etc.)
7. Stage 2: LLM judge on full abstract
   - Confidence-based rejection: only reject if quality < 65 AND confidence >= 80
8. If accepted: download PDF, ingest PDF + manifest
9. Emit run summary

IMPORTANT: Ingest ALL accepted papers. No quality-based capping.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from contextual_arxiv_feed.arxiv import ArxivAPI, ArxivFeedParser, ArxivMetadata, PDFDownloader
from contextual_arxiv_feed.arxiv.throttle import ArxivThrottle
from contextual_arxiv_feed.config import AppConfig
from contextual_arxiv_feed.contextual import (
    ContextualClient,
    parse_document_name,
)
from contextual_arxiv_feed.judge import JudgeOutput, create_judge
from contextual_arxiv_feed.judge.discovery_agent import DiscoveryAgent
from contextual_arxiv_feed.matcher import KeywordMatcher
from contextual_arxiv_feed.pipeline.venue import detect_top_venue
from contextual_arxiv_feed.store import ChromaDBStore

logger = logging.getLogger(__name__)

# Maximum papers to ingest into Contextual AI per run (to control credits).
# ChromaDB has NO cap — all accepted papers always go to ChromaDB.
# Set to None for unlimited Contextual AI ingestion.
MAX_INGEST_PER_RUN = 10


@dataclass
class PaperResult:
    """Result for a single paper processing."""

    arxiv_id: str
    version: int
    title: str
    stage1_passed: bool = False
    stage1_topics: list[str] = field(default_factory=list)
    auto_ingest_reason: str = ""  # "revised_version", "top_venue", or ""
    auto_ingest_venue: str = ""  # Venue display name if auto-ingested for venue
    stage2_passed: bool = False
    judge_output: JudgeOutput | None = None
    metadata: ArxivMetadata | None = None  # Full arXiv metadata (for decisions payload)
    ingested: bool = False
    skipped_exists: bool = False
    download_failed: bool = False
    ingest_failed: bool = False
    error: str = ""


@dataclass
class PipelineStats:
    """Statistics for a pipeline run."""

    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    candidates_total: int = 0
    stage1_passed: int = 0
    stage1_failed: int = 0
    discovery_agent_passed: int = 0
    already_exists: int = 0
    auto_ingest_revised: int = 0  # v2+ papers auto-ingested
    auto_ingest_venue: int = 0  # Top venue papers auto-ingested
    stage2_passed: int = 0
    stage2_failed: int = 0
    accepted: int = 0
    accepted_low_confidence: int = 0  # Accepted due to low LLM confidence
    rejected_topicality: int = 0
    rejected_quality: int = 0
    download_failed: int = 0
    ingest_failed: int = 0
    ingested: int = 0
    contextual_ingested: int = 0  # Contextual AI only (respects daily cap)
    results: list[PaperResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "candidates_total": self.candidates_total,
            "stage1_passed": self.stage1_passed,
            "stage1_failed": self.stage1_failed,
            "discovery_agent_passed": self.discovery_agent_passed,
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
            "contextual_ingested": self.contextual_ingested,
        }


class DailyPipeline:
    """Daily pipeline for ingesting new arXiv papers."""

    def __init__(self, config: AppConfig, dry_run: bool = False):
        """Initialize pipeline.

        Args:
            config: Application configuration.
            dry_run: If True, skip actual uploads.
        """
        self._config = config
        self._dry_run = dry_run or config.dry_run

        self._throttle = ArxivThrottle(config.arxiv_throttle_seconds)
        self._feed_parser = ArxivFeedParser(self._throttle)
        self._api = ArxivAPI(self._throttle)
        self._pdf_downloader = PDFDownloader(self._throttle, config.max_download_mb)
        self._matcher = KeywordMatcher(config.topics)

        # Build shared LLM key pool (team keys rotate across judge + discovery)
        from contextual_arxiv_feed.judge.llm_judge import _build_llm_key_pool

        llm_key_pool = _build_llm_key_pool()

        self._judge = create_judge(
            config.judge, config.topics.get_enabled_topics(), key_pool=llm_key_pool
        )

        # Discovery Agent for Stage 1.5
        # Uses DISCOVERY_* env vars if set, else falls back to LLM_* vars.
        # Does NOT share key pool with judge — each stage manages its own.
        self._discovery_agent = DiscoveryAgent(
            topics=config.topics.get_enabled_topics(),
        )

        # ChromaDB: only if persistent storage is configured
        # Skip on GitHub Actions where there's no persistent ChromaDB
        self._chromadb: ChromaDBStore | None = None
        if not self._dry_run and (os.getenv("CHROMADB_HOST") or os.getenv("CHROMADB_PERSIST_DIR")):
            self._chromadb = ChromaDBStore()

        # Contextual AI: optional, only if API key is set
        self._contextual: ContextualClient | None = None
        if not self._dry_run and config.contextual_api_key:
            self._contextual = ContextualClient(
                config.contextual_api_key,
                config.contextual_datastore_id,
                config.contextual_base_url,
            )
            self._contextual.configure_standard_ingestion()

    def close(self) -> None:
        """Close all resources."""
        self._feed_parser.close()
        self._api.close()
        self._pdf_downloader.close()
        self._judge.close()
        self._discovery_agent.close()
        if self._chromadb:
            self._chromadb.close()
        if self._contextual:
            self._contextual.close()

    def __enter__(self) -> DailyPipeline:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def run(self) -> PipelineStats:
        """Run the daily pipeline.

        Returns:
            PipelineStats with run results.
        """
        stats = PipelineStats(
            run_id=str(uuid.uuid4())[:8],
            started_at=datetime.utcnow(),
        )

        logger.info(f"Starting daily pipeline run {stats.run_id}")

        categories = self._get_categories()
        if not categories:
            logger.warning("No categories to fetch. Check topics config.")
            stats.finished_at = datetime.utcnow()
            return stats

        logger.info(f"Fetching RSS feeds for {len(categories)} categories")
        entries = self._feed_parser.fetch_multiple_feeds(categories)
        stats.candidates_total = len(entries)
        logger.info(f"Found {len(entries)} unique papers")

        stage1_failed_entries = []
        for entry in entries:
            match_result = self._matcher.match(entry.title, entry.abstract_snippet)
            if match_result.passes_stage1:
                result = self._process_paper_after_stage1(entry, match_result, stats)
                stats.results.append(result)
            else:
                stage1_failed_entries.append(entry)
                stats.stage1_failed += 1

        if self._discovery_agent and stage1_failed_entries:
            logger.info(f"Running Discovery Agent on {len(stage1_failed_entries)} papers")
            for entry in stage1_failed_entries:
                discovery = self._discovery_agent.check(entry.title, entry.abstract_snippet)
                if discovery.is_relevant and discovery.matched_topics:
                    logger.info(
                        f"Discovery Agent found: {entry.arxiv_id} -> {discovery.matched_topics}"
                    )
                    stats.discovery_agent_passed += 1
                    # Create a mock match result for discovered papers
                    from contextual_arxiv_feed.matcher.keyword_matcher import MatchResult
                    mock_match = MatchResult(
                        matched=True,
                        matched_topics=discovery.matched_topics,
                    )
                    result = self._process_paper_after_stage1(entry, mock_match, stats)
                    stats.results.append(result)
                else:
                    result = PaperResult(
                        arxiv_id=entry.arxiv_id,
                        version=entry.version,
                        title=entry.title,
                    )
                    stats.results.append(result)

        stats.finished_at = datetime.utcnow()
        logger.info(
            f"Pipeline complete: {stats.ingested} ingested, "
            f"{stats.rejected_topicality} rejected (topicality), "
            f"{stats.rejected_quality} rejected (quality)"
        )

        return stats

    def _get_categories(self) -> list[str]:
        """Get unique arXiv categories from enabled topics."""
        categories = set()
        for topic in self._config.topics.get_enabled_topics():
            categories.update(topic.arxiv_categories)
        return list(categories)

    def _process_paper_after_stage1(self, entry, match_result, stats: PipelineStats) -> PaperResult:
        """Process a paper that passed Stage 1 (or Discovery Agent).

        Args:
            entry: FeedEntry from RSS.
            match_result: MatchResult from keyword matcher or Discovery Agent.
            stats: Pipeline stats to update.

        Returns:
            PaperResult for this paper.
        """
        result = PaperResult(
            arxiv_id=entry.arxiv_id,
            version=entry.version,
            title=entry.title,
        )

        logger.debug(f"Processing: {entry.arxiv_id} - {entry.title[:50]}...")

        result.stage1_passed = True
        result.stage1_topics = match_result.matched_topics
        stats.stage1_passed += 1

        if self._check_exists(entry.arxiv_id, entry.version):
            result.skipped_exists = True
            stats.already_exists += 1
            logger.debug(f"Already exists: {entry.arxiv_id}v{entry.version}")
            return result

        metadata = self._api.fetch_by_id(entry.id_with_version)
        if not metadata:
            result.error = "Failed to fetch metadata"
            logger.warning(f"Failed to fetch metadata for {entry.arxiv_id}")
            return result

        result.metadata = metadata

        if entry.version >= 2:
            result.auto_ingest_reason = "revised_version"
            stats.auto_ingest_revised += 1
            logger.info(f"Auto-ingest (revised v{entry.version}): {entry.arxiv_id}")
            return self._download_and_ingest(entry, metadata, result, stats, auto_ingest=True)

        venue_result = detect_top_venue(metadata.comments, metadata.journal_ref)
        if venue_result and venue_result.detected:
            result.auto_ingest_reason = "top_venue"
            result.auto_ingest_venue = venue_result.venue_display
            stats.auto_ingest_venue += 1
            logger.info(f"Auto-ingest (venue: {venue_result.venue_display}): {entry.arxiv_id}")
            return self._download_and_ingest(entry, metadata, result, stats, auto_ingest=True)

        judge_result = self._judge.judge(metadata.title, metadata.abstract)
        if not judge_result.success:
            result.error = f"Judge error: {judge_result.error}"
            stats.stage2_failed += 1
            logger.warning(f"Judge failed for {entry.arxiv_id}: {judge_result.error}")
            # Fallback: ingest on LLM failure (benefit of doubt)
            logger.info(f"Fallback ingest (LLM failed): {entry.arxiv_id}")
            return self._download_and_ingest(entry, metadata, result, stats, auto_ingest=True)

        result.judge_output = judge_result.output
        result.stage2_passed = True
        stats.stage2_passed += 1

        output = judge_result.output
        quality_ok = output.quality_i >= 65
        low_confidence = output.confidence_i < 80

        if quality_ok:
            stats.accepted += 1
            logger.debug(f"Accepted {entry.arxiv_id}: quality={output.quality_i}")
        elif low_confidence:
            stats.accepted += 1
            stats.accepted_low_confidence += 1
            logger.info(
                f"Accepted (low confidence) {entry.arxiv_id}: "
                f"quality={output.quality_i}, confidence={output.confidence_i}"
            )
        else:
            stats.rejected_quality += 1
            logger.debug(
                f"Rejected {entry.arxiv_id}: quality={output.quality_i}, "
                f"confidence={output.confidence_i} (confident rejection)"
            )
            return result

        return self._download_and_ingest(entry, metadata, result, stats, auto_ingest=False)

    def _download_and_ingest(
        self,
        entry,
        metadata: ArxivMetadata,
        result: PaperResult,
        stats: PipelineStats,
        auto_ingest: bool = False,
    ) -> PaperResult:
        """Download PDF and store in ChromaDB + optionally Contextual AI.

        ChromaDB storage ALWAYS happens (no cap).
        Contextual AI ingestion only happens if API key is configured
        AND the daily cap hasn't been reached.

        Args:
            entry: FeedEntry from RSS.
            metadata: Full arXiv metadata.
            result: PaperResult to update.
            stats: Pipeline stats to update.
            auto_ingest: Whether this is an auto-ingest (no judge output).

        Returns:
            Updated PaperResult.
        """
        if self._dry_run:
            reason = result.auto_ingest_reason or "judge_accepted"
            logger.info(f"[DRY RUN] Would ingest ({reason}): {entry.arxiv_id}v{entry.version}")
            result.ingested = True
            stats.ingested += 1
            return result

        judge_output: JudgeOutput
        if auto_ingest and result.judge_output is None:
            judge_output = self._create_auto_ingest_judge_output(result)
        else:
            assert result.judge_output is not None
            judge_output = result.judge_output

        pdf_result = self._pdf_downloader.download(metadata.pdf_url)
        if not pdf_result.success:
            result.download_failed = True
            result.error = f"Download failed: {pdf_result.error_message}"
            stats.download_failed += 1
            logger.error(f"Download failed for {entry.arxiv_id}: {pdf_result.error_message}")
            return result

        from contextual_arxiv_feed.arxiv.pdf import compress_pdf_bytes

        original_size = len(pdf_result.pdf_bytes)
        pdf_bytes = compress_pdf_bytes(pdf_result.pdf_bytes)
        if len(pdf_bytes) < original_size:
            saved = original_size - len(pdf_bytes)
            logger.info(
                f"PDF compressed: {original_size / (1024*1024):.1f}MB → "
                f"{len(pdf_bytes) / (1024*1024):.1f}MB (saved {saved / 1024:.0f}KB)"
            )
        else:
            pdf_bytes = pdf_result.pdf_bytes

        # ---- ChromaDB: ALWAYS store, NO cap ----
        if self._chromadb:
            try:
                authors_str = "|".join(a.name for a in metadata.authors)
                pub_date = metadata.published.strftime("%Y-%m-%d") if metadata.published else ""
                self._chromadb.store_paper(
                    arxiv_id=metadata.arxiv_id,
                    version=metadata.version,
                    title=metadata.title,
                    topics=result.stage1_topics,
                    quality_i=judge_output.quality_i if judge_output else 80,
                    rationale=judge_output.rationale if judge_output else "",
                    pdf_bytes=pdf_bytes,
                    published=pub_date,
                    authors=authors_str,
                )
            except Exception as e:
                logger.warning(f"ChromaDB store failed for {entry.arxiv_id}: {e}")

        # ---- Contextual AI: optional (paid), respects daily cap ----
        if self._contextual:
            if MAX_INGEST_PER_RUN is not None and stats.contextual_ingested >= MAX_INGEST_PER_RUN:
                logger.info(
                    f"Contextual AI daily cap reached ({MAX_INGEST_PER_RUN}), "
                    f"skipping paid ingestion for: {entry.arxiv_id}"
                )
            else:
                # Delete old versions before ingesting new one
                if entry.version > 1:
                    deleted = self._delete_old_versions(entry.arxiv_id, entry.version)
                    if deleted > 0:
                        logger.info(f"Replaced {deleted} old version(s) of {entry.arxiv_id}")

                success = self._ingest_paper(
                    metadata, judge_output, pdf_bytes, stats, result.stage1_topics
                )
                if not success:
                    result.ingest_failed = True
                    stats.ingest_failed += 1
                else:
                    stats.contextual_ingested += 1

        result.ingested = True
        stats.ingested += 1
        return result

    def _create_auto_ingest_judge_output(self, result: PaperResult) -> JudgeOutput:
        """Create placeholder JudgeOutput for auto-ingested papers.

        Args:
            result: PaperResult with auto_ingest_reason set.

        Returns:
            JudgeOutput with auto-ingest metadata.
        """
        from contextual_arxiv_feed.judge.schema import QualityBreakdown

        reason = result.auto_ingest_reason
        venue = result.auto_ingest_venue

        return JudgeOutput(
            prompt_version=0,  # 0 indicates auto-ingest, no LLM evaluation
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
        """Check if paper already exists in any datastore."""
        if self._dry_run:
            return False
        if self._chromadb and self._chromadb.paper_exists(arxiv_id, version):
            return True
        return bool(self._contextual and self._contextual.document_exists(arxiv_id, version))

    def _delete_old_versions(self, arxiv_id: str, new_version: int) -> int:
        """Delete older versions of a paper before ingesting new version.

        Args:
            arxiv_id: arXiv paper ID.
            new_version: New version being ingested.

        Returns:
            Number of old versions deleted.
        """
        if self._dry_run or not self._contextual:
            return 0

        prefix = f"arxiv:{arxiv_id}v"
        existing = self._contextual.list_documents(prefix=prefix)

        deleted = 0
        for doc_name in existing:
            info = parse_document_name(doc_name)
            if info and info.arxiv_id == arxiv_id and info.version < new_version:
                if self._contextual.delete_document_by_name(doc_name):
                    logger.info(f"Deleted old version: {doc_name}")
                    deleted += 1
                else:
                    logger.warning(f"Failed to delete old version: {doc_name}")

        return deleted

    def _ingest_paper(
        self,
        metadata: ArxivMetadata,
        judge_output: JudgeOutput,
        pdf_bytes: bytes,
        stats: PipelineStats,
        topics: list[str] | None = None,
    ) -> bool:
        """Ingest PDF and manifest to datastore.

        Args:
            metadata: Full arXiv metadata.
            judge_output: Judge evaluation result.
            pdf_bytes: PDF content.
            stats: Pipeline stats for run_id.
            topics: Topic keys from Stage 1 matcher.

        Returns:
            True if both ingests succeeded.
        """
        custom_metadata = self._build_custom_metadata(metadata, judge_output, topics or [])

        pdf_result = self._contextual.ingest_pdf(
            metadata.arxiv_id,
            metadata.version,
            pdf_bytes,
            custom_metadata,
        )
        if not pdf_result.success:
            logger.error(f"PDF ingest failed: {pdf_result.error}")
            return False

        # No manifest ingestion — PDF + metadata is sufficient for RAG retrieval.
        # Judge scores and full metadata are in the 'source' field and audit logs.

        logger.info(f"Ingested: {metadata.arxiv_id}v{metadata.version}")
        return True

    def _build_custom_metadata(
        self,
        metadata: ArxivMetadata,
        judge_output: JudgeOutput,
        topics: list[str],
    ) -> dict[str, Any]:
        """Build custom_metadata dict for Contextual AI ingestion."""
        from contextual_arxiv_feed.contextual.metadata import build_paper_metadata

        breakdown = judge_output.quality_breakdown_i
        authors_str = "|".join(a.name for a in metadata.authors)
        pub_date = metadata.published.strftime("%Y-%m-%d") if metadata.published else ""

        return build_paper_metadata(
            arxiv_id=metadata.arxiv_id,
            version=metadata.version,
            title=metadata.title,
            categories=metadata.categories,
            primary_category=metadata.primary_category,
            authors=authors_str,
            publication_date=pub_date,
            doi=metadata.doi or "",
            journal_ref=metadata.journal_ref or "",
            comments=metadata.comments or "",
            topics=topics,
            quality_verdict=judge_output.quality_verdict,
            quality_i=judge_output.quality_i,
            novelty_i=breakdown.novelty_i,
            relevance_i=breakdown.relevance_i,
            technical_depth_i=breakdown.technical_depth_i,
            confidence_i=judge_output.confidence_i,
        )

    def _build_manifest(
        self, metadata: ArxivMetadata, judge_output: JudgeOutput, run_id: str
    ) -> dict[str, Any]:
        """Build manifest content (full metadata, everything).

        Args:
            metadata: Full arXiv metadata.
            judge_output: Full judge output.
            run_id: Pipeline run ID.

        Returns:
            Dict with all metadata.
        """
        return {
            "arxiv_metadata": metadata.to_dict(),
            "judge_output": judge_output.to_dict(),
            "discovery_channel": "daily_rss",
            "citation_enrichment": None,  # Will be populated by citations refresh
            "run_metadata": {
                "run_id": run_id,
                "ingested_at": datetime.utcnow().isoformat(),
                "pipeline": "daily",
            },
        }
