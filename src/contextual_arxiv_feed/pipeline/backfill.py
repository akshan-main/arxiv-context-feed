"""Backfill pipeline for historical arXiv papers.

Supports three modes:
- Date range: ingest papers from start_date to end_date
- Single date: ingest papers from a specific date
- Identifiers: ingest specific papers by arXiv ID, DOI, or arXiv URL

Uses arXiv API search instead of RSS feeds.
Same two-stage filtering as daily pipeline.
Never posts to Reddit (backfill is silent ingestion only).
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from contextual_arxiv_feed.arxiv import ArxivAPI, ArxivMetadata, PDFDownloader
from contextual_arxiv_feed.arxiv.throttle import ArxivThrottle
from contextual_arxiv_feed.config import AppConfig
from contextual_arxiv_feed.contextual import ContextualClient
from contextual_arxiv_feed.contextual.metadata import build_paper_metadata
from contextual_arxiv_feed.judge import JudgeOutput, create_judge
from contextual_arxiv_feed.judge.schema import QualityBreakdown
from contextual_arxiv_feed.keys.rotator import KeyRotator
from contextual_arxiv_feed.matcher import KeywordMatcher
from contextual_arxiv_feed.pipeline.citations import OpenAlexClient
from contextual_arxiv_feed.pipeline.venue import detect_top_venue

logger = logging.getLogger(__name__)

MAX_RESULTS_PER_QUERY = 1000
# Stop processing at 2h30m to leave time for cleanup (workflow timeout is 3h).
WALL_CLOCK_LIMIT_SECONDS = 150 * 60

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")
ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(v\d+)?")
DOI_RE = re.compile(r"10\.\d{4,}/\S+")
ARXIV_DOI_RE = re.compile(r"10\.48550/arXiv\.(\d{4}\.\d{4,5})")


def resolve_identifier(identifier: str) -> tuple[str, str]:
    """Resolve an identifier to an arXiv ID.

    Args:
        identifier: arXiv ID, arXiv URL, or DOI.

    Returns:
        Tuple of (arxiv_id, identifier_type).

    Raises:
        ValueError: If identifier cannot be resolved.
    """
    identifier = identifier.strip()

    # arXiv URL
    url_match = ARXIV_URL_RE.search(identifier)
    if url_match:
        arxiv_id = url_match.group(1)
        version = url_match.group(2) or ""
        return f"{arxiv_id}{version}", "arxiv_url"

    # arXiv DOI (10.48550/arXiv.YYMM.NNNNN)
    doi_match = ARXIV_DOI_RE.match(identifier)
    if doi_match:
        return doi_match.group(1), "arxiv_doi"

    # Plain arXiv ID
    id_match = ARXIV_ID_RE.match(identifier)
    if id_match:
        arxiv_id = id_match.group(1)
        version = id_match.group(2) or ""
        return f"{arxiv_id}{version}", "arxiv_id"

    # Generic DOI — cannot directly resolve to arXiv without external API
    if DOI_RE.match(identifier):
        raise ValueError(
            f"Non-arXiv DOI '{identifier}' cannot be resolved directly. "
            "Use arXiv DOI format (10.48550/arXiv.YYMM.NNNNN) or arXiv ID."
        )

    raise ValueError(f"Cannot resolve identifier: '{identifier}'")


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
    mode: str = "date_range"
    start_date: datetime | None = None
    end_date: datetime | None = None
    identifiers: list[str] = field(default_factory=list)
    candidates_total: int = 0
    stage1_passed: int = 0
    stage1_failed: int = 0
    already_exists: int = 0
    auto_ingest_revised: int = 0
    auto_ingest_venue: int = 0
    stage2_passed: int = 0
    stage2_failed: int = 0
    accepted: int = 0
    rejected_topicality: int = 0
    rejected_quality: int = 0
    download_failed: int = 0
    ingest_failed: int = 0
    ingested: int = 0
    remaining_count: int = 0
    continuation_issue: str = ""
    results: list[BackfillResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "mode": self.mode,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "identifiers": self.identifiers,
            "candidates_total": self.candidates_total,
            "stage1_passed": self.stage1_passed,
            "stage1_failed": self.stage1_failed,
            "already_exists": self.already_exists,
            "auto_ingest_revised": self.auto_ingest_revised,
            "auto_ingest_venue": self.auto_ingest_venue,
            "stage2_passed": self.stage2_passed,
            "stage2_failed": self.stage2_failed,
            "accepted": self.accepted,
            "rejected_topicality": self.rejected_topicality,
            "rejected_quality": self.rejected_quality,
            "download_failed": self.download_failed,
            "ingest_failed": self.ingest_failed,
            "ingested": self.ingested,
            "remaining_count": self.remaining_count,
            "continuation_issue": self.continuation_issue,
        }


class BackfillPipeline:
    """Pipeline for backfilling historical arXiv papers.

    Uses arXiv API search_by_date_range or fetch_by_id.
    Same two-stage filtering as daily pipeline.
    Never posts to Reddit.
    """

    def __init__(
        self,
        config: AppConfig,
        dry_run: bool = False,
    ):
        self._config = config
        self._dry_run = dry_run or config.dry_run

        self._throttle = ArxivThrottle(config.arxiv_throttle_seconds)
        self._api = ArxivAPI(self._throttle)
        self._pdf_downloader = PDFDownloader(self._throttle, config.max_download_mb)
        self._matcher = KeywordMatcher(config.topics)
        self._judge = create_judge(config.judge, config.topics.get_enabled_topics())

        self._start_time = time.monotonic()

        self._contextual: ContextualClient | None = None
        if not self._dry_run and config.contextual_api_key:
            self._contextual = ContextualClient(
                config.contextual_api_key,
                config.contextual_datastore_id,
                config.contextual_base_url,
            )
            self._contextual.configure_standard_ingestion()

    def close(self) -> None:
        self._api.close()
        self._pdf_downloader.close()
        self._judge.close()
        if self._contextual:
            self._contextual.close()

    def __enter__(self) -> BackfillPipeline:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def run_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        top_n: int = 0,
        top_n_granularity: str = "month",
    ) -> BackfillStats:
        """Run backfill for a date range.

        Args:
            start_date: Start of date range.
            end_date: End of date range.
            top_n: If > 0, only process the top N papers (by citations) per period.
            top_n_granularity: Period for top_n selection ("month" or "year").
        """
        stats = BackfillStats(
            run_id=str(uuid.uuid4())[:8],
            started_at=datetime.utcnow(),
            mode="date_range",
            start_date=start_date,
            end_date=end_date,
        )

        logger.info(
            f"Backfill {stats.run_id}: {start_date.date()} to {end_date.date()}"
        )

        # If top_n is set, split into periods and take top N from each
        if top_n > 0:
            periods = self._split_into_periods(start_date, end_date, top_n_granularity)
            logger.info(f"Top-{top_n} mode: {len(periods)} {top_n_granularity}(s)")
            all_papers: list[ArxivMetadata] = []
            categories = self._get_categories()

            for period_start, period_end in periods:
                if self._should_stop():
                    break
                logger.info(f"Fetching {period_start.date()} to {period_end.date()}")
                period_papers = self._api.search_by_date_range(
                    period_start, period_end, categories, max_results=MAX_RESULTS_PER_QUERY,
                )
                logger.info(f"  Found {len(period_papers)} papers, selecting top {top_n}")
                sorted_papers = self._sort_by_citations(period_papers)
                all_papers.extend(sorted_papers[:top_n])

            papers = all_papers
        else:
            categories = self._get_categories()
            papers = self._api.search_by_date_range(
                start_date, end_date, categories, max_results=MAX_RESULTS_PER_QUERY,
            )
            # Sort by citation count so most impactful papers are processed first
            papers = self._sort_by_citations(papers)

        stats.candidates_total = len(papers)
        logger.info(f"Processing {len(papers)} papers total")

        if not top_n and len(papers) >= MAX_RESULTS_PER_QUERY:
            logger.warning(
                f"Hit max results ({MAX_RESULTS_PER_QUERY}). "
                "Split into smaller date ranges."
            )

        for i, metadata in enumerate(papers):
            if self._should_stop():
                remaining = papers[i:]
                logger.warning(
                    f"Time limit reached at paper {i}/{len(papers)}. "
                    f"{len(remaining)} papers remaining."
                )
                self._create_continuation_issue(remaining, stats)
                break

            if i > 0 and i % 50 == 0:
                logger.info(f"Progress: {i}/{len(papers)}")
            result = self._process_paper(metadata, stats)
            stats.results.append(result)

        stats.finished_at = datetime.utcnow()
        self._log_summary(stats)
        return stats

    def run_single_date(self, date: datetime, top_n: int = 0) -> BackfillStats:
        """Run backfill for a single date."""
        stats = self.run_date_range(date, date, top_n=top_n)
        stats.mode = "single_date"
        return stats

    def run_identifiers(self, identifiers: list[str]) -> BackfillStats:
        """Run backfill for specific identifiers."""
        stats = BackfillStats(
            run_id=str(uuid.uuid4())[:8],
            started_at=datetime.utcnow(),
            mode="identifiers",
            identifiers=identifiers,
        )

        arxiv_ids = []
        for ident in identifiers:
            try:
                arxiv_id, id_type = resolve_identifier(ident)
                arxiv_ids.append(arxiv_id)
                logger.info(f"Resolved {ident} -> {arxiv_id} ({id_type})")
            except ValueError as e:
                logger.error(str(e))

        if not arxiv_ids:
            logger.error("No valid identifiers to process")
            stats.finished_at = datetime.utcnow()
            return stats

        papers = self._api.fetch_by_ids(arxiv_ids)
        stats.candidates_total = len(papers)
        logger.info(f"Fetched metadata for {len(papers)} papers")

        for i, metadata in enumerate(papers):
            if self._should_stop():
                remaining = papers[i:]
                logger.warning(
                    f"Time limit reached at paper {i}/{len(papers)}. "
                    f"{len(remaining)} papers remaining."
                )
                self._create_continuation_issue(remaining, stats)
                break

            result = self._process_paper(metadata, stats)
            stats.results.append(result)

        stats.finished_at = datetime.utcnow()
        self._log_summary(stats)
        return stats

    def _get_categories(self) -> list[str]:
        categories = set()
        for topic in self._config.topics.get_enabled_topics():
            categories.update(topic.arxiv_categories)
        return list(categories)

    @staticmethod
    def _split_into_periods(
        start_date: datetime, end_date: datetime, granularity: str,
    ) -> list[tuple[datetime, datetime]]:
        """Split a date range into month or year periods."""
        periods: list[tuple[datetime, datetime]] = []
        current = start_date

        while current <= end_date:
            if granularity == "year":
                period_end = datetime(current.year, 12, 31)
            else:  # month
                # Last day of current month
                if current.month == 12:
                    period_end = datetime(current.year, 12, 31)
                else:
                    period_end = datetime(current.year, current.month + 1, 1) - timedelta(days=1)

            period_end = min(period_end, end_date)
            periods.append((current, period_end))

            # Advance to next period
            current = period_end + timedelta(days=1)

        return periods

    def _log_summary(self, stats: BackfillStats) -> None:
        logger.info(
            f"Backfill complete: {stats.ingested} ingested, "
            f"{stats.rejected_quality} rejected, "
            f"{stats.already_exists} already existed"
        )
        if stats.remaining_count > 0:
            logger.info(
                f"Time limit reached — {stats.remaining_count} papers remaining. "
                f"Continuation issue: {stats.continuation_issue}"
            )

    def _should_stop(self) -> bool:
        """Check if we're approaching the wall-clock time limit."""
        elapsed = time.monotonic() - self._start_time
        return elapsed >= WALL_CLOCK_LIMIT_SECONDS

    def _sort_by_citations(
        self, papers: list[ArxivMetadata],
    ) -> list[ArxivMetadata]:
        """Sort papers by citation count (highest first) using OpenAlex.

        Papers without a DOI or without citation data sort last (original order).
        """
        rotator = KeyRotator.from_environment(cooldown_seconds=60)
        pool = rotator.get_pool("openalex")
        client = OpenAlexClient(key_pool=pool)

        citation_map: dict[str, int] = {}
        looked_up = 0
        for paper in papers:
            doi = paper.doi
            if not doi:
                continue
            data = client.get_by_doi(doi)
            looked_up += 1
            if data:
                citation_map[paper.arxiv_id] = data.citation_count
            if looked_up % 50 == 0:
                logger.info(f"Citation lookup progress: {looked_up}/{len(papers)}")

        client.close()
        logger.info(
            f"Citation lookup done: {len(citation_map)} papers with data "
            f"out of {looked_up} looked up"
        )

        # Stable sort: papers with citations first (desc), then the rest in original order
        return sorted(
            papers,
            key=lambda p: citation_map.get(p.arxiv_id, -1),
            reverse=True,
        )

    def _create_continuation_issue(
        self,
        remaining_papers: list[ArxivMetadata],
        stats: BackfillStats,
    ) -> str:
        """Create a GitHub issue with remaining paper IDs for auto-continuation.

        Returns the issue URL or empty string on failure.
        """
        ids = [p.arxiv_id for p in remaining_papers]
        stats.remaining_count = len(ids)

        payload = {
            "request_type": "identifiers",
            "identifiers": ids,
            "dry_run": self._dry_run,
            "requested_by": "auto-continuation",
            "note": f"Continuation of run {stats.run_id} — {len(ids)} remaining papers",
        }

        title = f"[Backfill] continuation — {len(ids)} papers from run {stats.run_id}"
        body = (
            "## Auto-Continuation Backfill Request\n\n"
            f"**Parent run:** `{stats.run_id}`\n"
            f"**Remaining papers:** {len(ids)}\n"
            f"**Dry run:** `{self._dry_run}`\n"
            "\n### Payload\n\n"
            f"```json\n{json.dumps(payload, indent=2)}\n```\n\n"
            "---\n*Auto-created by backfill pipeline (time limit reached)*\n"
        )

        repo = os.getenv("GITHUB_REPOSITORY", "")
        if not repo:
            logger.warning("GITHUB_REPOSITORY not set — cannot create continuation issue")
            return ""

        try:
            result = subprocess.run(
                [
                    "gh", "issue", "create",
                    "--repo", repo,
                    "--title", title,
                    "--body", body,
                    "--label", "backfill",
                ],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                issue_url = result.stdout.strip()
                stats.continuation_issue = issue_url
                logger.info(f"Created continuation issue: {issue_url}")
                return issue_url
            else:
                logger.error(f"gh issue create failed: {result.stderr}")
                return ""
        except Exception as e:
            logger.error(f"Failed to create continuation issue: {e}")
            return ""

    def _process_paper(
        self, metadata: ArxivMetadata, stats: BackfillStats,
    ) -> BackfillResult:
        result = BackfillResult(
            arxiv_id=metadata.arxiv_id,
            version=metadata.version,
            title=metadata.title,
        )

        if self._check_exists(metadata.arxiv_id, metadata.version):
            result.skipped_exists = True
            stats.already_exists += 1
            return result

        abstract_snippet = metadata.abstract[:500] if metadata.abstract else ""
        match_result = self._matcher.match(metadata.title, abstract_snippet)

        if not match_result.passes_stage1:
            stats.stage1_failed += 1
            return result

        result.stage1_passed = True
        result.stage1_topics = match_result.matched_topics
        stats.stage1_passed += 1

        if metadata.version >= 2:
            result.auto_ingest_reason = "revised_version"
            stats.auto_ingest_revised += 1
            return self._download_and_ingest(metadata, result, stats, auto_ingest=True)

        venue_result = detect_top_venue(metadata.comments, metadata.journal_ref)
        if venue_result and venue_result.detected:
            result.auto_ingest_reason = "top_venue"
            result.auto_ingest_venue = venue_result.venue_display
            stats.auto_ingest_venue += 1
            return self._download_and_ingest(metadata, result, stats, auto_ingest=True)

        judge_result = self._judge.judge(metadata.title, metadata.abstract)
        if not judge_result.success:
            result.error = f"Judge error: {judge_result.error}"
            stats.stage2_failed += 1
            logger.warning(f"Judge failed for {metadata.arxiv_id}: {judge_result.error}")
            return result

        result.judge_output = judge_result.output
        result.stage2_passed = True
        stats.stage2_passed += 1

        output = judge_result.output
        min_quality = self._config.judge.get_thresholds().min_quality_i

        # Cross-batch validated (4 batches, 205 papers): simple q>=65 is most stable
        accepted = output.quality_i >= min_quality

        if accepted:
            stats.accepted += 1
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
        if self._dry_run:
            reason = result.auto_ingest_reason or "judge_accepted"
            logger.info(f"[DRY RUN] Would ingest ({reason}): {metadata.arxiv_id}v{metadata.version}")
            result.ingested = True
            stats.ingested += 1
            return result

        pdf_result = self._pdf_downloader.download(metadata.pdf_url)
        if not pdf_result.success:
            result.download_failed = True
            result.error = f"Download failed: {pdf_result.error_message}"
            stats.download_failed += 1
            return result

        judge_output: JudgeOutput
        if auto_ingest and result.judge_output is None:
            judge_output = self._create_auto_ingest_judge_output(result)
        else:
            assert result.judge_output is not None
            judge_output = result.judge_output

        success = self._ingest_paper(
            metadata, judge_output, pdf_result.pdf_bytes, stats.run_id, result
        )
        if success:
            result.ingested = True
            stats.ingested += 1
        else:
            result.ingest_failed = True
            stats.ingest_failed += 1

        return result

    def _create_auto_ingest_judge_output(self, result: BackfillResult) -> JudgeOutput:
        reason = result.auto_ingest_reason
        venue = result.auto_ingest_venue
        return JudgeOutput(
            prompt_version=0,
            model_id=f"auto_ingest:{reason}",
            quality_verdict="accept",
            quality_i=100 if reason == "top_venue" else 80,
            quality_breakdown_i=QualityBreakdown(
                novelty_i=80, relevance_i=80, technical_depth_i=80,
            ),
            confidence_i=100,
            rationale=f"Auto-ingested: {reason}" + (f" ({venue})" if venue else ""),
        )

    def _check_exists(self, arxiv_id: str, version: int) -> bool:
        if self._dry_run or not self._contextual:
            return False
        return self._contextual.document_exists(arxiv_id, version)

    def _ingest_paper(
        self,
        metadata: ArxivMetadata,
        judge_output: JudgeOutput,
        pdf_bytes: bytes,
        run_id: str,
        result: BackfillResult,
    ) -> bool:
        if not self._contextual:
            logger.warning("No Contextual AI client — skipping ingestion")
            return False

        breakdown = judge_output.quality_breakdown_i
        authors_str = "|".join(a.name for a in metadata.authors)
        pub_date = metadata.published.strftime("%Y-%m-%d") if metadata.published else ""

        custom_metadata = build_paper_metadata(
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
            topics=result.stage1_topics,
            quality_verdict=judge_output.quality_verdict,
            quality_i=judge_output.quality_i,
            novelty_i=breakdown.novelty_i,
            relevance_i=breakdown.relevance_i,
            technical_depth_i=breakdown.technical_depth_i,
            confidence_i=judge_output.confidence_i,
        )

        pdf_result = self._contextual.ingest_pdf(
            metadata.arxiv_id, metadata.version, pdf_bytes, custom_metadata,
        )
        if not pdf_result.success:
            logger.error(f"PDF ingest failed: {pdf_result.error}")
            return False

        logger.info(f"Ingested: {metadata.arxiv_id}v{metadata.version}")
        return True
