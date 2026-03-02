"""Command-line interface for contextual-arxiv-feed.

Commands:
- run-analyze: Full daily pipeline on Oracle Cloud (analyze + ingest + reddit + audit log)
- run-daily: Legacy full pipeline (for GitHub Actions manual fallback)
- ingest-from-issue: Manual fallback to ingest from a decisions.json file
- run-updates: Run weekly updates pipeline
- refresh-citations: Refresh citation data for papers with DOI
- backfill: Backfill papers from a date range
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv

from contextual_arxiv_feed.config import load_config
from contextual_arxiv_feed.pipeline.backfill import BackfillPipeline
from contextual_arxiv_feed.pipeline.citations import CitationsRefresh
from contextual_arxiv_feed.pipeline.daily import DailyPipeline
from contextual_arxiv_feed.pipeline.updates import UpdatesPipeline
from contextual_arxiv_feed.report import generate_reports

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def get_config_dir() -> Path:
    """Get config directory path."""
    # Check for CONFIG_DIR env var, otherwise use default
    config_dir = os.getenv("CONFIG_DIR")
    if config_dir:
        return Path(config_dir)
    return Path(__file__).parent.parent.parent / "config"


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """Contextual arXiv Feed - Ingest arXiv papers into Contextual AI Datastore."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


def _get_accepted_papers(stats) -> list[dict]:
    """Get all accepted papers from pipeline stats.

    Independent of ingestion status — a paper is "accepted" if it reached
    the download/ingest step (whether it succeeded or not).
    """
    accepted = []
    for result in stats.results:
        # Paper was accepted if it reached download/ingest step
        if not (result.ingested or result.download_failed or result.ingest_failed):
            continue
        paper = {
            "arxiv_id": result.arxiv_id,
            "title": result.title,
            "topics": result.stage1_topics,
        }
        if result.judge_output:
            paper["quality_i"] = result.judge_output.quality_i
            paper["rationale"] = result.judge_output.rationale
        elif result.auto_ingest_reason:
            paper["quality_i"] = 100 if result.auto_ingest_reason == "top_venue" else 80
            paper["rationale"] = f"Auto: {result.auto_ingest_reason}"
        accepted.append(paper)
    return accepted


def _create_github_issue(audit_summary: dict) -> bool:
    """Create a GitHub Issue as an audit log of the daily run.

    This is purely informational — no workflow is triggered.
    Preserves a daily trail of what was analyzed, accepted, and ingested.

    Requires `gh` CLI to be installed and authenticated.

    Args:
        audit_summary: Dict with run stats, accepted papers, and ingestion results.

    Returns:
        True if issue was created successfully.
    """
    run_id = audit_summary.get("run_id", "unknown")
    date = audit_summary.get("date", "unknown")
    accepted_count = audit_summary.get("accepted_count", 0)
    ingested_count = audit_summary.get("ingested_count", 0)
    failed_count = audit_summary.get("failed_count", 0)

    title = f"Daily Run {date} — {ingested_count} ingested, {accepted_count} accepted (run {run_id})"

    body_lines = [
        f"## Daily Run Summary — {date}",
        "",
        f"- **Run ID:** {run_id}",
        f"- **Candidates:** {audit_summary.get('candidates_total', 0)}",
        f"- **Stage 1 passed:** {audit_summary.get('stage1_passed', 0)}",
        f"- **Discovery Agent found:** {audit_summary.get('discovery_agent_passed', 0)}",
        f"- **Accepted by judge:** {accepted_count}",
        f"- **Ingested to Contextual AI:** {ingested_count}",
        f"- **Failed:** {failed_count}",
        f"- **Reddit posts:** {audit_summary.get('reddit_posted', 0)}",
        "",
        "## Accepted Papers",
        "",
    ]

    for paper in audit_summary.get("papers", []):
        status = paper.get("status", "unknown")
        emoji = {"ingested": "+", "failed": "x", "skipped": "-"}.get(status, "?")
        body_lines.append(
            f"- [{emoji}] `{paper['arxiv_id']}` {paper.get('title', '')[:80]} "
            f"(q={paper.get('quality_i', '?')}, {status})"
        )

    body = "\n".join(body_lines)

    try:
        result = subprocess.run(
            [
                "gh", "issue", "create",
                "--title", title,
                "--body", body,
                "--label", "daily-run-log",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            issue_url = result.stdout.strip()
            logger.info(f"Created audit issue: {issue_url}")
            return True
        else:
            logger.error(f"gh issue create failed: {result.stderr}")
            return False
    except FileNotFoundError:
        logger.error("gh CLI not found. Install: https://cli.github.com/")
        return False
    except subprocess.TimeoutExpired:
        logger.error("gh issue create timed out")
        return False


def _post_to_reddit(config, config_dir: Path, stats) -> int:
    """Post accepted papers to Reddit. Returns count of successful posts.

    Fully independent of Contextual AI — runs even without ingestion.
    """
    try:
        from contextual_arxiv_feed.reddit.poster import RedditPoster, load_reddit_config

        reddit_config = load_reddit_config(config_dir)
        if not reddit_config.enabled:
            return 0
        topic_names = {t.key: t.name for t in config.topics.get_enabled_topics()}
        poster = RedditPoster(reddit_config, topic_names=topic_names)
        accepted_papers = _get_accepted_papers(stats)
        results = poster.post_top_papers(accepted_papers)
        return sum(1 for r in results if r.success)
    except Exception as e:
        logger.warning(f"Reddit posting failed: {e}")
        return 0


@main.command("run-daily")
@click.option("--dry-run", is_flag=True, help="Run without downloading/storing")
@click.pass_context
def run_daily(ctx: click.Context, dry_run: bool) -> None:
    """Run daily RSS feed ingestion pipeline.

    Always: analysis + ChromaDB storage + Reddit posting.
    Optional: Contextual AI ingestion (if CONTEXTUAL_API_KEY is set).
    """
    logger.info("Starting daily pipeline...")

    config_dir = get_config_dir()
    config = load_config(config_dir)

    if not config.contextual_api_key:
        logger.info("CONTEXTUAL_API_KEY not set — ChromaDB + Reddit only (no Contextual AI)")

    with DailyPipeline(config, dry_run=dry_run) as pipeline:
        stats = pipeline.run()

    # Post top papers to Reddit (independent of everything)
    reddit_posted = _post_to_reddit(config, config_dir, stats)

    # Generate reports
    generate_reports(stats, "daily")

    click.echo("Daily pipeline complete:")
    click.echo(f"  Candidates: {stats.candidates_total}")
    click.echo(f"  Stage 1 passed: {stats.stage1_passed}")
    click.echo(f"  Stage 2 passed: {stats.stage2_passed}")
    click.echo(f"  Accepted: {stats.accepted}")
    click.echo(f"  Stored (ChromaDB): {stats.ingested}")
    if reddit_posted > 0:
        click.echo(f"  Reddit posts: {reddit_posted}")

    if stats.download_failed > 0:
        click.echo(f"  Download failures: {stats.download_failed}", err=True)
    if stats.ingest_failed > 0:
        click.echo(f"  Contextual AI failures: {stats.ingest_failed}", err=True)


@main.command("run-analyze")
@click.option("--dry-run", is_flag=True, help="Run without uploading or creating issues")
@click.option("--output", "-o", default="decisions.json", help="Output file for decisions")
@click.pass_context
def run_analyze(ctx: click.Context, dry_run: bool, output: str) -> None:
    """Full daily pipeline for Oracle Cloud.

    Runs the entire pipeline on Oracle Cloud:
    1. RSS fetch + keyword matching + Discovery Agent + LLM judge
    2. Download PDFs + ingest to Contextual AI
    3. Post top papers to Reddit
    4. Create GitHub Issue as audit log
    """
    logger.info("Starting daily pipeline (Oracle Cloud)...")

    config_dir = get_config_dir()
    config = load_config(config_dir)

    # Stage 1-2: Analysis (RSS, keywords, discovery, judge)
    # Run with dry_run=True so DailyPipeline doesn't try to ingest
    with DailyPipeline(config, dry_run=True) as pipeline:
        stats = pipeline.run()

    # Collect accepted papers with full metadata
    accepted_papers = []
    for result in stats.results:
        if result.ingested or (result.stage2_passed and result.judge_output and result.judge_output.is_accepted):
            paper_data: dict = {
                "arxiv_id": result.arxiv_id,
                "version": result.version,
                "title": result.title,
                "topics": result.stage1_topics,
                "auto_ingest_reason": result.auto_ingest_reason,
            }
            if result.metadata:
                paper_data["arxiv_metadata"] = result.metadata.to_dict()
            if result.judge_output:
                paper_data["judge_output"] = result.judge_output.to_dict()
            elif result.auto_ingest_reason:
                paper_data["quality_i"] = 100 if result.auto_ingest_reason == "top_venue" else 80
                paper_data["rationale"] = f"Auto: {result.auto_ingest_reason}"
            accepted_papers.append(paper_data)

    if not accepted_papers:
        click.echo("No papers accepted. Nothing to ingest.")
        generate_reports(stats, "analyze")
        return

    # Write decisions to file (local audit copy)
    decisions = {
        "run_id": stats.run_id,
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "stats": stats.to_dict(),
        "accepted_papers": accepted_papers,
    }
    output_path = Path(output)
    with open(output_path, "w") as f:
        json.dump(decisions, f, indent=2)

    # Stage 3: Download PDFs + ingest to Contextual AI
    ingested_count = 0
    failed_count = 0
    skipped_count = 0
    paper_statuses: list[dict] = []

    if not dry_run and config.contextual_api_key:
        from contextual_arxiv_feed.arxiv import PDFDownloader
        from contextual_arxiv_feed.arxiv.throttle import ArxivThrottle
        from contextual_arxiv_feed.contextual import ContextualClient
        from contextual_arxiv_feed.judge.schema import (
            JudgeOutput,
            QualityBreakdown,
            parse_judge_output,
        )

        throttle = ArxivThrottle(config.arxiv_throttle_seconds)
        pdf_downloader = PDFDownloader(throttle, config.max_download_mb)
        contextual = ContextualClient(
            config.contextual_api_key,
            config.contextual_datastore_id,
            config.contextual_base_url,
        )
        contextual.configure_text_only_ingestion()

        try:
            for i, paper in enumerate(accepted_papers):
                arxiv_id = paper["arxiv_id"]
                version = paper.get("version", 1)
                title = paper.get("title", "")
                topics = paper.get("topics", [])

                if i > 0 and i % 50 == 0:
                    logger.info(f"Ingestion progress: {i}/{len(accepted_papers)}")

                # Check if already exists
                if contextual.document_exists(arxiv_id, version):
                    logger.info(f"Already exists: {arxiv_id}v{version}")
                    skipped_count += 1
                    paper_statuses.append({"arxiv_id": arxiv_id, "title": title, "quality_i": paper.get("judge_output", {}).get("quality_i", "?"), "status": "skipped"})
                    continue

                # Download PDF (one at a time — no disk/memory buildup)
                pdf_result = pdf_downloader.download_by_arxiv_id(arxiv_id, version)
                if not pdf_result.success:
                    logger.error(f"Download failed: {arxiv_id}v{version}: {pdf_result.error_message}")
                    failed_count += 1
                    paper_statuses.append({"arxiv_id": arxiv_id, "title": title, "quality_i": paper.get("judge_output", {}).get("quality_i", "?"), "status": "failed"})
                    continue

                # Reconstruct judge output
                judge_data = paper.get("judge_output")
                if judge_data:
                    judge_output = parse_judge_output(judge_data)
                else:
                    quality_i = paper.get("quality_i", 80)
                    auto_reason = paper.get("auto_ingest_reason", "")
                    judge_output = JudgeOutput(
                        prompt_version=0,
                        model_id=f"oracle_cloud:{auto_reason}" if auto_reason else "oracle_cloud:judge",
                        quality_verdict="accept",
                        quality_i=quality_i,
                        quality_breakdown_i=QualityBreakdown(
                            novelty_i=quality_i, relevance_i=quality_i, technical_depth_i=quality_i,
                        ),
                        confidence_i=100,
                        rationale=paper.get("rationale", f"Pre-approved (run {stats.run_id})"),
                    )

                # Build metadata
                arxiv_meta = paper.get("arxiv_metadata", {})
                authors_list = arxiv_meta.get("authors", [])
                authors_str = ", ".join(
                    a["name"] if isinstance(a, dict) else str(a) for a in authors_list
                )[:200] if authors_list else ""
                breakdown = judge_output.quality_breakdown_i

                custom_metadata = {
                    "arxiv_id": arxiv_id,
                    "arxiv_version": version,
                    "title": title,
                    "primary_category": arxiv_meta.get("primary_category", ""),
                    "categories": "|".join(arxiv_meta.get("categories", [])) if isinstance(arxiv_meta.get("categories"), list) else arxiv_meta.get("categories", ""),
                    "doi": arxiv_meta.get("doi", ""),
                    "year": arxiv_meta.get("year", 0),
                    "topics": "|".join(topics),
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
                    "authors": authors_str,
                    "publication_date": arxiv_meta.get("published", "") or "",
                    "paper_type": "",
                    "open_access": False,
                    "judge_model_id": judge_output.model_id,
                    "judge_prompt_version": judge_output.prompt_version,
                }

                # Upload PDF + manifest (then discard bytes)
                pdf_ingest = contextual.ingest_pdf(arxiv_id, version, pdf_result.pdf_bytes, custom_metadata)
                if not pdf_ingest.success:
                    logger.error(f"PDF ingest failed: {arxiv_id}v{version}: {pdf_ingest.error}")
                    failed_count += 1
                    paper_statuses.append({"arxiv_id": arxiv_id, "title": title, "quality_i": judge_output.quality_i, "status": "failed"})
                    continue

                manifest_content = {
                    "arxiv_metadata": arxiv_meta,
                    "judge_output": judge_output.to_dict(),
                    "discovery_channel": "oracle_cloud",
                    "run_metadata": {
                        "run_id": stats.run_id,
                        "ingested_at": datetime.utcnow().isoformat(),
                        "pipeline": "run_analyze",
                    },
                }
                manifest_ingest = contextual.ingest_manifest(arxiv_id, version, manifest_content, custom_metadata)
                if not manifest_ingest.success:
                    logger.error(f"Manifest ingest failed: {arxiv_id}v{version}: {manifest_ingest.error}")
                    failed_count += 1
                    paper_statuses.append({"arxiv_id": arxiv_id, "title": title, "quality_i": judge_output.quality_i, "status": "failed"})
                    continue

                ingested_count += 1
                paper_statuses.append({"arxiv_id": arxiv_id, "title": title, "quality_i": judge_output.quality_i, "status": "ingested"})
                logger.info(f"Ingested: {arxiv_id}v{version} - {title[:60]}")

        finally:
            pdf_downloader.close()
            contextual.close()
    elif dry_run:
        ingested_count = len(accepted_papers)
        paper_statuses = [{"arxiv_id": p["arxiv_id"], "title": p.get("title", ""), "quality_i": p.get("judge_output", {}).get("quality_i", "?"), "status": "ingested"} for p in accepted_papers]
    else:
        click.echo("Warning: CONTEXTUAL_API_KEY not set, skipping ingestion", err=True)

    # Post top papers to Reddit
    reddit_posted = _post_to_reddit(config, config_dir, stats)

    # Create GitHub Issue as audit log (not a trigger — just a record)
    issue_created = False
    if not dry_run:
        audit_summary = {
            "run_id": stats.run_id,
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "candidates_total": stats.candidates_total,
            "stage1_passed": stats.stage1_passed,
            "discovery_agent_passed": stats.discovery_agent_passed,
            "accepted_count": len(accepted_papers),
            "ingested_count": ingested_count,
            "failed_count": failed_count,
            "reddit_posted": reddit_posted,
            "papers": paper_statuses,
        }
        issue_created = _create_github_issue(audit_summary)

    generate_reports(stats, "analyze")

    click.echo("Daily pipeline complete:")
    click.echo(f"  Candidates: {stats.candidates_total}")
    click.echo(f"  Stage 1 passed: {stats.stage1_passed}")
    click.echo(f"  Discovery Agent found: {stats.discovery_agent_passed}")
    click.echo(f"  Accepted: {len(accepted_papers)}")
    click.echo(f"  Ingested: {ingested_count}")
    if skipped_count > 0:
        click.echo(f"  Skipped (exists): {skipped_count}")
    if failed_count > 0:
        click.echo(f"  Failed: {failed_count}", err=True)
    if reddit_posted > 0:
        click.echo(f"  Reddit posts: {reddit_posted}")
    if issue_created:
        click.echo("  Audit log: GitHub Issue created")
    click.echo(f"  Decisions saved: {output_path}")


@main.command("ingest-from-issue")
@click.option("--decisions", required=True, help="Path to decisions.json from Oracle Cloud analysis")
@click.option("--dry-run", is_flag=True, help="Run without uploading")
@click.pass_context
def ingest_from_issue(ctx: click.Context, decisions: str, dry_run: bool) -> None:
    """Manual fallback: ingest from a decisions.json file.

    Normally run-analyze handles everything on Oracle Cloud.
    Use this only if you need to re-ingest from a saved decisions file.
    No LLM keys needed — just downloads PDFs and uploads to Contextual AI.
    """
    decisions_path = Path(decisions)
    if not decisions_path.exists():
        click.echo(f"Error: {decisions_path} not found", err=True)
        sys.exit(1)

    with open(decisions_path) as f:
        payload = json.load(f)

    accepted_papers = payload.get("accepted_papers", [])
    run_id = payload.get("run_id", "unknown")

    if not accepted_papers:
        click.echo("No accepted papers to ingest.")
        sys.exit(0)

    logger.info(f"Ingesting {len(accepted_papers)} pre-approved papers (run {run_id})")

    config_dir = get_config_dir()
    config = load_config(config_dir)

    if not config.contextual_api_key and not dry_run:
        click.echo("Error: CONTEXTUAL_API_KEY is required", err=True)
        sys.exit(1)

    from contextual_arxiv_feed.arxiv import PDFDownloader
    from contextual_arxiv_feed.arxiv.throttle import ArxivThrottle
    from contextual_arxiv_feed.contextual import ContextualClient
    from contextual_arxiv_feed.judge.schema import JudgeOutput, QualityBreakdown, parse_judge_output

    throttle = ArxivThrottle(config.arxiv_throttle_seconds)
    pdf_downloader = PDFDownloader(throttle, config.max_download_mb)

    contextual: ContextualClient | None = None
    if not dry_run:
        contextual = ContextualClient(
            config.contextual_api_key,
            config.contextual_datastore_id,
            config.contextual_base_url,
        )
        contextual.configure_text_only_ingestion()

    ingested = 0
    failed = 0
    skipped = 0

    try:
        for i, paper in enumerate(accepted_papers):
            arxiv_id = paper["arxiv_id"]
            version = paper.get("version", 1)
            title = paper.get("title", "")
            topics = paper.get("topics", [])
            auto_reason = paper.get("auto_ingest_reason", "")

            if i > 0 and i % 50 == 0:
                logger.info(f"Progress: {i}/{len(accepted_papers)} papers processed")

            # Check if already exists
            if contextual and contextual.document_exists(arxiv_id, version):
                logger.info(f"Already exists: {arxiv_id}v{version}")
                skipped += 1
                continue

            if dry_run:
                logger.info(f"[DRY RUN] Would ingest: {arxiv_id}v{version}")
                ingested += 1
                continue

            # Download PDF
            pdf_result = pdf_downloader.download_by_arxiv_id(arxiv_id, version)
            if not pdf_result.success:
                logger.error(f"Download failed for {arxiv_id}v{version}: {pdf_result.error_message}")
                failed += 1
                continue

            # Full metadata already in payload from Oracle Cloud's run-analyze
            arxiv_meta = paper.get("arxiv_metadata", {})

            # Extract fields from payload metadata
            primary_category = arxiv_meta.get("primary_category", "")
            categories = arxiv_meta.get("categories", [])
            doi = arxiv_meta.get("doi", "")
            year = arxiv_meta.get("year", 0)
            authors_list = arxiv_meta.get("authors", [])
            authors_str = ", ".join(
                a["name"] if isinstance(a, dict) else str(a) for a in authors_list
            )[:200] if authors_list else ""
            pub_date = arxiv_meta.get("published", "") or ""

            # Reconstruct judge output from payload
            judge_data = paper.get("judge_output")
            if judge_data:
                judge_output = parse_judge_output(judge_data)
            else:
                quality_i = paper.get("quality_i", 80)
                judge_output = JudgeOutput(
                    prompt_version=0,
                    model_id=f"oracle_cloud:{auto_reason}" if auto_reason else "oracle_cloud:judge",
                    quality_verdict="accept",
                    quality_i=quality_i,
                    quality_breakdown_i=QualityBreakdown(
                        novelty_i=quality_i,
                        relevance_i=quality_i,
                        technical_depth_i=quality_i,
                    ),
                    confidence_i=100,
                    rationale=paper.get("rationale", f"Pre-approved (run {run_id})"),
                )

            # Build custom_metadata with real data from Oracle Cloud
            breakdown = judge_output.quality_breakdown_i
            custom_metadata = {
                "arxiv_id": arxiv_id,
                "arxiv_version": version,
                "title": title,
                "primary_category": primary_category,
                "categories": "|".join(categories) if isinstance(categories, list) else categories,
                "doi": doi,
                "year": year,
                "topics": "|".join(topics),
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
                "authors": authors_str,
                "publication_date": pub_date,
                "paper_type": "",
                "open_access": False,
                "judge_model_id": judge_output.model_id,
                "judge_prompt_version": judge_output.prompt_version,
            }

            # Ingest PDF
            assert contextual is not None
            pdf_ingest = contextual.ingest_pdf(arxiv_id, version, pdf_result.pdf_bytes, custom_metadata)
            if not pdf_ingest.success:
                logger.error(f"PDF ingest failed for {arxiv_id}v{version}: {pdf_ingest.error}")
                failed += 1
                continue

            # Ingest manifest
            manifest_content = {
                "arxiv_metadata": arxiv_meta,
                "judge_output": judge_output.to_dict(),
                "discovery_channel": "oracle_cloud_issue",
                "run_metadata": {
                    "run_id": run_id,
                    "ingested_at": datetime.utcnow().isoformat(),
                    "pipeline": "ingest_from_issue",
                },
            }

            manifest_ingest = contextual.ingest_manifest(
                arxiv_id, version, manifest_content, custom_metadata,
            )
            if not manifest_ingest.success:
                logger.error(f"Manifest ingest failed for {arxiv_id}v{version}: {manifest_ingest.error}")
                failed += 1
                continue

            ingested += 1
            logger.info(f"Ingested: {arxiv_id}v{version} - {title[:60]}")

    finally:
        pdf_downloader.close()
        if contextual:
            contextual.close()

    click.echo(f"Ingestion complete (run {run_id}):")
    click.echo(f"  Accepted: {len(accepted_papers)}")
    click.echo(f"  Ingested: {ingested}")
    click.echo(f"  Skipped (exists): {skipped}")
    if failed > 0:
        click.echo(f"  Failed: {failed}", err=True)


@main.command("run-updates")
@click.option("--lookback-days", default=7, help="Days to look back for updates")
@click.option("--dry-run", is_flag=True, help="Run without uploading")
@click.pass_context
def run_updates(ctx: click.Context, lookback_days: int, dry_run: bool) -> None:
    """Run weekly updates pipeline for new versions and DOI enrichment."""
    logger.info(f"Starting updates pipeline (lookback: {lookback_days} days)...")

    config_dir = get_config_dir()
    config = load_config(config_dir)

    if not config.contextual_api_key and not dry_run:
        click.echo("Error: CONTEXTUAL_API_KEY is required", err=True)
        sys.exit(1)

    with UpdatesPipeline(config, lookback_days=lookback_days, dry_run=dry_run) as pipeline:
        stats = pipeline.run()

    # Generate reports
    generate_reports(stats, "updates")

    click.echo("Updates pipeline complete:")
    click.echo(f"  Candidates: {stats.candidates_total}")
    click.echo(f"  New versions found: {stats.new_versions_found}")
    click.echo(f"  New versions ingested: {stats.new_versions_ingested}")
    click.echo(f"  DOI updates found: {stats.doi_updates_found}")
    click.echo(f"  DOI updates applied: {stats.doi_updates_applied}")


@main.command("refresh-citations")
@click.option("--dry-run", is_flag=True, help="Run without updating")
@click.pass_context
def refresh_citations(ctx: click.Context, dry_run: bool) -> None:
    """Refresh citation data for papers with DOI."""
    logger.info("Starting citations refresh...")

    config_dir = get_config_dir()
    config = load_config(config_dir)

    if not config.contextual_api_key and not dry_run:
        click.echo("Error: CONTEXTUAL_API_KEY is required", err=True)
        sys.exit(1)

    openalex_key = os.getenv("OPENALEX_API_KEY", "")

    with CitationsRefresh(config, openalex_key, dry_run=dry_run) as pipeline:
        stats = pipeline.run()

    # Generate reports
    generate_reports(stats, "citations")

    click.echo("Citations refresh complete:")
    click.echo(f"  Total documents: {stats.total_documents}")
    click.echo(f"  Documents with DOI: {stats.documents_with_doi}")
    click.echo(f"  Refreshed: {stats.refreshed}")
    click.echo(f"  Failed: {stats.failed}")
    click.echo(f"  Skipped (no DOI): {stats.skipped_no_doi}")


@main.command("prune-chromadb")
@click.option("--max-age-days", default=270, help="Delete papers older than this (default 270 = 9 months)")
@click.option("--dry-run", is_flag=True, help="Show what would be deleted without deleting")
@click.pass_context
def prune_chromadb(ctx: click.Context, max_age_days: int, dry_run: bool) -> None:
    """Prune old papers from ChromaDB to free disk space."""
    from contextual_arxiv_feed.store import ChromaDBStore

    store = ChromaDBStore()
    total_before = store._collection.count()

    if dry_run:
        click.echo(f"[DRY RUN] Would prune papers older than {max_age_days} days")
        click.echo(f"  Total chunks in ChromaDB: {total_before}")
        return

    deleted = store.prune_old_papers(max_age_days)
    total_after = store._collection.count()

    click.echo(f"Pruned {deleted} chunks from ChromaDB")
    click.echo(f"  Before: {total_before}")
    click.echo(f"  After: {total_after}")


@main.command("backfill")
@click.option("--start", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", required=True, help="End date (YYYY-MM-DD)")
@click.option("--dry-run", is_flag=True, help="Run without uploading")
@click.pass_context
def backfill(ctx: click.Context, start: str, end: str, dry_run: bool) -> None:
    """Backfill papers from a date range.

    Disabled by default. To enable, set BACKFILL_ENABLED = True
    in src/contextual_arxiv_feed/pipeline/backfill.py
    """
    from contextual_arxiv_feed.pipeline.backfill import BACKFILL_ENABLED

    if not BACKFILL_ENABLED:
        click.echo("Backfill is disabled. Set BACKFILL_ENABLED = True in pipeline/backfill.py")
        sys.exit(0)

    logger.info(f"Starting backfill from {start} to {end}...")

    try:
        start_date = datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.strptime(end, "%Y-%m-%d")
    except ValueError as e:
        click.echo(f"Error: Invalid date format: {e}", err=True)
        sys.exit(1)

    if start_date > end_date:
        click.echo("Error: Start date must be before end date", err=True)
        sys.exit(1)

    config_dir = get_config_dir()
    config = load_config(config_dir)

    if not config.contextual_api_key and not dry_run:
        click.echo("Error: CONTEXTUAL_API_KEY is required", err=True)
        sys.exit(1)

    with BackfillPipeline(config, start_date, end_date, dry_run=dry_run) as pipeline:
        stats = pipeline.run()

    # Generate reports
    generate_reports(stats, "backfill")

    click.echo("Backfill complete:")
    click.echo(f"  Date range: {start} to {end}")
    click.echo(f"  Papers found: {stats.candidates_total}")
    click.echo(f"  Stage 1 passed: {stats.stage1_passed}")
    click.echo(f"  Accepted: {stats.accepted}")
    click.echo(f"  Ingested: {stats.ingested}")

    if stats.download_failed > 0:
        click.echo(f"  Download failures: {stats.download_failed}", err=True)
    if stats.ingest_failed > 0:
        click.echo(f"  Ingest failures: {stats.ingest_failed}", err=True)


@main.command("dry-run")
@click.option("--mode", type=click.Choice(["daily", "updates"]), default="daily", help="Pipeline to run")
@click.pass_context
def dry_run_cmd(ctx: click.Context, mode: str) -> None:
    """Run pipeline in dry-run mode (no uploads).

    Prints decisions without actually ingesting papers.
    """
    logger.info(f"Starting dry-run in {mode} mode...")

    config_dir = get_config_dir()
    config = load_config(config_dir)

    if mode == "daily":
        with DailyPipeline(config, dry_run=True) as pipeline:
            stats = pipeline.run()
        generate_reports(stats, "daily_dryrun")
        click.echo(f"[DRY RUN] Would ingest {stats.accepted} papers")

    elif mode == "updates":
        with UpdatesPipeline(config, dry_run=True) as pipeline:
            stats = pipeline.run()
        generate_reports(stats, "updates_dryrun")
        click.echo(f"[DRY RUN] Would ingest {stats.new_versions_ingested} new versions")

    click.echo("Reports generated in artifacts/")


@main.command("validate-config")
@click.pass_context
def validate_config(ctx: click.Context) -> None:
    """Validate all configuration files."""
    logger.info("Validating configuration...")

    config_dir = get_config_dir()

    try:
        config = load_config(config_dir)

        click.echo("Configuration valid:")
        click.echo(f"  Topics: {len(config.topics.topics)} ({len(config.topics.get_enabled_topics())} enabled)")
        click.echo(f"  Categories: {len(config.categories.categories)}")
        click.echo(f"  Judge provider: {config.judge.provider}")
        click.echo(f"  Judge model: {config.judge.model_id}")
        click.echo(f"  Strictness: {config.judge.strictness}")

        # Validate topic categories
        from contextual_arxiv_feed.config import validate_topic_against_categories

        errors = []
        for topic in config.topics.topics:
            topic_errors = validate_topic_against_categories(topic, config.categories)
            errors.extend(topic_errors)

        if errors:
            click.echo("\nWarnings:", err=True)
            for error in errors:
                click.echo(f"  - {error}", err=True)
        else:
            click.echo("\nAll topic categories are valid.")

    except Exception as e:
        click.echo(f"Configuration error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
