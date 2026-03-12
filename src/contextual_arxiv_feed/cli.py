"""Command-line interface for contextual-arxiv-feed.

Commands:
- run-daily: Daily RSS ingestion pipeline
- backfill: Backfill papers from a date range
- backfill-date: Backfill papers from a single date
- backfill-identifiers: Backfill specific papers by ID/DOI/URL
- run-updates: Weekly updates pipeline
- refresh-citations: Citation refresh for papers with DOI
- prune-chromadb: Remove old papers from local ChromaDB
- dry-run: Execute pipelines without uploads
- validate-config: Validate YAML configuration
"""

from __future__ import annotations

import logging
import os
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

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def get_config_dir() -> Path:
    """Get config directory path."""
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

    Independent of ingestion status: a paper is considered accepted if it reached
    download/ingest stage (success or failure).
    """
    accepted = []
    for result in stats.results:
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


def _post_to_reddit(config, config_dir: Path, stats) -> int:
    """Post accepted papers to Reddit. Returns successful post count."""
    try:
        from contextual_arxiv_feed.reddit.poster import RedditPoster, load_reddit_config

        reddit_config = load_reddit_config(config_dir)
        if not reddit_config.enabled:
            return 0

        topic_names = {t.key: t.name for t in config.topics.get_enabled_topics()}
        poster = RedditPoster(reddit_config, topic_names=topic_names)
        accepted_papers = _get_accepted_papers(stats)
        results = poster.post_top_papers(accepted_papers)
        return sum(1 for result in results if result.success)
    except Exception as e:
        logger.warning(f"Reddit posting failed: {e}")
        return 0


@main.command("run-daily")
@click.option("--dry-run", is_flag=True, help="Run without downloading/storing")
@click.pass_context
def run_daily(ctx: click.Context, dry_run: bool) -> None:
    """Run daily RSS feed ingestion pipeline."""
    logger.info("Starting daily pipeline...")

    config_dir = get_config_dir()
    config = load_config(config_dir)

    if not config.contextual_api_key:
        logger.info("CONTEXTUAL_API_KEY not set - ChromaDB + Reddit only (no Contextual AI)")

    with DailyPipeline(config, dry_run=dry_run) as pipeline:
        stats = pipeline.run()

    reddit_posted = _post_to_reddit(config, config_dir, stats)
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


@main.command("backfill")
@click.option("--start", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", required=True, help="End date (YYYY-MM-DD)")
@click.option("--top-n", default=0, type=int, help="Limit to top N papers by citations per period (0=no limit)")
@click.option("--top-n-granularity", default="month", type=click.Choice(["month", "year"]), help="Period for top-N selection")
@click.option("--dry-run", is_flag=True, help="Run without ingesting")
@click.pass_context
def backfill(ctx: click.Context, start: str, end: str, top_n: int, top_n_granularity: str, dry_run: bool) -> None:
    """Backfill papers from a date range."""
    logger.info(f"Starting backfill: {start} to {end}")

    config_dir = get_config_dir()
    config = load_config(config_dir)

    if not dry_run and not config.contextual_api_key:
        click.echo("Error: CONTEXTUAL_API_KEY is required for non-dry-run backfill", err=True)
        sys.exit(1)

    try:
        start_date = datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.strptime(end, "%Y-%m-%d")
    except ValueError:
        click.echo(f"Error: invalid date format (expected YYYY-MM-DD)", err=True)
        sys.exit(1)

    if start_date > end_date:
        click.echo(f"Error: start date {start} is after end date {end}", err=True)
        sys.exit(1)

    with BackfillPipeline(config, dry_run=dry_run) as pipeline:
        stats = pipeline.run_date_range(start_date, end_date, top_n=top_n, top_n_granularity=top_n_granularity)

    generate_reports(stats, "backfill")

    click.echo("Backfill complete:")
    click.echo(f"  Candidates: {stats.candidates_total}")
    click.echo(f"  Stage 1 passed: {stats.stage1_passed}")
    click.echo(f"  Accepted: {stats.accepted}")
    click.echo(f"  Ingested: {stats.ingested}")
    click.echo(f"  Already existed: {stats.already_exists}")
    if stats.rejected_quality > 0:
        click.echo(f"  Rejected (quality): {stats.rejected_quality}")


@main.command("backfill-date")
@click.option("--date", required=True, help="Date to backfill (YYYY-MM-DD)")
@click.option("--top-n", default=0, type=int, help="Limit to top N papers by citations (0=no limit)")
@click.option("--top-n-granularity", default="month", type=click.Choice(["month", "year"]), help="Period for top-N (unused for single date)")
@click.option("--dry-run", is_flag=True, help="Run without ingesting")
@click.pass_context
def backfill_date(ctx: click.Context, date: str, top_n: int, top_n_granularity: str, dry_run: bool) -> None:
    """Backfill papers from a single date."""
    logger.info(f"Starting backfill for date: {date}")

    config_dir = get_config_dir()
    config = load_config(config_dir)

    if not dry_run and not config.contextual_api_key:
        click.echo("Error: CONTEXTUAL_API_KEY is required for non-dry-run backfill", err=True)
        sys.exit(1)

    try:
        target_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        click.echo(f"Error: invalid date '{date}' (expected YYYY-MM-DD)", err=True)
        sys.exit(1)

    with BackfillPipeline(config, dry_run=dry_run) as pipeline:
        stats = pipeline.run_single_date(target_date, top_n=top_n)

    generate_reports(stats, "backfill")

    click.echo(f"Backfill ({date}) complete:")
    click.echo(f"  Candidates: {stats.candidates_total}")
    click.echo(f"  Ingested: {stats.ingested}")
    click.echo(f"  Already existed: {stats.already_exists}")


@main.command("backfill-identifiers")
@click.option(
    "--identifier", "-i", multiple=True, required=True,
    help="arXiv ID, DOI, or arXiv URL (repeatable)",
)
@click.option("--dry-run", is_flag=True, help="Run without ingesting")
@click.pass_context
def backfill_identifiers(ctx: click.Context, identifier: tuple[str, ...], dry_run: bool) -> None:
    """Backfill specific papers by identifier (arXiv ID, DOI, or URL)."""
    logger.info(f"Starting backfill for {len(identifier)} identifier(s)")

    config_dir = get_config_dir()
    config = load_config(config_dir)

    if not dry_run and not config.contextual_api_key:
        click.echo("Error: CONTEXTUAL_API_KEY is required for non-dry-run backfill", err=True)
        sys.exit(1)

    with BackfillPipeline(config, dry_run=dry_run) as pipeline:
        stats = pipeline.run_identifiers(list(identifier))

    generate_reports(stats, "backfill")

    click.echo(f"Backfill (identifiers) complete:")
    click.echo(f"  Resolved: {stats.candidates_total}")
    click.echo(f"  Ingested: {stats.ingested}")
    click.echo(f"  Already existed: {stats.already_exists}")


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


@main.command("dry-run")
@click.option("--mode", type=click.Choice(["daily", "updates"]), default="daily", help="Pipeline to run")
@click.pass_context
def dry_run_cmd(ctx: click.Context, mode: str) -> None:
    """Run pipeline in dry-run mode (no uploads)."""
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
