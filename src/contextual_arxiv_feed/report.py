"""Report generation for pipeline runs.

Generates:
- artifacts/run_summary.json (machine-readable)
- artifacts/run_summary.md (human-readable)
- JSONL logging for key events
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default artifacts directory
ARTIFACTS_DIR = Path("artifacts")


def ensure_artifacts_dir(artifacts_dir: Path = ARTIFACTS_DIR) -> None:
    """Ensure artifacts directory exists."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)


def generate_json_report(
    stats: Any,
    pipeline_name: str,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> Path:
    """Generate JSON run summary.

    Args:
        stats: Pipeline stats object (must have to_dict method).
        pipeline_name: Name of pipeline (daily, updates, citations).
        artifacts_dir: Output directory.

    Returns:
        Path to generated file.
    """
    ensure_artifacts_dir(artifacts_dir)

    report = {
        "pipeline": pipeline_name,
        "generated_at": datetime.utcnow().isoformat(),
        "stats": stats.to_dict(),
    }

    # Add top accepted papers for daily/updates
    if hasattr(stats, "results"):
        accepted = [
            {
                "arxiv_id": r.arxiv_id,
                "version": r.version,
                "title": r.title if hasattr(r, "title") else "",
                "topics": r.stage1_topics if hasattr(r, "stage1_topics") else [],
                "quality_i": r.judge_output.quality_i if r.judge_output else 0,
            }
            for r in stats.results
            if hasattr(r, "ingested") and r.ingested
        ][:20]
        report["top_accepted"] = accepted

        # Rejection reasons breakdown
        rejection_reasons = {}
        for r in stats.results:
            if hasattr(r, "stage1_passed") and not r.stage1_passed:
                rejection_reasons["stage1_no_match"] = rejection_reasons.get("stage1_no_match", 0) + 1
            elif hasattr(r, "skipped_exists") and r.skipped_exists:
                rejection_reasons["already_exists"] = rejection_reasons.get("already_exists", 0) + 1
            elif hasattr(r, "judge_output") and r.judge_output and r.judge_output.quality_verdict != "accept":
                rejection_reasons["quality"] = rejection_reasons.get("quality", 0) + 1

        report["rejection_reasons"] = rejection_reasons

    output_path = artifacts_dir / "run_summary.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Generated JSON report: {output_path}")
    return output_path


def generate_markdown_report(
    stats: Any,
    pipeline_name: str,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> Path:
    """Generate human-readable Markdown report.

    Args:
        stats: Pipeline stats object.
        pipeline_name: Name of pipeline.
        artifacts_dir: Output directory.

    Returns:
        Path to generated file.
    """
    ensure_artifacts_dir(artifacts_dir)

    lines = [
        f"# {pipeline_name.title()} Pipeline Run Summary",
        "",
        f"**Run ID:** {stats.run_id}",
        f"**Started:** {stats.started_at.isoformat()}",
        f"**Finished:** {stats.finished_at.isoformat() if stats.finished_at else 'In Progress'}",
        "",
        "## Statistics",
        "",
    ]

    # Add stats based on pipeline type
    stats_dict = stats.to_dict()
    for key, value in stats_dict.items():
        if key not in ("run_id", "started_at", "finished_at", "results"):
            lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")

    # Add top accepted papers
    if hasattr(stats, "results"):
        accepted = [r for r in stats.results if hasattr(r, "ingested") and r.ingested]
        if accepted:
            lines.extend([
                "",
                "## Top Accepted Papers",
                "",
            ])
            for r in accepted[:20]:
                title = r.title if hasattr(r, "title") else f"{r.arxiv_id}v{r.version}"
                if r.judge_output:
                    topics = r.stage1_topics if hasattr(r, "stage1_topics") else []
                    lines.append(
                        f"- **{r.arxiv_id}v{r.version}** - {title[:60]}... "
                        f"(quality: {r.judge_output.quality_i}, topics: {', '.join(topics)})"
                    )
                else:
                    lines.append(f"- **{r.arxiv_id}v{r.version}** - {title[:60]}...")

        # Failed downloads/ingests
        failed_downloads = [r for r in stats.results if hasattr(r, "download_failed") and r.download_failed]
        if failed_downloads:
            lines.extend([
                "",
                "## Failed Downloads",
                "",
            ])
            for r in failed_downloads:
                lines.append(f"- {r.arxiv_id}v{r.version}: {r.error}")

        failed_ingests = [r for r in stats.results if hasattr(r, "ingest_failed") and r.ingest_failed]
        if failed_ingests:
            lines.extend([
                "",
                "## Failed Ingests",
                "",
            ])
            for r in failed_ingests:
                lines.append(f"- {r.arxiv_id}v{r.version}: {r.error}")

    output_path = artifacts_dir / "run_summary.md"
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Generated Markdown report: {output_path}")
    return output_path


def generate_reports(
    stats: Any,
    pipeline_name: str,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> tuple[Path, Path]:
    """Generate both JSON and Markdown reports.

    Args:
        stats: Pipeline stats object.
        pipeline_name: Name of pipeline.
        artifacts_dir: Output directory.

    Returns:
        Tuple of (json_path, md_path).
    """
    json_path = generate_json_report(stats, pipeline_name, artifacts_dir)
    md_path = generate_markdown_report(stats, pipeline_name, artifacts_dir)
    return json_path, md_path
