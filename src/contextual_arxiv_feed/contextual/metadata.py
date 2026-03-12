"""Build Contextual AI custom_metadata for arXiv papers.

Workspace limit: 35 metadata fields shared across ALL datastores
(arXiv, Reddit, future blog). 2KB total metadata size per document.

Shared fields with Reddit (2): title, url
arXiv-specific fields (18):
  arxiv_id, categories, primary_category, authors, published,
  source, pdf_url, doi, journal_ref, comments,
  topics, quality_verdict, quality_i, confidence_i,
  novelty_i, relevance_i, technical_depth_i, citation_count

Total workspace usage: 20 (arXiv) + 14 (Reddit) - 2 (shared) = 32 / 35
"""

from __future__ import annotations

from typing import Any


def build_paper_metadata(
    arxiv_id: str,
    version: int,
    title: str,
    categories: str | list[str],
    primary_category: str = "",
    authors: str = "",
    publication_date: str = "",
    doi: str = "",
    journal_ref: str = "",
    comments: str = "",
    topics: str | list[str] = "",
    quality_verdict: str = "",
    quality_i: int = 0,
    novelty_i: int = 0,
    relevance_i: int = 0,
    technical_depth_i: int = 0,
    confidence_i: int = 0,
    citation_count: int = 0,
) -> dict[str, Any]:
    """Build metadata with individual fields.

    Uses 20 fields. Truncates authors to 200 chars, title to 200 chars,
    comments to 200 chars to stay within 2KB limit.
    """
    if isinstance(categories, list):
        categories = "|".join(categories)
    if isinstance(topics, list):
        topics = "|".join(topics)

    publication_date = str(publication_date)[:10]

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}v{version}.pdf"

    metadata: dict[str, Any] = {
        "title": title[:200],
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "arxiv_id": f"{arxiv_id}v{version}",
        "categories": categories,
        "primary_category": primary_category,
        "authors": (authors or "")[:200],
        "published": publication_date,
        "source": pdf_url,
        "pdf_url": pdf_url,
        "doi": doi,
        "journal_ref": journal_ref[:200] if journal_ref else "",
        "comments": (comments or "")[:200],
        "topics": topics,
        "quality_verdict": quality_verdict,
        "quality_i": quality_i,
        "confidence_i": confidence_i,
        "novelty_i": novelty_i,
        "relevance_i": relevance_i,
        "technical_depth_i": technical_depth_i,
        "citation_count": citation_count,
    }

    return metadata
