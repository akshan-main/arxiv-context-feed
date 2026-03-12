"""arXiv RSS/Atom feed parsing.

Fetches and parses RSS feeds for configured arXiv categories.
Extracts paper IDs, titles, and abstract snippets for Stage 1 filtering.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime

import feedparser
import httpx

from contextual_arxiv_feed.arxiv.throttle import ArxivThrottle, check_response_status

logger = logging.getLogger(__name__)

ARXIV_RSS_URL = "https://rss.arxiv.org/rss/{category}"

ARXIV_ID_PATTERN = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5})(v\d+)?")

VERSION_PATTERN = re.compile(r"v(\d+)$")


@dataclass
class FeedEntry:
    """A paper entry from an RSS feed."""

    arxiv_id: str  # e.g., "2401.12345"
    version: int  # e.g., 1 for v1
    title: str
    abstract_snippet: str  # RSS may have truncated abstract
    categories: list[str]
    authors: list[str]
    published: datetime | None
    updated: datetime | None
    link: str

    @property
    def id_with_version(self) -> str:
        """Return arxiv_id with version, e.g., '2401.12345v1'."""
        return f"{self.arxiv_id}v{self.version}"


class ArxivFeedParser:
    """Parses arXiv RSS feeds to extract paper entries."""

    def __init__(self, throttle: ArxivThrottle | None = None):
        """Initialize parser.

        Args:
            throttle: Rate limiter for requests. Creates default if None.
        """
        self._throttle = throttle or ArxivThrottle()
        self._client = httpx.Client(timeout=30.0)

    def close(self) -> None:
        """Close HTTP client."""
        self._client.close()

    def __enter__(self) -> ArxivFeedParser:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def fetch_feed(self, category: str) -> list[FeedEntry]:
        """Fetch and parse RSS feed for a category.

        Args:
            category: arXiv category (e.g., "cs.LG").

        Returns:
            List of FeedEntry objects.
        """
        url = ARXIV_RSS_URL.format(category=category)
        logger.info(f"Fetching RSS feed for {category}: {url}")

        self._throttle.sync_wait()

        try:
            response = self._client.get(url)
            check_response_status(response.status_code, url)

            if response.status_code != 200:
                logger.error(f"Failed to fetch feed {url}: {response.status_code}")
                return []

            return self._parse_feed(response.text, category)

        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching feed {url}: {e}")
            return []

    def _parse_feed(self, content: str, source_category: str) -> list[FeedEntry]:
        """Parse RSS feed content.

        Args:
            content: Raw RSS XML content.
            source_category: Category the feed was fetched from.

        Returns:
            List of FeedEntry objects.
        """
        feed = feedparser.parse(content)
        entries = []

        for entry in feed.entries:
            parsed = self._parse_entry(entry, source_category)
            if parsed:
                entries.append(parsed)

        logger.info(f"Parsed {len(entries)} entries from {source_category} feed")
        return entries

    def _parse_entry(self, entry: dict, source_category: str) -> FeedEntry | None:
        """Parse a single feed entry.

        Args:
            entry: feedparser entry dict.
            source_category: Category the feed was fetched from.

        Returns:
            FeedEntry or None if parsing fails.
        """
        try:
            link = entry.get("link", "")
            match = ARXIV_ID_PATTERN.search(link)
            if not match:
                logger.debug(f"Could not extract arXiv ID from link: {link}")
                return None

            arxiv_id = match.group(1)
            version_str = match.group(2)
            version = 1
            if version_str:
                version = int(version_str[1:])  # Remove 'v' prefix

            title = entry.get("title", "").strip()
            title = " ".join(title.split())  # Normalize whitespace

            abstract_snippet = entry.get("summary", "").strip()
            abstract_snippet = " ".join(abstract_snippet.split())

            categories = [source_category]
            if "tags" in entry:
                for tag in entry.tags:
                    term = tag.get("term", "")
                    if term and term not in categories:
                        categories.append(term)

            authors = []
            if "authors" in entry:
                for author in entry.authors:
                    name = author.get("name", "")
                    if name:
                        authors.append(name)

            published = None
            updated = None
            if "published_parsed" in entry and entry.published_parsed:
                published = datetime(*entry.published_parsed[:6])
            if "updated_parsed" in entry and entry.updated_parsed:
                updated = datetime(*entry.updated_parsed[:6])

            return FeedEntry(
                arxiv_id=arxiv_id,
                version=version,
                title=title,
                abstract_snippet=abstract_snippet,
                categories=categories,
                authors=authors,
                published=published,
                updated=updated,
                link=link,
            )

        except Exception as e:
            logger.warning(f"Error parsing feed entry: {e}")
            return None

    def fetch_multiple_feeds(self, categories: list[str]) -> list[FeedEntry]:
        """Fetch feeds for multiple categories and deduplicate.

        Args:
            categories: List of arXiv categories.

        Returns:
            Deduplicated list of FeedEntry objects.
        """
        all_entries: dict[str, FeedEntry] = {}

        for category in categories:
            entries = self.fetch_feed(category)
            for entry in entries:
                # Use arxiv_id+version as key for deduplication
                key = entry.id_with_version
                if key not in all_entries:
                    all_entries[key] = entry
                else:
                    # Merge categories from duplicate entries
                    existing = all_entries[key]
                    for cat in entry.categories:
                        if cat not in existing.categories:
                            existing.categories.append(cat)

        logger.info(
            f"Fetched {len(all_entries)} unique papers from {len(categories)} categories"
        )
        return list(all_entries.values())
