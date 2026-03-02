"""Reddit posting agent for sharing top arXiv papers.

Posts to your own dedicated subreddit. Topic flairs are auto-derived
from topics.yaml — no manual flair mapping needed.

Flair behavior:
- On first run, auto-creates flair templates for each topic on your subreddit.
- Each post gets its primary topic as a clickable flair.
- Reddit only allows ONE flair per post. All matched topics are listed in the body.
- Users can click a flair to see all papers in that topic.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Flair colors for topic categories (Reddit's predefined CSS classes)
FLAIR_COLORS = [
    ("#0079d3", "#ffffff"),  # blue/white
    ("#7b2292", "#ffffff"),  # purple/white
    ("#0e8a16", "#ffffff"),  # green/white
    ("#c24848", "#ffffff"),  # red/white
    ("#e09b00", "#000000"),  # amber/black
    ("#014980", "#ffffff"),  # navy/white
]


@dataclass
class RedditConfig:
    """Configuration for Reddit posting."""

    enabled: bool = False
    subreddits: list[str] = field(default_factory=list)
    min_quality_i: int = 0
    max_posts_per_run: int = 0
    flair_overrides: dict[str, str] = field(default_factory=dict)
    dry_run: bool = False


@dataclass
class PostResult:
    """Result of posting a paper to Reddit."""

    arxiv_id: str
    subreddit: str
    title: str
    success: bool = False
    post_url: str = ""
    error: str = ""


def load_reddit_config(config_dir: Path | None = None) -> RedditConfig:
    """Load Reddit config from YAML.

    Args:
        config_dir: Config directory path.

    Returns:
        RedditConfig.
    """
    if config_dir is None:
        config_dir = Path(__file__).parent.parent.parent.parent / "config"

    config_path = config_dir / "reddit.yaml"
    if not config_path.exists():
        return RedditConfig()

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    return RedditConfig(
        enabled=data.get("enabled", False),
        subreddits=data.get("subreddits", []),
        min_quality_i=data.get("min_quality_i", 0),
        max_posts_per_run=data.get("max_posts_per_run", 0),
        flair_overrides=data.get("flair_overrides", {}),
        dry_run=data.get("dry_run", False),
    )


class RedditPoster:
    """Posts top papers to Reddit.

    Posts go to your own dedicated subreddit(s). Uses PRAW for Reddit API.
    Topic flairs are auto-created on first run so users can click to filter.
    """

    def __init__(
        self,
        config: RedditConfig,
        topic_names: dict[str, str] | None = None,
    ):
        """Initialize Reddit poster.

        Args:
            config: Reddit configuration.
            topic_names: Mapping of topic_key -> topic_name for flair generation.
        """
        self._config = config
        self._topic_names = topic_names or {}
        self._reddit = None
        self._flairs_ensured: set[str] = set()  # subreddits where flairs are set up

        if config.enabled and not config.dry_run:
            self._init_reddit()

    def _init_reddit(self) -> None:
        """Initialize PRAW Reddit client."""
        try:
            import praw

            client_id = os.getenv("REDDIT_CLIENT_ID", "")
            client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
            username = os.getenv("REDDIT_USERNAME", "")
            password = os.getenv("REDDIT_PASSWORD", "")

            if not all([client_id, client_secret, username, password]):
                logger.warning("Reddit credentials not fully configured, disabling")
                self._config.enabled = False
                return

            self._reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                username=username,
                password=password,
                user_agent=f"arxiv-context-feed:v1.0 (by /u/{username})",
            )
            logger.info(f"Reddit client initialized as /u/{username}")

        except ImportError:
            logger.warning("PRAW not installed. Install with: pip install praw")
            self._config.enabled = False

    def _ensure_flairs(self, subreddit_name: str) -> None:
        """Ensure topic flair templates exist on the subreddit.

        Creates missing flairs so users can click to filter by topic.
        Only runs once per subreddit per session.

        Args:
            subreddit_name: Subreddit to set up flairs on.
        """
        if subreddit_name in self._flairs_ensured:
            return
        if not self._reddit:
            return

        try:
            subreddit = self._reddit.subreddit(subreddit_name)

            # Get existing flair templates
            existing = set()
            for flair in subreddit.flair.link_templates:
                existing.add(flair["text"])

            # Create missing flairs for each topic
            for i, (key, name) in enumerate(self._topic_names.items()):
                flair_text = self._config.flair_overrides.get(key, name)
                if flair_text not in existing:
                    bg_color, text_color = FLAIR_COLORS[i % len(FLAIR_COLORS)]
                    subreddit.flair.link_templates.add(
                        text=flair_text,
                        css_class="",
                        text_editable=False,
                        background_color=bg_color,
                        text_color="light" if text_color == "#ffffff" else "dark",
                    )
                    logger.info(f"Created flair '{flair_text}' on r/{subreddit_name}")

            self._flairs_ensured.add(subreddit_name)

        except Exception as e:
            logger.warning(f"Failed to ensure flairs on r/{subreddit_name}: {e}")
            # Non-fatal — posts will still work, just without colored flairs
            self._flairs_ensured.add(subreddit_name)

    def post_top_papers(self, accepted_papers: list[dict[str, Any]]) -> list[PostResult]:
        """Post top papers to configured subreddits.

        Args:
            accepted_papers: List of accepted paper dicts with quality_i, title, etc.

        Returns:
            List of PostResults.
        """
        if not self._config.enabled:
            return []

        if not self._config.subreddits:
            logger.warning("No subreddits configured for Reddit posting")
            return []

        # Filter by quality and sort
        qualified = [
            p for p in accepted_papers
            if p.get("quality_i", 0) >= self._config.min_quality_i
        ]
        qualified.sort(key=lambda p: p.get("quality_i", 0), reverse=True)
        if self._config.max_posts_per_run > 0:
            top_papers = qualified[: self._config.max_posts_per_run]
        else:
            top_papers = qualified  # 0 = no limit, post all

        if not top_papers:
            logger.info("No papers met the quality threshold for Reddit posting")
            return []

        results = []
        for subreddit_name in self._config.subreddits:
            # Ensure flairs exist before posting
            self._ensure_flairs(subreddit_name)

            for paper in top_papers:
                result = self._post_paper(paper, subreddit_name)
                results.append(result)

        posted = sum(1 for r in results if r.success)
        logger.info(f"Reddit: posted {posted}/{len(results)} papers")
        return results

    def _post_paper(self, paper: dict[str, Any], subreddit_name: str) -> PostResult:
        """Post a single paper to a subreddit.

        Args:
            paper: Paper dict with title, arxiv_id, topics, rationale.
            subreddit_name: Target subreddit name.

        Returns:
            PostResult.
        """
        title = self._build_title(paper)
        body = self._build_body(paper)
        result = PostResult(
            arxiv_id=paper.get("arxiv_id", ""),
            subreddit=subreddit_name,
            title=title,
        )

        if self._config.dry_run:
            logger.info(f"[DRY RUN] Would post to r/{subreddit_name}: {title}")
            result.success = True
            return result

        if not self._reddit:
            result.error = "Reddit client not initialized"
            return result

        try:
            subreddit = self._reddit.subreddit(subreddit_name)

            # Find the flair template ID for the primary topic
            flair_id, flair_text = self._find_flair_template(paper, subreddit)

            submission = subreddit.submit(
                title=title,
                selftext=body,
                flair_id=flair_id,
                flair_text=flair_text,
            )

            result.success = True
            result.post_url = f"https://reddit.com{submission.permalink}"
            logger.info(f"Posted to r/{subreddit_name}: {title} [flair: {flair_text}]")

        except Exception as e:
            result.error = str(e)
            logger.error(f"Reddit post failed: {e}")

        return result

    def _find_flair_template(
        self, paper: dict[str, Any], subreddit: Any
    ) -> tuple[str | None, str]:
        """Find the matching flair template ID for a paper's primary topic.

        Args:
            paper: Paper dict.
            subreddit: PRAW subreddit object.

        Returns:
            (flair_id, flair_text) tuple. flair_id is None if no template found.
        """
        flair_text = self._get_flair(paper)
        if not flair_text:
            return None, ""

        try:
            for template in subreddit.flair.link_templates:
                if template["text"] == flair_text:
                    return template["id"], flair_text
        except Exception:
            pass

        # No template found — use text-only flair (won't be clickable)
        return None, flair_text

    def _build_title(self, paper: dict[str, Any]) -> str:
        """Build Reddit post title.

        Format: [Primary Topic] Paper Title

        Args:
            paper: Paper dict.

        Returns:
            Post title string.
        """
        topics = paper.get("topics", [])
        primary_topic = topics[0] if topics else "Research"

        # Use topic name if available, otherwise use key
        topic_display = self._topic_names.get(primary_topic, primary_topic)

        title = paper.get("title", "Untitled")
        return f"[{topic_display}] {title}"

    def _build_body(self, paper: dict[str, Any]) -> str:
        """Build Reddit post body.

        Contains arXiv link, ALL matching topics, and judge rationale.

        Args:
            paper: Paper dict.

        Returns:
            Post body string.
        """
        arxiv_id = paper.get("arxiv_id", "")
        topics = paper.get("topics", [])
        rationale = paper.get("rationale", "")
        quality_i = paper.get("quality_i", 0)

        lines = [
            f"**Paper:** https://arxiv.org/abs/{arxiv_id}",
            "",
        ]

        if topics:
            topic_displays = [
                self._topic_names.get(t, t) for t in topics
            ]
            lines.append(f"**Topics:** {', '.join(topic_displays)}")
            lines.append("")

        if rationale:
            lines.append(f"**Why interesting:** {rationale}")
            lines.append("")

        lines.append(f"*Quality score: {quality_i}/100*")
        lines.append("")
        lines.append("---")
        lines.append("*Posted by arxiv-context-feed bot*")

        return "\n".join(lines)

    def _get_flair(self, paper: dict[str, Any]) -> str:
        """Get flair text for a paper's primary topic.

        Reddit only allows one flair per post. The primary topic is used.
        All topics are listed in the post body.

        Args:
            paper: Paper dict.

        Returns:
            Flair text string.
        """
        topics = paper.get("topics", [])
        if not topics:
            return ""

        primary = topics[0]

        # Check overrides first
        if primary in self._config.flair_overrides:
            return self._config.flair_overrides[primary]

        # Auto-derive from topic name
        return self._topic_names.get(primary, primary)
