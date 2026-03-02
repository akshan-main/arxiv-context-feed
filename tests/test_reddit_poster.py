"""Tests for Reddit posting agent."""



from contextual_arxiv_feed.reddit.poster import RedditConfig, RedditPoster


def make_paper(arxiv_id="2401.00001", quality_i=80, title="Test Paper", topics=None):
    return {
        "arxiv_id": arxiv_id,
        "version": 1,
        "title": title,
        "topics": topics if topics is not None else ["context-engineering"],
        "quality_i": quality_i,
        "rationale": "This paper is interesting because it does cool stuff.",
    }


class TestRedditConfig:
    """Test RedditConfig defaults."""

    def test_defaults(self):
        config = RedditConfig()
        assert config.enabled is False
        assert config.subreddits == []
        assert config.min_quality_i == 0
        assert config.max_posts_per_run == 0

    def test_no_limit_means_post_all(self):
        config = RedditConfig(
            enabled=True,
            subreddits=["TestSub"],
            max_posts_per_run=0,
            dry_run=True,
        )
        poster = RedditPoster(config)
        papers = [make_paper(arxiv_id=str(i), quality_i=60 + i) for i in range(20)]
        results = poster.post_top_papers(papers)
        assert len(results) == 20


class TestRedditPoster:
    """Test RedditPoster functionality."""

    def test_disabled_returns_empty(self):
        config = RedditConfig(enabled=False)
        poster = RedditPoster(config)
        results = poster.post_top_papers([make_paper()])
        assert results == []

    def test_no_subreddits_returns_empty(self):
        config = RedditConfig(enabled=True, subreddits=[], dry_run=True)
        poster = RedditPoster(config)
        results = poster.post_top_papers([make_paper()])
        assert results == []

    def test_quality_filtering(self):
        config = RedditConfig(
            enabled=True,
            subreddits=["TestSub"],
            min_quality_i=75,
            dry_run=True,
        )
        poster = RedditPoster(config)

        papers = [
            make_paper(arxiv_id="1", quality_i=80),
            make_paper(arxiv_id="2", quality_i=60),  # filtered out
            make_paper(arxiv_id="3", quality_i=90),
        ]

        results = poster.post_top_papers(papers)
        assert len(results) == 2
        assert results[0].arxiv_id == "3"
        assert results[1].arxiv_id == "1"

    def test_max_posts_cap(self):
        config = RedditConfig(
            enabled=True,
            subreddits=["TestSub"],
            min_quality_i=50,
            max_posts_per_run=2,
            dry_run=True,
        )
        poster = RedditPoster(config)

        papers = [make_paper(arxiv_id=str(i), quality_i=80) for i in range(5)]
        results = poster.post_top_papers(papers)
        assert len(results) == 2

    def test_build_title_format(self):
        config = RedditConfig(enabled=True, dry_run=True)
        topic_names = {"context-engineering": "Context Engineering"}
        poster = RedditPoster(config, topic_names=topic_names)

        paper = make_paper(title="Cool Paper", topics=["context-engineering"])
        title = poster._build_title(paper)
        assert title == "[Context Engineering] Cool Paper"

    def test_build_title_unknown_topic(self):
        config = RedditConfig(enabled=True, dry_run=True)
        poster = RedditPoster(config)

        paper = make_paper(title="Paper", topics=["unknown-topic"])
        title = poster._build_title(paper)
        assert title == "[unknown-topic] Paper"

    def test_build_body_contains_link(self):
        config = RedditConfig(enabled=True, dry_run=True)
        poster = RedditPoster(config)

        paper = make_paper(arxiv_id="2401.00001")
        body = poster._build_body(paper)
        assert "https://arxiv.org/abs/2401.00001" in body

    def test_build_body_contains_rationale(self):
        config = RedditConfig(enabled=True, dry_run=True)
        poster = RedditPoster(config)

        paper = make_paper()
        body = poster._build_body(paper)
        assert "cool stuff" in body

    def test_flair_override(self):
        config = RedditConfig(
            enabled=True,
            dry_run=True,
            flair_overrides={"context-engineering": "Custom Flair"},
        )
        poster = RedditPoster(config)

        paper = make_paper(topics=["context-engineering"])
        flair = poster._get_flair(paper)
        assert flair == "Custom Flair"

    def test_flair_auto_from_topic_name(self):
        config = RedditConfig(enabled=True, dry_run=True)
        topic_names = {"context-engineering": "Context Engineering & Management"}
        poster = RedditPoster(config, topic_names=topic_names)

        paper = make_paper(topics=["context-engineering"])
        flair = poster._get_flair(paper)
        assert flair == "Context Engineering & Management"

    def test_flair_uses_primary_topic_only(self):
        """Reddit only allows one flair per post — uses primary (first) topic."""
        config = RedditConfig(enabled=True, dry_run=True)
        topic_names = {
            "rag-retrieval": "RAG & Retrieval",
            "agents-tools": "Agents & Tools",
        }
        poster = RedditPoster(config, topic_names=topic_names)

        paper = make_paper(topics=["rag-retrieval", "agents-tools"])
        flair = poster._get_flair(paper)
        assert flair == "RAG & Retrieval"

    def test_multi_topic_all_listed_in_body(self):
        """All topics should appear in body even though flair is single."""
        config = RedditConfig(
            enabled=True,
            subreddits=["TestSub"],
            dry_run=True,
        )
        topic_names = {
            "rag-retrieval": "RAG & Retrieval",
            "agents-tools": "Agents & Tools",
        }
        poster = RedditPoster(config, topic_names=topic_names)

        paper = make_paper(topics=["rag-retrieval", "agents-tools"])
        body = poster._build_body(paper)
        assert "RAG & Retrieval" in body
        assert "Agents & Tools" in body

    def test_empty_topics_no_flair(self):
        config = RedditConfig(enabled=True, dry_run=True)
        poster = RedditPoster(config)
        paper = make_paper(topics=[])
        flair = poster._get_flair(paper)
        assert flair == ""
