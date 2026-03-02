"""Contextual arXiv Feed - Production-grade arXiv paper ingestion into Contextual AI Datastore."""

__version__ = "0.1.0"
__author__ = "Contextual AI Team"

from contextual_arxiv_feed.config import (
    JudgeConfig,
    SourcesConfig,
    TopicConfig,
    TopicsConfig,
    load_config,
)

__all__ = [
    "__version__",
    "JudgeConfig",
    "SourcesConfig",
    "TopicConfig",
    "TopicsConfig",
    "load_config",
]
