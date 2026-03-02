"""LLM Judge for Stage 2 abstract evaluation.

Provides the JudgeResult dataclass and create_judge() factory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contextual_arxiv_feed.config import JudgeConfig, TopicConfig
from contextual_arxiv_feed.judge.schema import JudgeOutput

logger = logging.getLogger(__name__)

# Path to prompt templates
TEMPLATES_DIR = Path(__file__).parent / "prompt_templates"


@dataclass
class JudgeResult:
    """Result of judging a paper."""

    success: bool
    output: JudgeOutput | None = None
    error: str = ""
    raw_response: str = ""

    @property
    def is_accepted(self) -> bool:
        """Whether paper is accepted (passes both gates)."""
        return self.success and self.output is not None and self.output.is_accepted


def create_judge(
    config: JudgeConfig,
    topics: list[TopicConfig],
    key_pool: Any = None,
) -> Any:
    """Factory function to create judge.

    Args:
        config: Judge configuration.
        topics: List of enabled topics.
        key_pool: Optional KeyPool for API key rotation.

    Returns:
        Configured LLMJudge instance.
    """
    from contextual_arxiv_feed.judge.llm_judge import LLMJudge

    return LLMJudge(config, topics, key_pool=key_pool)
