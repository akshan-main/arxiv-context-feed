"""LLM Judge system for Stage 2 abstract evaluation."""

from contextual_arxiv_feed.judge.judge import JudgeResult, create_judge
from contextual_arxiv_feed.judge.schema import JudgeOutput, QualityBreakdown

__all__ = ["JudgeResult", "JudgeOutput", "QualityBreakdown", "create_judge"]

# Conditionally export LLMJudge
try:
    from contextual_arxiv_feed.judge.llm_judge import LLMJudge  # noqa: F401

    __all__.append("LLMJudge")
except ImportError:
    pass
