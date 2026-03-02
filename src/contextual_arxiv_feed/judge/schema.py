"""Schema definitions for LLM judge output.

Strict validation with INT-ONLY enforcement for all numeric fields.
No floats allowed anywhere.

The judge's job is ONLY quality scoring + confidence + rationale.
Topic assignment is handled by Stage 1 (keyword matcher) + Discovery Agent.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class QualityBreakdown(BaseModel):
    """Breakdown of quality scores by dimension (abstract-appropriate criteria).

    All scores are integers 0-100. NO FLOATS.
    """

    novelty_i: int = Field(..., ge=0, le=100, description="Does it claim something new or improved?")
    relevance_i: int = Field(..., ge=0, le=100, description="How central/deep is the topic coverage?")
    technical_depth_i: int = Field(..., ge=0, le=100, description="Does it describe a concrete technical approach?")

    @field_validator("*", mode="before")
    @classmethod
    def enforce_int(cls, v: int | float) -> int:
        """Reject floats, enforce int only."""
        if isinstance(v, float):
            raise ValueError(
                f"Float values not allowed in judge output. Got {v}, expected int."
            )
        if not isinstance(v, int):
            raise ValueError(f"Expected int, got {type(v).__name__}: {v}")
        return v

    def compute_weighted_average(self) -> int:
        """Compute weighted average quality score.

        Novelty-weighted: Novelty=0.40, Relevance=0.35, Technical Depth=0.25
        """
        weighted = (
            self.novelty_i * 0.40
            + self.relevance_i * 0.35
            + self.technical_depth_i * 0.25
        )
        return int(round(weighted))


class JudgeOutput(BaseModel):
    """Complete output from the LLM judge.

    Judge focuses on quality scoring only. Topic assignment is done upstream
    by Stage 1 (keyword matcher) + Stage 1.5 (Discovery Agent).

    All numeric values must be integers. NO FLOATS.
    """

    model_config = {"protected_namespaces": ()}

    prompt_version: int = Field(..., description="Version of prompt template used")
    model_id: str = Field(..., description="LLM model ID used")
    quality_verdict: Literal["accept", "reject"] = Field(
        ..., description="Whether paper meets quality threshold"
    )
    quality_i: int = Field(..., ge=0, le=100, description="Overall quality score (0-100)")
    quality_breakdown_i: QualityBreakdown = Field(
        ..., description="Quality scores by dimension"
    )
    confidence_i: int = Field(
        ..., ge=0, le=100, description="LLM confidence in overall assessment (0-100)"
    )
    rationale: str = Field(..., description="Brief explanation of decision")

    @field_validator("prompt_version", "quality_i", "confidence_i", mode="before")
    @classmethod
    def enforce_int(cls, v: int | float) -> int:
        """Reject floats, enforce int only."""
        if isinstance(v, float):
            raise ValueError(
                f"Float values not allowed in judge output. Got {v}, expected int."
            )
        if not isinstance(v, int):
            raise ValueError(f"Expected int, got {type(v).__name__}: {v}")
        return v

    @field_validator("rationale")
    @classmethod
    def validate_rationale_length(cls, v: str) -> str:
        """Validate rationale length (will be truncated if too long)."""
        return v

    @property
    def is_accepted(self) -> bool:
        """Whether the paper passes quality check."""
        return self.quality_verdict == "accept"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt_version": self.prompt_version,
            "model_id": self.model_id,
            "quality_verdict": self.quality_verdict,
            "quality_i": self.quality_i,
            "quality_breakdown_i": {
                "novelty_i": self.quality_breakdown_i.novelty_i,
                "relevance_i": self.quality_breakdown_i.relevance_i,
                "technical_depth_i": self.quality_breakdown_i.technical_depth_i,
            },
            "confidence_i": self.confidence_i,
            "rationale": self.rationale,
        }


def parse_judge_output(json_data: dict[str, Any]) -> JudgeOutput:
    """Parse and validate judge output from JSON.

    Args:
        json_data: Raw JSON dict from LLM response.

    Returns:
        Validated JudgeOutput.

    Raises:
        ValueError: If validation fails (including any float values).
    """
    return JudgeOutput(**json_data)


def truncate_rationale(rationale: str, max_length: int = 300) -> str:
    """Truncate rationale to max length while preserving word boundaries."""
    if len(rationale) <= max_length:
        return rationale

    # Truncate at word boundary
    truncated = rationale[:max_length]
    last_space = truncated.rfind(" ")
    if last_space > max_length * 0.8:  # Don't cut off too much
        truncated = truncated[:last_space]

    return truncated + "..."
