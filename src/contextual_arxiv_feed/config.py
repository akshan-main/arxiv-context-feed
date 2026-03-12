"""Configuration models and loaders for contextual-arxiv-feed.

All configuration is loaded from YAML files and validated with Pydantic.
Strict int-only enforcement for all numeric fields.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

TOPIC_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9]*([_-][a-z0-9]+)*$")

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


class TopicConfig(BaseModel):
    """Configuration for a single topic."""

    key: str = Field(..., description="Unique identifier for the topic")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(default="", description="Topic description")
    arxiv_categories: list[str] = Field(
        default_factory=list, description="arXiv categories to monitor"
    )
    keywords: list[str] = Field(
        default_factory=list, description="Single words for stemmed matching"
    )
    phrases: list[str] = Field(
        default_factory=list, description="Multi-word phrases for substring matching"
    )
    negatives: list[str] = Field(
        default_factory=list, description="Keywords/phrases to exclude"
    )
    inclusion_notes: str = Field(default="", description="Notes on what to include")
    exclusion_notes: str = Field(default="", description="Notes on what to exclude")
    enabled: bool = Field(default=True, description="Whether this topic is active")

    @field_validator("key")
    @classmethod
    def validate_key(cls, v: str) -> str:
        """Validate topic key matches required pattern."""
        if not TOPIC_KEY_PATTERN.match(v):
            raise ValueError(
                f"Topic key '{v}' must match pattern: lowercase alphanumeric with "
                "optional hyphens/underscores (e.g., 'llm-inference', 'ml_ops')"
            )
        return v

    @model_validator(mode="after")
    def validate_has_matching_criteria(self) -> TopicConfig:
        """Ensure topic has at least keywords or phrases for matching."""
        if not self.keywords and not self.phrases:
            raise ValueError(
                f"Topic '{self.key}' must have at least one keyword or phrase for matching"
            )
        return self


class TopicsConfig(BaseModel):
    """Configuration for all topics."""

    topics: list[TopicConfig] = Field(default_factory=list)

    def get_enabled_topics(self) -> list[TopicConfig]:
        """Return only enabled topics."""
        return [t for t in self.topics if t.enabled]

    def get_topic(self, key: str) -> TopicConfig | None:
        """Get topic by key."""
        for topic in self.topics:
            if topic.key == key:
                return topic
        return None


class StrictnessPreset(BaseModel):
    """Quality thresholds for a strictness level."""

    min_quality_i: int = Field(..., ge=0, le=100)

    @field_validator("min_quality_i", mode="before")
    @classmethod
    def enforce_int(cls, v: int | float) -> int:
        """Reject floats, enforce int only."""
        if isinstance(v, float):
            raise ValueError(f"Float values not allowed. Got {v}, expected int.")
        return v


class JudgeConfig(BaseModel):
    """Configuration for the LLM judge."""

    model_config = {"protected_namespaces": ()}

    provider: str = Field(default="gemini", description="LLM provider")
    model_id: str = Field(default="gemini-2.5-flash", description="Model ID")
    prompt_version: int = Field(default=1, description="Prompt template version")
    strictness: Literal["low", "medium", "high"] = Field(
        default="medium", description="Strictness preset"
    )
    strictness_presets: dict[str, StrictnessPreset] = Field(
        default_factory=lambda: {
            "low": StrictnessPreset(min_quality_i=40),
            "medium": StrictnessPreset(min_quality_i=60),
            "high": StrictnessPreset(min_quality_i=80),
        }
    )
    max_rationale_length: int = Field(default=300, ge=50, le=1000)

    # Whitelisted editable fields via Google Form
    editable_fields: list[str] = Field(
        default_factory=lambda: [
            "provider",
            "model_id",
            "strictness",
        ]
    )

    @field_validator("prompt_version", "max_rationale_length", mode="before")
    @classmethod
    def enforce_int(cls, v: int | float) -> int:
        """Reject floats, enforce int only."""
        if isinstance(v, float):
            raise ValueError(f"Float values not allowed. Got {v}, expected int.")
        return v

    def get_thresholds(self) -> StrictnessPreset:
        """Get current strictness thresholds."""
        return self.strictness_presets[self.strictness]


class SourcesConfig(BaseModel):
    """Configuration for citation sources."""

    enable_openalex: bool = Field(default=True, description="Use OpenAlex API")
    citations_refresh_days: int = Field(
        default=7, description="Days between citation refreshes"
    )
    openalex_rate_limit_per_second: int = Field(default=10)
    key_cooldown_seconds: int = Field(
        default=60, description="Cooldown in seconds after API key rate limit"
    )

    @field_validator(
        "citations_refresh_days",
        "openalex_rate_limit_per_second",
        "key_cooldown_seconds",
        mode="before",
    )
    @classmethod
    def enforce_int(cls, v: int | float) -> int:
        """Reject floats, enforce int only."""
        if isinstance(v, float):
            raise ValueError(f"Float values not allowed. Got {v}, expected int.")
        return v


class CategoriesConfig(BaseModel):
    """Allowlist of valid arXiv categories."""

    categories: list[str] = Field(default_factory=list)

    def is_valid(self, category: str) -> bool:
        """Check if category is in allowlist."""
        return category in self.categories


class AppConfig(BaseModel):
    """Combined application configuration."""

    topics: TopicsConfig
    judge: JudgeConfig
    sources: SourcesConfig
    categories: CategoriesConfig

    contextual_api_key: str = Field(default="")
    contextual_datastore_id: str = Field(default="")
    contextual_base_url: str = Field(default="https://api.contextual.ai")
    judge_provider: str = Field(default="")
    judge_model_id: str = Field(default="")
    llm_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai"
    )
    llm_api_key: str = Field(default="")
    arxiv_throttle_seconds: int = Field(default=3)
    max_download_mb: int = Field(default=100)
    dry_run: bool = Field(default=False)

    @field_validator("arxiv_throttle_seconds", "max_download_mb", mode="before")
    @classmethod
    def enforce_int(cls, v: int | float) -> int:
        """Reject floats, enforce int only."""
        if isinstance(v, float):
            raise ValueError(f"Float values not allowed. Got {v}, expected int.")
        return v


def load_yaml(path: Path) -> dict:
    """Load YAML file."""
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(config_dir: Path | None = None) -> AppConfig:
    """Load all configuration from YAML files and environment."""
    config_dir = config_dir or CONFIG_DIR

    topics_data = load_yaml(config_dir / "topics.yaml")
    judge_data = load_yaml(config_dir / "judge.yaml")
    sources_data = load_yaml(config_dir / "sources.yaml")
    categories_data = load_yaml(config_dir / "categories.yaml")

    judge_provider = os.getenv("JUDGE_PROVIDER", "")
    judge_model_id = os.getenv("JUDGE_MODEL_ID", "")
    if judge_provider:
        judge_data["provider"] = judge_provider
    if judge_model_id:
        judge_data["model_id"] = judge_model_id

    return AppConfig(
        topics=TopicsConfig(**topics_data),
        judge=JudgeConfig(**judge_data),
        sources=SourcesConfig(**sources_data),
        categories=CategoriesConfig(**categories_data),
        contextual_api_key=os.getenv("CONTEXTUAL_API_KEY", ""),
        contextual_datastore_id=os.getenv("CONTEXTUAL_DATASTORE_ID", ""),
        contextual_base_url=os.getenv("CONTEXTUAL_BASE_URL", "https://api.contextual.ai"),
        judge_provider=judge_provider,
        judge_model_id=judge_model_id,
        llm_base_url=os.getenv(
            "LLM_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        ),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        arxiv_throttle_seconds=int(os.getenv("ARXIV_THROTTLE_SECONDS", "3")),
        max_download_mb=int(os.getenv("MAX_DOWNLOAD_MB", "100")),
        dry_run=os.getenv("DRY_RUN", "").lower() in ("true", "1", "yes"),
    )


def validate_topic_against_categories(
    topic: TopicConfig, categories: CategoriesConfig
) -> list[str]:
    """Validate topic's arXiv categories against allowlist. Return list of errors."""
    errors = []
    for cat in topic.arxiv_categories:
        if not categories.is_valid(cat):
            errors.append(f"Invalid arXiv category '{cat}' in topic '{topic.key}'")
    return errors
