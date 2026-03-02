"""Pytest fixtures for tests."""

from pathlib import Path

import pytest
import yaml

from contextual_arxiv_feed.config import (
    AppConfig,
    CategoriesConfig,
    JudgeConfig,
    SourcesConfig,
    TopicConfig,
    TopicsConfig,
)


@pytest.fixture
def sample_topic() -> TopicConfig:
    """Sample topic configuration."""
    return TopicConfig(
        key="test-topic",
        name="Test Topic",
        description="A test topic for unit tests",
        arxiv_categories=["cs.LG", "cs.AI"],
        keywords=["inference", "optimization", "quantization"],
        phrases=["large language model", "model serving"],
        negatives=["survey"],
        inclusion_notes="Include technical papers",
        exclusion_notes="Exclude surveys",
        enabled=True,
    )


@pytest.fixture
def sample_topics_config(sample_topic: TopicConfig) -> TopicsConfig:
    """Sample topics configuration."""
    return TopicsConfig(topics=[sample_topic])


@pytest.fixture
def sample_judge_config() -> JudgeConfig:
    """Sample judge configuration."""
    return JudgeConfig(
        provider="local",
        model_id="qwen2.5-14b-instruct-q4_k_m",
        prompt_version=3,
        strictness="medium",
    )


@pytest.fixture
def sample_sources_config() -> SourcesConfig:
    """Sample sources configuration."""
    return SourcesConfig(
        enable_openalex=True,
        citations_refresh_days=7,
    )


@pytest.fixture
def sample_categories_config() -> CategoriesConfig:
    """Sample categories configuration."""
    return CategoriesConfig(
        categories=[
            "cs.AI",
            "cs.CL",
            "cs.CV",
            "cs.LG",
            "stat.ML",
        ]
    )


@pytest.fixture
def sample_app_config(
    sample_topics_config: TopicsConfig,
    sample_judge_config: JudgeConfig,
    sample_sources_config: SourcesConfig,
    sample_categories_config: CategoriesConfig,
) -> AppConfig:
    """Sample application configuration."""
    return AppConfig(
        topics=sample_topics_config,
        judge=sample_judge_config,
        sources=sample_sources_config,
        categories=sample_categories_config,
    )


@pytest.fixture
def temp_config_dir(tmp_path: Path) -> Path:
    """Create temporary config directory with sample files."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Topics
    topics_data = {
        "topics": [
            {
                "key": "test-topic",
                "name": "Test Topic",
                "description": "Test description",
                "arxiv_categories": ["cs.LG"],
                "keywords": ["test", "keyword"],
                "phrases": ["test phrase"],
                "negatives": [],
                "enabled": True,
            }
        ]
    }
    with open(config_dir / "topics.yaml", "w") as f:
        yaml.dump(topics_data, f)

    # Judge
    judge_data = {
        "provider": "local",
        "model_id": "qwen2.5-14b-instruct-q4_k_m",
        "prompt_version": 3,
        "strictness": "medium",
    }
    with open(config_dir / "judge.yaml", "w") as f:
        yaml.dump(judge_data, f)

    # Sources
    sources_data = {
        "enable_openalex": True,
        "citations_refresh_days": 7,
    }
    with open(config_dir / "sources.yaml", "w") as f:
        yaml.dump(sources_data, f)

    # Categories
    categories_data = {
        "categories": ["cs.AI", "cs.CL", "cs.LG", "stat.ML"]
    }
    with open(config_dir / "categories.yaml", "w") as f:
        yaml.dump(categories_data, f)

    return config_dir


@pytest.fixture
def sample_judge_output_dict() -> dict:
    """Sample valid judge output as dict."""
    return {
        "prompt_version": 1,
        "model_id": "qwen2.5-14b-instruct-q4_k_m",
        "quality_verdict": "accept",
        "quality_i": 78,
        "quality_breakdown_i": {
            "novelty_i": 80,
            "relevance_i": 78,
            "technical_depth_i": 75,
        },
        "confidence_i": 85,
        "rationale": "The paper presents a novel approach to LLM inference optimization.",
    }


@pytest.fixture
def sample_arxiv_metadata_dict() -> dict:
    """Sample arXiv metadata as dict."""
    return {
        "arxiv_id": "2401.12345",
        "version": 1,
        "title": "Test Paper: A Novel Approach to LLM Inference",
        "abstract": "We present a novel approach to optimizing inference for large language models. Our method achieves significant speedups through quantization and efficient caching strategies.",
        "authors": [
            {"name": "John Doe", "affiliations": ["MIT"]},
            {"name": "Jane Smith", "affiliations": ["Stanford"]},
        ],
        "categories": ["cs.LG", "cs.CL"],
        "primary_category": "cs.LG",
        "published": "2024-01-15T00:00:00Z",
        "updated": "2024-01-15T00:00:00Z",
        "doi": "10.1234/test.12345",
        "journal_ref": "",
        "comments": "10 pages, 5 figures",
        "links": {"pdf": "https://arxiv.org/pdf/2401.12345.pdf"},
    }
