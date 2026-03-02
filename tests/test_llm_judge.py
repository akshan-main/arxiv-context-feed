"""Tests for LLM judge and factory routing."""

from unittest.mock import MagicMock, patch

from contextual_arxiv_feed.config import JudgeConfig, TopicConfig
from contextual_arxiv_feed.judge.judge import create_judge


def make_topic(key="test-topic", name="Test Topic"):
    return TopicConfig(
        key=key,
        name=name,
        description="Test description",
        arxiv_categories=["cs.AI"],
        keywords=["test"],
        phrases=["test phrase"],
        inclusion_notes="Include test papers",
        exclusion_notes="Exclude nothing",
    )


def make_config(provider="local", model_id="qwen2.5-14b-instruct-q4_k_m"):
    return JudgeConfig(provider=provider, model_id=model_id)


class TestCreateJudgeFactory:
    """Test create_judge factory routing."""

    def test_local_provider_routes_to_llm_judge(self):
        config = make_config(provider="local", model_id="qwen2.5-14b")
        mock_judge = MagicMock()
        with patch(
            "contextual_arxiv_feed.judge.llm_judge.LLMJudge",
            return_value=mock_judge,
        ):
            judge = create_judge(config, [make_topic()])
            assert judge is mock_judge

    def test_cerebras_provider_routes_to_llm_judge(self):
        config = make_config(provider="cerebras", model_id="qwen-3-235b")
        mock_judge = MagicMock()
        with patch(
            "contextual_arxiv_feed.judge.llm_judge.LLMJudge",
            return_value=mock_judge,
        ):
            judge = create_judge(config, [make_topic()])
            assert judge is mock_judge


class TestLLMJudgeIntCasting:
    """Test that LLM judge casts floats to ints."""

    def test_cast_numerics_to_int(self):
        from contextual_arxiv_feed.judge.llm_judge import LLMJudge

        judge = LLMJudge.__new__(LLMJudge)

        data = {
            "prompt_version": 3.0,
            "quality_i": 72.0,
            "confidence_i": 90.0,
            "quality_breakdown_i": {
                "novelty_i": 80.0,
                "relevance_i": 75.5,
                "technical_depth_i": 70.0,
            },
        }

        result = judge._cast_numerics_to_int(data)

        assert result["prompt_version"] == 3
        assert isinstance(result["prompt_version"], int)
        assert result["quality_i"] == 72
        assert result["confidence_i"] == 90
        assert result["quality_breakdown_i"]["novelty_i"] == 80
        assert result["quality_breakdown_i"]["relevance_i"] == 76  # rounded from 75.5
        assert result["quality_breakdown_i"]["technical_depth_i"] == 70

    def test_cast_already_int(self):
        from contextual_arxiv_feed.judge.llm_judge import LLMJudge

        judge = LLMJudge.__new__(LLMJudge)

        data = {
            "prompt_version": 3,
            "quality_i": 72,
            "confidence_i": 90,
            "quality_breakdown_i": {
                "novelty_i": 80,
                "relevance_i": 75,
                "technical_depth_i": 70,
            },
        }

        result = judge._cast_numerics_to_int(data)
        assert result["prompt_version"] == 3
        assert result["quality_i"] == 72
