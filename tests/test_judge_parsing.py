"""Tests for judge output parsing and validation.

Critical: All numeric values must be integers. NO FLOATS.
"""

import pytest

from contextual_arxiv_feed.judge.schema import (
    JudgeOutput,
    QualityBreakdown,
    parse_judge_output,
    truncate_rationale,
)


class TestQualityBreakdown:
    """Tests for QualityBreakdown validation."""

    def test_accepts_int_values(self):
        """Test that all int values are accepted."""
        breakdown = QualityBreakdown(
            novelty_i=80,
            relevance_i=75,
            technical_depth_i=70,
        )
        assert breakdown.novelty_i == 80
        assert breakdown.relevance_i == 75
        assert breakdown.technical_depth_i == 70

    def test_rejects_float_novelty(self):
        """Test that float novelty is rejected."""
        with pytest.raises(ValueError, match="Float values not allowed"):
            QualityBreakdown(
                novelty_i=80.5,  # Float!
                relevance_i=75,
                technical_depth_i=70,
            )

    def test_rejects_float_relevance(self):
        """Test that float relevance is rejected."""
        with pytest.raises(ValueError, match="Float values not allowed"):
            QualityBreakdown(
                novelty_i=80,
                relevance_i=75.5,  # Float!
                technical_depth_i=70,
            )

    def test_rejects_float_technical_depth(self):
        """Test that float technical_depth is rejected."""
        with pytest.raises(ValueError, match="Float values not allowed"):
            QualityBreakdown(
                novelty_i=80,
                relevance_i=75,
                technical_depth_i=70.1,  # Float!
            )

    def test_rejects_float_even_zero(self):
        """Test that float is rejected even when .0."""
        with pytest.raises(ValueError, match="Float values not allowed"):
            QualityBreakdown(
                novelty_i=80.0,  # Float! (even .0)
                relevance_i=75,
                technical_depth_i=70,
            )

    def test_weighted_average(self):
        """Test weighted average calculation (0.40, 0.35, 0.25)."""
        breakdown = QualityBreakdown(
            novelty_i=80,  # 0.40 weight
            relevance_i=80,  # 0.35 weight
            technical_depth_i=80,  # 0.25 weight
        )
        # All 80s should give 80
        assert breakdown.compute_weighted_average() == 80

    def test_weighted_average_different_values(self):
        """Test weighted average with different values."""
        breakdown = QualityBreakdown(
            novelty_i=100,  # 0.40 weight -> 40
            relevance_i=50,  # 0.35 weight -> 17.5
            technical_depth_i=50,  # 0.25 weight -> 12.5
        )
        # 40 + 17.5 + 12.5 = 70
        assert breakdown.compute_weighted_average() == 70


class TestJudgeOutput:
    """Tests for JudgeOutput validation."""

    def test_accepts_valid_output(self, sample_judge_output_dict: dict):
        """Test that valid output is accepted."""
        output = JudgeOutput(**sample_judge_output_dict)
        assert output.quality_verdict == "accept"
        assert output.quality_i == 78

    def test_rejects_float_quality_i(self, sample_judge_output_dict: dict):
        """Test that float quality_i is rejected."""
        sample_judge_output_dict["quality_i"] = 78.5
        with pytest.raises(ValueError, match="Float values not allowed"):
            JudgeOutput(**sample_judge_output_dict)

    def test_rejects_float_prompt_version(self, sample_judge_output_dict: dict):
        """Test that float prompt_version is rejected."""
        sample_judge_output_dict["prompt_version"] = 1.0
        with pytest.raises(ValueError, match="Float values not allowed"):
            JudgeOutput(**sample_judge_output_dict)

    def test_rejects_float_confidence(self, sample_judge_output_dict: dict):
        """Test that float confidence_i is rejected."""
        sample_judge_output_dict["confidence_i"] = 85.5
        with pytest.raises(ValueError, match="Float values not allowed"):
            JudgeOutput(**sample_judge_output_dict)

    def test_is_accepted_property(self, sample_judge_output_dict: dict):
        """Test is_accepted property."""
        output = JudgeOutput(**sample_judge_output_dict)
        assert output.is_accepted is True

        # Test rejection
        sample_judge_output_dict["quality_verdict"] = "reject"
        output = JudgeOutput(**sample_judge_output_dict)
        assert output.is_accepted is False

    def test_to_dict(self, sample_judge_output_dict: dict):
        """Test serialization to dict."""
        output = JudgeOutput(**sample_judge_output_dict)
        result = output.to_dict()

        assert result["quality_i"] == 78
        assert result["confidence_i"] == 85
        assert "quality_breakdown_i" in result


class TestParseJudgeOutput:
    """Tests for parse_judge_output function."""

    def test_parse_valid_output(self, sample_judge_output_dict: dict):
        """Test parsing valid output."""
        output = parse_judge_output(sample_judge_output_dict)
        assert output.quality_verdict == "accept"

    def test_parse_rejects_float_values(self, sample_judge_output_dict: dict):
        """Test that parsing rejects float values."""
        sample_judge_output_dict["quality_i"] = 78.5
        with pytest.raises(ValueError, match="Float values not allowed"):
            parse_judge_output(sample_judge_output_dict)


class TestTruncateRationale:
    """Tests for rationale truncation."""

    def test_no_truncation_needed(self):
        """Test that short rationale is not truncated."""
        rationale = "Short rationale."
        result = truncate_rationale(rationale, max_length=300)
        assert result == rationale

    def test_truncation_at_word_boundary(self):
        """Test truncation at word boundary."""
        rationale = "This is a " + "very " * 100 + "long rationale."
        result = truncate_rationale(rationale, max_length=50)
        assert len(result) <= 53  # 50 + "..."
        assert result.endswith("...")
        assert " " not in result[-4:-3]  # Should end at word boundary

    def test_exact_length(self):
        """Test rationale at exact max length."""
        rationale = "x" * 300
        result = truncate_rationale(rationale, max_length=300)
        assert result == rationale
