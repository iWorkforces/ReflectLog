#!/usr/bin/env python3
"""Unit tests for logging utilities."""

import os
import sys

import pytest

# Add the project root to path to avoid triggering package __init__.py
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

# Import directly from the module file to avoid triggering config singleton
from openmemories.application.utils.logging import (
    format_fusion_score_status,
)


@pytest.mark.unit
class TestFormatFusionScoreStatus:
    """Tests for format_fusion_score_status function."""

    def test_high_score_above_threshold_returns_keep(self):
        """Score above threshold should return KEEP status."""
        status, interpretation = format_fusion_score_status(0.8, 0.5)
        assert status == "KEEP"
        assert "High" in interpretation

    def test_low_score_below_threshold_returns_filter(self):
        """Score below threshold should return FILTER status."""
        status, interpretation = format_fusion_score_status(0.3, 0.5)
        assert status == "FILTER"
        assert "Minimal" in interpretation

    def test_score_at_threshold_returns_keep(self):
        """Score exactly at threshold should return KEEP."""
        status, _ = format_fusion_score_status(0.5, 0.5)
        assert status == "KEEP"

    def test_score_just_below_threshold_returns_filter(self):
        """Score just below threshold should return FILTER."""
        status, _ = format_fusion_score_status(0.49, 0.5)
        assert status == "FILTER"

    @pytest.mark.parametrize(
        "score,expected_keyword",
        [
            (0.9, "High"),
            (0.8, "High"),
            (0.7, "Moderate"),
            (0.6, "Moderate"),
            (0.5, "Low"),
            (0.4, "Low"),
            (0.3, "Minimal"),
            (0.0, "Minimal"),
        ],
    )
    def test_interpretation_ranges(self, score, expected_keyword):
        """Test interpretation contains expected keyword for score ranges."""
        _, interpretation = format_fusion_score_status(score, 0.5)
        assert expected_keyword in interpretation

    def test_zero_score_returns_filter(self):
        """Zero score should return FILTER with minimal interpretation."""
        status, interpretation = format_fusion_score_status(0.0, 0.5)
        assert status == "FILTER"
        assert "Minimal" in interpretation

    def test_perfect_score_returns_keep(self):
        """Perfect score (1.0) should return KEEP with high interpretation."""
        status, interpretation = format_fusion_score_status(1.0, 0.5)
        assert status == "KEEP"
        assert "High" in interpretation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
