"""Unit tests for reranker score normalization utilities.

Tests cover:
- normalize_reranker_scores: Batch min-max normalization to [0, 1]
- apply_threshold_with_safety_net: Threshold filtering with optional safety net
"""

import pytest

from openmemories.application.memory.reranking import (
    apply_threshold_with_safety_net,
    normalize_reranker_scores,
)


class TestNormalizeRerankerScores:
    """Tests for normalize_reranker_scores function."""

    def test_empty_input(self) -> None:
        """Empty input should return empty list."""
        result = normalize_reranker_scores([])
        assert result == []

    def test_single_result(self) -> None:
        """Single result should get score 1.0 (best by definition)."""
        result = normalize_reranker_scores([("doc1", 0.17)])
        assert len(result) == 1
        assert result[0][0] == "doc1"
        assert result[0][1] == 1.0

    def test_two_results(self) -> None:
        """With two results, best=1.0, worst=0.0."""
        result = normalize_reranker_scores([("best", 0.9), ("worst", 0.1)])
        assert len(result) == 2

        # Find results by document
        best_result = next(r for r in result if r[0] == "best")
        worst_result = next(r for r in result if r[0] == "worst")

        assert best_result[1] == 1.0
        assert worst_result[1] == 0.0

    def test_cross_encoder_range(self) -> None:
        """CrossEncoder typical range (0.001-0.17) should normalize to 0-1."""
        scored = [
            ("doc1", 0.17),  # best
            ("doc2", 0.05),  # middle
            ("doc3", 0.001),  # worst
        ]
        result = normalize_reranker_scores(scored)

        assert len(result) == 3

        # Best should be 1.0, worst should be 0.0
        doc1_result = next(r for r in result if r[0] == "doc1")
        doc3_result = next(r for r in result if r[0] == "doc3")

        assert doc1_result[1] == 1.0
        assert doc3_result[1] == 0.0

        # Middle should be in between
        doc2_result = next(r for r in result if r[0] == "doc2")
        assert 0.0 < doc2_result[1] < 1.0

    def test_llm_range(self) -> None:
        """LLM typical range (0.7-0.9) should normalize to 0-1."""
        scored = [
            ("doc1", 0.90),  # best
            ("doc2", 0.85),  # middle
            ("doc3", 0.70),  # worst
        ]
        result = normalize_reranker_scores(scored)

        assert len(result) == 3

        # Best should be 1.0, worst should be 0.0
        doc1_result = next(r for r in result if r[0] == "doc1")
        doc3_result = next(r for r in result if r[0] == "doc3")

        assert doc1_result[1] == 1.0
        assert doc3_result[1] == 0.0

        # Middle should be 0.75 ((0.85 - 0.70) / (0.90 - 0.70))
        doc2_result = next(r for r in result if r[0] == "doc2")
        assert doc2_result[1] == pytest.approx(0.75)

    def test_all_equal_scores(self) -> None:
        """All equal scores should become 1.0 (all equally good)."""
        scored = [
            ("doc1", 0.5),
            ("doc2", 0.5),
            ("doc3", 0.5),
        ]
        result = normalize_reranker_scores(scored)

        assert len(result) == 3
        for doc, score in result:
            assert score == 1.0

    def test_preserves_document_order(self) -> None:
        """Document order should be preserved."""
        scored = [
            ("first", 0.3),
            ("second", 0.5),
            ("third", 0.4),
        ]
        result = normalize_reranker_scores(scored)

        assert result[0][0] == "first"
        assert result[1][0] == "second"
        assert result[2][0] == "third"

    def test_negative_scores(self) -> None:
        """Negative scores should be handled correctly."""
        scored = [
            ("doc1", 0.5),
            ("doc2", -0.5),
        ]
        result = normalize_reranker_scores(scored)

        doc1_result = next(r for r in result if r[0] == "doc1")
        doc2_result = next(r for r in result if r[0] == "doc2")

        assert doc1_result[1] == 1.0
        assert doc2_result[1] == 0.0


class TestApplyThresholdWithSafetyNet:
    """Tests for apply_threshold_with_safety_net function."""

    def test_empty_input(self) -> None:
        """Empty input should return empty list."""
        result = apply_threshold_with_safety_net([], 0.5)
        assert result == []

    def test_all_above_threshold(self) -> None:
        """All scores above threshold should be kept."""
        scored = [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.6)]
        result = apply_threshold_with_safety_net(scored, 0.5)
        assert len(result) == 3

    def test_some_below_threshold(self) -> None:
        """Scores below threshold should be filtered out."""
        scored = [("doc1", 0.9), ("doc2", 0.4), ("doc3", 0.3)]
        result = apply_threshold_with_safety_net(scored, 0.5)
        assert len(result) == 1
        assert result[0][0] == "doc1"

    def test_all_below_threshold_no_safety_net(self) -> None:
        """All below threshold with min_results=0 should return empty."""
        scored = [("doc1", 0.4), ("doc2", 0.3)]
        result = apply_threshold_with_safety_net(scored, 0.5, min_results=0)
        assert result == []

    def test_safety_net_when_enabled(self) -> None:
        """Safety net should return min_results when all below threshold."""
        # Results must be sorted by score descending for safety net to work
        scored = [("doc1", 0.4), ("doc2", 0.3), ("doc3", 0.2)]
        result = apply_threshold_with_safety_net(scored, 0.5, min_results=2)

        # Should return top 2 even though all below threshold
        assert len(result) == 2
        assert result[0][0] == "doc1"
        assert result[1][0] == "doc2"

    def test_safety_net_partial_filter(self) -> None:
        """Safety net should return min_results when filtered too much."""
        # 1 passes threshold, but min_results=2
        scored = [("doc1", 0.6), ("doc2", 0.4), ("doc3", 0.3)]
        result = apply_threshold_with_safety_net(scored, 0.5, min_results=2)

        # Should return top 2 (safety net kicks in)
        assert len(result) == 2
        assert result[0][0] == "doc1"
        assert result[1][0] == "doc2"

    def test_safety_net_not_enough_candidates(self) -> None:
        """Safety net should return all available if less than min_results."""
        scored = [("doc1", 0.4)]  # Only 1 candidate, min_results=2
        result = apply_threshold_with_safety_net(scored, 0.5, min_results=2)

        # Should return all available (1)
        assert len(result) == 1
        assert result[0][0] == "doc1"

    def test_threshold_zero(self) -> None:
        """Threshold of 0.0 should keep all results."""
        scored = [("doc1", 0.9), ("doc2", 0.1), ("doc3", 0.0)]
        result = apply_threshold_with_safety_net(scored, 0.0)
        assert len(result) == 3

    def test_threshold_one(self) -> None:
        """Threshold of 1.0 should only keep perfect scores."""
        scored = [("doc1", 1.0), ("doc2", 0.9)]
        result = apply_threshold_with_safety_net(scored, 1.0)
        assert len(result) == 1
        assert result[0][0] == "doc1"

    def test_threshold_at_boundary(self) -> None:
        """Score equal to threshold should be kept (inclusive)."""
        scored = [("doc1", 0.5), ("doc2", 0.4)]
        result = apply_threshold_with_safety_net(scored, 0.5)
        assert len(result) == 1
        assert result[0][0] == "doc1"

    def test_min_results_zero_default(self) -> None:
        """Default min_results=0 allows empty results."""
        scored = [("doc1", 0.4), ("doc2", 0.3)]
        result = apply_threshold_with_safety_net(scored, 0.5)  # default min_results=0
        assert result == []


class TestIntegration:
    """Integration tests combining normalization and threshold filtering."""

    def test_cross_encoder_workflow(self) -> None:
        """Simulate CrossEncoder reranking workflow.

        CrossEncoder produces low scores (0.001-0.17), which after
        normalization become 0-1, making threshold 0.5 useful.
        """
        # Raw CrossEncoder scores (typical range)
        raw_scored = [
            ("doc1", 0.17),  # Best
            ("doc2", 0.05),  # Should be filtered
            ("doc3", 0.003),  # Should be filtered
        ]

        # Step 1: Normalize
        normalized = normalize_reranker_scores(raw_scored)

        # Step 2: Sort by score descending
        normalized.sort(key=lambda x: x[1], reverse=True)

        # Step 3: Apply threshold
        result = apply_threshold_with_safety_net(normalized, 0.5, min_results=0)

        # Only doc1 (score=1.0) should pass threshold 0.5
        assert len(result) == 1
        assert result[0][0] == "doc1"
        assert result[0][1] == 1.0

    def test_llm_reranker_workflow(self) -> None:
        """Simulate LLM reranking workflow.

        LLM produces calibrated scores (0.7-0.9), which after normalization
        might filter more aggressively.
        """
        # Raw LLM scores (typical range)
        raw_scored = [
            ("doc1", 0.90),  # Best -> 1.0
            ("doc2", 0.85),  # Middle -> 0.75
            ("doc3", 0.70),  # Worst -> 0.0
        ]

        # Step 1: Normalize
        normalized = normalize_reranker_scores(raw_scored)

        # Step 2: Sort by score descending
        normalized.sort(key=lambda x: x[1], reverse=True)

        # Step 3: Apply threshold 0.5
        result = apply_threshold_with_safety_net(normalized, 0.5, min_results=0)

        # doc1 (1.0) and doc2 (0.75) pass threshold 0.5
        assert len(result) == 2
        assert result[0][0] == "doc1"
        assert result[1][0] == "doc2"
