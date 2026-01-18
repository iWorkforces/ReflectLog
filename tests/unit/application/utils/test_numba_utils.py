"""Unit tests for reflectlog.application.utils.numba_utils module."""

import numpy as np

from reflectlog.application.utils.numba_utils import (
    compute_rrf_scores_batch,
    distance_to_similarity_cosine,
    filter_scores_by_threshold,
    normalize_scores_minmax,
    warmup_numba_functions,
)


class TestNormalizeScoresMinmax:
    """Tests for normalize_scores_minmax function."""

    def test_basic_normalization(self) -> None:
        """Test basic min-max normalization."""
        scores = np.array([0.1, 0.5, 0.9], dtype=np.float64)
        result = normalize_scores_minmax(scores)
        np.testing.assert_array_almost_equal(result, [0.0, 0.5, 1.0])

    def test_empty_array(self) -> None:
        """Test handling of empty array."""
        scores = np.array([], dtype=np.float64)
        result = normalize_scores_minmax(scores)
        assert len(result) == 0

    def test_single_element(self) -> None:
        """Test handling of single-element array."""
        scores = np.array([0.5], dtype=np.float64)
        result = normalize_scores_minmax(scores)
        # Single element should be normalized to 1.0 (all same = max)
        np.testing.assert_array_almost_equal(result, [1.0])

    def test_all_same_values(self) -> None:
        """Test handling of array with all identical values."""
        scores = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        result = normalize_scores_minmax(scores)
        # All same values should normalize to 1.0
        np.testing.assert_array_almost_equal(result, [1.0, 1.0, 1.0])

    def test_negative_values(self) -> None:
        """Test normalization with negative values."""
        scores = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        result = normalize_scores_minmax(scores)
        np.testing.assert_array_almost_equal(result, [0.0, 0.5, 1.0])

    def test_does_not_modify_original(self) -> None:
        """Test that original array is not modified."""
        original = np.array([0.1, 0.5, 0.9], dtype=np.float64)
        original_copy = original.copy()
        _ = normalize_scores_minmax(original)
        np.testing.assert_array_equal(original, original_copy)

    def test_large_array(self) -> None:
        """Test normalization of large array."""
        scores = np.random.random(10000).astype(np.float64)
        result = normalize_scores_minmax(scores)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_zero_range(self) -> None:
        """Test with min == max (zero range)."""
        scores = np.array([42.0, 42.0, 42.0], dtype=np.float64)
        result = normalize_scores_minmax(scores)
        np.testing.assert_array_almost_equal(result, [1.0, 1.0, 1.0])

    def test_very_small_range(self) -> None:
        """Test with very small value range."""
        scores = np.array([1.0, 1.0 + 1e-10, 1.0 + 2e-10], dtype=np.float64)
        result = normalize_scores_minmax(scores)
        # Should still normalize correctly
        assert result[0] < result[1] < result[2]

    def test_already_normalized(self) -> None:
        """Test array that's already in 0-1 range."""
        scores = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        result = normalize_scores_minmax(scores)
        np.testing.assert_array_almost_equal(result, [0.0, 0.5, 1.0])


class TestFilterScoresByThreshold:
    """Tests for filter_scores_by_threshold function."""

    def test_basic_filtering(self) -> None:
        """Test basic threshold filtering."""
        scores = np.array([0.3, 0.8, 0.5, 0.9], dtype=np.float64)
        indices, filtered = filter_scores_by_threshold(scores, 0.6)
        np.testing.assert_array_equal(indices, [1, 3])
        np.testing.assert_array_almost_equal(filtered, [0.8, 0.9])

    def test_empty_array(self) -> None:
        """Test filtering of empty array."""
        scores = np.array([], dtype=np.float64)
        indices, filtered = filter_scores_by_threshold(scores, 0.5)
        assert len(indices) == 0
        assert len(filtered) == 0

    def test_all_pass_threshold(self) -> None:
        """Test when all scores pass threshold."""
        scores = np.array([0.6, 0.7, 0.8], dtype=np.float64)
        indices, filtered = filter_scores_by_threshold(scores, 0.5)
        np.testing.assert_array_equal(indices, [0, 1, 2])
        np.testing.assert_array_almost_equal(filtered, [0.6, 0.7, 0.8])

    def test_all_fail_threshold(self) -> None:
        """Test when all scores fail threshold."""
        scores = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        indices, filtered = filter_scores_by_threshold(scores, 0.5)
        assert len(indices) == 0
        assert len(filtered) == 0

    def test_exact_threshold(self) -> None:
        """Test scores exactly at threshold."""
        scores = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        indices, filtered = filter_scores_by_threshold(scores, 0.5)
        # >= threshold, so should pass
        np.testing.assert_array_equal(indices, [0, 1, 2])

    def test_zero_threshold(self) -> None:
        """Test with zero threshold."""
        scores = np.array([0.0, 0.1, 0.5], dtype=np.float64)
        indices, filtered = filter_scores_by_threshold(scores, 0.0)
        np.testing.assert_array_equal(indices, [0, 1, 2])

    def test_one_threshold(self) -> None:
        """Test with threshold of 1.0."""
        scores = np.array([0.9, 1.0, 0.99], dtype=np.float64)
        indices, filtered = filter_scores_by_threshold(scores, 1.0)
        np.testing.assert_array_equal(indices, [1])
        np.testing.assert_array_almost_equal(filtered, [1.0])

    def test_single_element_pass(self) -> None:
        """Test single element that passes."""
        scores = np.array([0.8], dtype=np.float64)
        indices, filtered = filter_scores_by_threshold(scores, 0.5)
        np.testing.assert_array_equal(indices, [0])
        np.testing.assert_array_almost_equal(filtered, [0.8])

    def test_single_element_fail(self) -> None:
        """Test single element that fails."""
        scores = np.array([0.3], dtype=np.float64)
        indices, filtered = filter_scores_by_threshold(scores, 0.5)
        assert len(indices) == 0
        assert len(filtered) == 0


class TestComputeRrfScoresBatch:
    """Tests for compute_rrf_scores_batch function."""

    def test_basic_rrf_computation(self) -> None:
        """Test basic RRF score computation."""
        # Doc 0: rank 1 in ranking 0, rank 2 in ranking 1
        # Doc 1: rank 2 in ranking 0, not in ranking 1 (0)
        ranks = np.array([[1, 2], [2, 0]], dtype=np.int64)
        scores = compute_rrf_scores_batch(ranks, k=60)

        # Doc 0: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
        # Doc 1: 1/(60+2) + 0 = 0.01613
        expected_0 = 1 / 61 + 1 / 62
        expected_1 = 1 / 62

        np.testing.assert_almost_equal(scores[0], expected_0, decimal=5)
        np.testing.assert_almost_equal(scores[1], expected_1, decimal=5)

    def test_all_zeros_ranks(self) -> None:
        """Test with all zeros (no document in any ranking)."""
        ranks = np.array([[0, 0], [0, 0]], dtype=np.int64)
        scores = compute_rrf_scores_batch(ranks, k=60)
        np.testing.assert_array_almost_equal(scores, [0.0, 0.0])

    def test_single_document(self) -> None:
        """Test with single document."""
        ranks = np.array([[1, 1]], dtype=np.int64)
        scores = compute_rrf_scores_batch(ranks, k=60)
        expected = 2 / 61  # 1/(60+1) + 1/(60+1)
        np.testing.assert_almost_equal(scores[0], expected, decimal=5)

    def test_single_ranking(self) -> None:
        """Test with single ranking (one column)."""
        ranks = np.array([[1], [2], [3]], dtype=np.int64)
        scores = compute_rrf_scores_batch(ranks, k=60)
        expected = [1 / 61, 1 / 62, 1 / 63]
        np.testing.assert_array_almost_equal(scores, expected, decimal=5)

    def test_custom_k_value(self) -> None:
        """Test with custom k value."""
        ranks = np.array([[1, 1]], dtype=np.int64)
        scores_k30 = compute_rrf_scores_batch(ranks, k=30)
        scores_k60 = compute_rrf_scores_batch(ranks, k=60)
        # Lower k gives more weight to top ranks
        assert scores_k30[0] > scores_k60[0]

    def test_document_not_in_ranking(self) -> None:
        """Test documents not present in some rankings (rank=0)."""
        # Doc 0: in both rankings at rank 1
        # Doc 1: only in ranking 0 at rank 2
        ranks = np.array([[1, 1], [2, 0]], dtype=np.int64)
        scores = compute_rrf_scores_batch(ranks, k=60)

        # Doc 0 should have higher score (in both rankings)
        assert scores[0] > scores[1]

    def test_high_ranks(self) -> None:
        """Test with high rank values."""
        ranks = np.array([[100, 200]], dtype=np.int64)
        scores = compute_rrf_scores_batch(ranks, k=60)
        expected = 1 / 160 + 1 / 260
        np.testing.assert_almost_equal(scores[0], expected, decimal=5)

    def test_large_matrix(self) -> None:
        """Test with larger matrix."""
        # 100 docs, 5 rankings
        n_docs, n_rankings = 100, 5
        ranks = np.random.randint(0, 50, size=(n_docs, n_rankings), dtype=np.int64)
        scores = compute_rrf_scores_batch(ranks, k=60)

        assert len(scores) == n_docs
        assert all(s >= 0 for s in scores)


class TestDistanceToSimilarityCosine:
    """Tests for distance_to_similarity_cosine function."""

    def test_basic_conversion(self) -> None:
        """Test basic distance to similarity conversion."""
        distances = np.array([0.1, 0.3, 0.5], dtype=np.float32)
        similarities = distance_to_similarity_cosine(distances)
        np.testing.assert_array_almost_equal(similarities, [0.9, 0.7, 0.5])

    def test_zero_distance(self) -> None:
        """Test zero distance (perfect match)."""
        distances = np.array([0.0], dtype=np.float32)
        similarities = distance_to_similarity_cosine(distances)
        np.testing.assert_array_almost_equal(similarities, [1.0])

    def test_max_distance(self) -> None:
        """Test maximum distance (opposite vectors)."""
        distances = np.array([1.0], dtype=np.float32)
        similarities = distance_to_similarity_cosine(distances)
        np.testing.assert_array_almost_equal(similarities, [0.0])

    def test_empty_array(self) -> None:
        """Test empty array handling."""
        distances = np.array([], dtype=np.float32)
        similarities = distance_to_similarity_cosine(distances)
        assert len(similarities) == 0

    def test_single_element(self) -> None:
        """Test single element array."""
        distances = np.array([0.25], dtype=np.float32)
        similarities = distance_to_similarity_cosine(distances)
        np.testing.assert_array_almost_equal(similarities, [0.75])

    def test_preserves_order(self) -> None:
        """Test that order is preserved (inverted for similarity)."""
        distances = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        similarities = distance_to_similarity_cosine(distances)
        # Lower distance = higher similarity
        assert similarities[0] > similarities[1] > similarities[2]

    def test_boundary_values(self) -> None:
        """Test boundary values."""
        distances = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        similarities = distance_to_similarity_cosine(distances)
        np.testing.assert_array_almost_equal(similarities, [1.0, 0.5, 0.0])


class TestWarmupNumbaFunctions:
    """Tests for warmup_numba_functions function."""

    def test_warmup_completes_without_error(self) -> None:
        """Test that warmup runs without raising exceptions."""
        # Should not raise any exceptions
        warmup_numba_functions()

    def test_warmup_is_idempotent(self) -> None:
        """Test that warmup can be called multiple times."""
        # Should not raise even when called multiple times
        warmup_numba_functions()
        warmup_numba_functions()
        warmup_numba_functions()

    def test_functions_work_after_warmup(self) -> None:
        """Test that all functions work correctly after warmup."""
        warmup_numba_functions()

        # Test each function works
        scores = np.array([0.1, 0.5, 0.9], dtype=np.float64)
        assert len(normalize_scores_minmax(scores)) == 3

        indices, filtered = filter_scores_by_threshold(scores, 0.5)
        assert len(indices) > 0

        ranks = np.array([[1, 2]], dtype=np.int64)
        rrf_scores = compute_rrf_scores_batch(ranks, k=60)
        assert len(rrf_scores) == 1

        distances = np.array([0.1], dtype=np.float32)
        similarities = distance_to_similarity_cosine(distances)
        assert len(similarities) == 1
