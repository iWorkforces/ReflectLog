'''Numba-accelerated numerical utilities for performance-critical operations.

This module provides JIT-compiled functions using numba for numerical computations
that benefit from machine code compilation. These are used in fusion scoring,
normalization, and other vectorized operations.

Usage:
    from reflectlog.application.utils.numba_utils import (
        normalize_scores_minmax,
        filter_scores_by_threshold,
        compute_rrf_scores,
    )

Note:
    - Functions are compiled on first call (JIT) with caching enabled
    - Subsequent calls use cached machine code for near-native performance
    - All functions operate on numpy arrays for maximum efficiency
'''

from numba import jit, prange
import numpy as np
from numpy.typing import NDArray


@jit(nopython=True, cache=True, fastmath=True)
def _find_minmax(scores: NDArray[np.float64]) -> tuple[float, float]:
    '''Find min and max values in a single pass (numba-accelerated).

    Args:
        scores: 1D numpy array of float64 scores.

    Returns:
        Tuple of (min_value, max_value).
    '''
    if len(scores) == 0:
        return 0.0, 0.0

    min_val = scores[0]
    max_val = scores[0]

    for i in range(1, len(scores)):
        if scores[i] < min_val:
            min_val = scores[i]
        elif scores[i] > max_val:
            max_val = scores[i]

    return min_val, max_val


@jit(nopython=True, cache=True, fastmath=True)
def _normalize_inplace(
    scores: NDArray[np.float64],
    min_val: float,
    max_val: float,
) -> None:
    '''Normalize scores in-place using min-max normalization (numba-accelerated).

    Args:
        scores: 1D numpy array to normalize in-place.
        min_val: Minimum value for normalization.
        max_val: Maximum value for normalization.
    '''
    if max_val == min_val:
        # All scores are the same - set to 0.5 (neutral) since we have no
        # differentiation information. Using 0.5 rather than 1.0 (optimistic)
        # or 0.0 (pessimistic) provides a balanced neutral baseline.
        for i in range(len(scores)):
            scores[i] = 0.5
        return

    range_inv = 1.0 / (max_val - min_val)
    for i in range(len(scores)):
        scores[i] = (scores[i] - min_val) * range_inv


def normalize_scores_minmax(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    '''Normalize scores to 0-1 range using min-max normalization.

    This is a high-level wrapper that handles edge cases and calls
    the numba-accelerated implementation.

    Args:
        scores: 1D numpy array of scores to normalize.

    Returns:
        New numpy array with normalized scores (0-1 range).

    Example:
        >>> scores = np.array([0.1, 0.5, 0.9])
        >>> normalized = normalize_scores_minmax(scores)
        >>> print(normalized)  # [0.0, 0.5, 1.0]
    '''
    if len(scores) == 0:
        return scores.copy()

    # Create a copy to avoid modifying the original
    result = scores.astype(np.float64).copy()

    # Find min/max in single pass
    min_val, max_val = _find_minmax(result)

    # Normalize in-place
    _normalize_inplace(result, min_val, max_val)

    return result


@jit(nopython=True, cache=True, fastmath=True)
def _filter_by_threshold(
    scores: NDArray[np.float64],
    threshold: float,
) -> NDArray[np.bool_]:
    '''Create boolean mask for scores above threshold (numba-accelerated).

    Args:
        scores: 1D numpy array of scores.
        threshold: Minimum score threshold.

    Returns:
        Boolean mask array where True indicates score >= threshold.
    '''
    mask = np.empty(len(scores), dtype=np.bool_)
    for i in range(len(scores)):
        mask[i] = scores[i] >= threshold
    return mask


def filter_scores_by_threshold(
    scores: NDArray[np.float64],
    threshold: float,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    '''Filter scores by threshold and return indices and filtered scores.

    Args:
        scores: 1D numpy array of scores.
        threshold: Minimum score threshold.

    Returns:
        Tuple of (indices, filtered_scores) where indices are the original
        positions of scores that passed the threshold.

    Example:
        >>> scores = np.array([0.3, 0.8, 0.5, 0.9])
        >>> indices, filtered = filter_scores_by_threshold(scores, 0.6)
        >>> print(indices)   # [1, 3]
        >>> print(filtered)  # [0.8, 0.9]
    '''
    mask = _filter_by_threshold(scores, threshold)
    indices = np.where(mask)[0]
    return indices, scores[mask]


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def compute_rrf_scores_batch(
    ranks_matrix: NDArray[np.int64],
    k: int = 60,
) -> NDArray[np.float64]:
    """Compute RRF scores for multiple documents across multiple rankings.

    This is a parallel implementation of Reciprocal Rank Fusion:
    RRF_score(d) = sum over rankings: 1 / (k + rank(d))

    Rank Encoding:
        - 1-indexed ranks (1 = first place, 2 = second place, etc.)
        - 0 explicitly means "document not present in this ranking"
        - Documents absent from a ranking simply don't contribute to the score

    Args:
        ranks_matrix: 2D array of shape (n_docs, n_rankings) where each value
                     is the rank of that document in that ranking (1-indexed).
                     Use 0 to indicate document not present in ranking.
        k: RRF constant (default: 60). Lower values give more weight to top ranks.

    Returns:
        1D array of RRF scores for each document. Documents not present in any
        ranking will have a score of 0.0.

    Example:
        >>> # Doc 0: rank 1 in ranking 0, rank 2 in ranking 1
        >>> # Doc 1: rank 2 in ranking 0, not in ranking 1 (encoded as 0)
        >>> ranks = np.array([[1, 2], [2, 0]], dtype=np.int64)
        >>> scores = compute_rrf_scores_batch(ranks, k=60)
        >>> # scores[0] = 1/(60+1) + 1/(60+2) ≈ 0.032
        >>> # scores[1] = 1/(60+2) + 0 ≈ 0.016
    """
    n_docs = ranks_matrix.shape[0]
    n_rankings = ranks_matrix.shape[1]
    scores = np.zeros(n_docs, dtype=np.float64)

    for i in prange(n_docs):
        total = 0.0
        for j in range(n_rankings):
            rank = ranks_matrix[i, j]
            if (
                rank > 0
            ):  # Only count if document is present in this ranking (0 = absent)
                total += 1.0 / (k + rank)
        scores[i] = total

    return scores


@jit(nopython=True, cache=True, fastmath=True)
def distance_to_similarity_cosine(
    distances: NDArray[np.float32],
) -> NDArray[np.float32]:
    '''Convert cosine distances to similarity scores (numba-accelerated).

    Cosine similarity = 1 - cosine distance

    Args:
        distances: 1D numpy array of cosine distances.

    Returns:
        1D numpy array of similarity scores.

    Example:
        >>> distances = np.array([0.1, 0.3, 0.5], dtype=np.float32)
        >>> similarities = distance_to_similarity_cosine(distances)
        >>> print(similarities)  # [0.9, 0.7, 0.5]
    '''
    result = np.empty_like(distances)
    for i in range(len(distances)):
        result[i] = 1.0 - distances[i]
    return result


# Warm-up function to pre-compile all JIT functions
def warmup_numba_functions() -> bool:
    """Pre-compile all numba JIT functions to avoid first-call latency.

    Call this during application startup to ensure all numba functions
    are compiled before they're needed in production code paths.

    Returns:
        True if all functions compiled successfully, False otherwise.

    Raises:
        ImportError: If numpy or numba is not available.

    Example:
        # In server startup
        from reflectlog.application.utils.numba_utils import warmup_numba_functions
        success = warmup_numba_functions()
        if not success:
            logger.warning("Numba JIT warmup failed, first calls may be slower")
    """
    try:
        # Small test arrays for compilation
        test_scores = np.array([0.1, 0.5, 0.9], dtype=np.float64)
        test_distances = np.array([0.1, 0.3, 0.5], dtype=np.float32)
        test_ranks = np.array([[1, 2], [2, 0]], dtype=np.int64)

        # Trigger compilation of each function
        _ = normalize_scores_minmax(test_scores)
        _ = filter_scores_by_threshold(test_scores, 0.5)
        _ = compute_rrf_scores_batch(test_ranks, k=60)
        _ = distance_to_similarity_cosine(test_distances)

        return True
    except Exception as e:
        # Log but don't raise - allow application to continue with slower first calls
        import warnings

        warnings.warn(
            f'Numba JIT warmup failed: {e}. First calls will be slower.',
            RuntimeWarning,
            stacklevel=2,
        )
        return False
