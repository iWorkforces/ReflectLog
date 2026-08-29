"""Scoring, normalization, and recency decay utilities.

This module provides scoring functions used across infrastructure and
application layers. Placed in utility/ (order=0) so infrastructure/
can import without violating the dependency hierarchy.

Functions from numba_utils (JIT-compiled):
    - normalize_scores_minmax: Min-max normalization to [0, 1]
    - distance_to_similarity_cosine: Cosine distance → similarity
    - filter_scores_by_threshold: Threshold filter with indices
    - compute_rrf_scores_batch: Parallel RRF scoring
    - warmup_numba_functions: Pre-compile all JIT functions

Functions from reranking normalization:
    - normalize_reranker_scores: Batch min-max for reranker outputs
    - apply_threshold_with_safety_net: Threshold with min_results guarantee
    - calculate_recency_factor: Exponential recency decay factor
    - apply_recency_decay: Apply decay to scored results

Usage:
    from reflectlog.utility.scoring import (
        normalize_scores_minmax,
        distance_to_similarity_cosine,
        normalize_reranker_scores,
        apply_recency_decay,
    )
"""

from datetime import UTC, datetime
import math

from numba import jit, prange
import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Numba JIT-compiled helpers (from application/utils/numba_utils.py)
# ---------------------------------------------------------------------------


@jit(nopython=True, cache=True, fastmath=True)
def _find_minmax(scores: NDArray[np.float64]) -> tuple[float, float]:
    """Find min and max values in a single pass (numba-accelerated).

    Args:
        scores: 1D numpy array of float64 scores.

    Returns:
        Tuple of (min_value, max_value).
    """
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
    """Normalize scores in-place using min-max normalization (numba-accelerated).

    Args:
        scores: 1D numpy array to normalize in-place.
        min_val: Minimum value for normalization.
        max_val: Maximum value for normalization.
    """
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
    """Normalize scores to 0-1 range using min-max normalization.

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
    """
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
def distance_to_similarity_cosine(
    distances: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Convert cosine distances to similarity scores (numba-accelerated).

    Cosine similarity = 1 - cosine distance

    Args:
        distances: 1D numpy array of cosine distances.

    Returns:
        1D numpy array of similarity scores.

    Example:
        >>> distances = np.array([0.1, 0.3, 0.5], dtype=np.float32)
        >>> similarities = distance_to_similarity_cosine(distances)
        >>> print(similarities)  # [0.9, 0.7, 0.5]
    """
    result = np.empty_like(distances)
    for i in range(len(distances)):
        result[i] = 1.0 - distances[i]
    return result


@jit(nopython=True, cache=True, fastmath=True)
def _filter_by_threshold(
    scores: NDArray[np.float64],
    threshold: float,
) -> NDArray[np.bool_]:
    """Create boolean mask for scores above threshold (numba-accelerated).

    Args:
        scores: 1D numpy array of scores.
        threshold: Minimum score threshold.

    Returns:
        Boolean mask array where True indicates score >= threshold.
    """
    mask = np.empty(len(scores), dtype=np.bool_)
    for i in range(len(scores)):
        mask[i] = scores[i] >= threshold
    return mask


def filter_scores_by_threshold(
    scores: NDArray[np.float64],
    threshold: float,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Filter scores by threshold and return indices and filtered scores.

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
    """
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


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def compute_weighted_rrf_scores_batch(
    ranks_matrix: NDArray[np.int64],
    weights: NDArray[np.float64],
    k: int = 60,
) -> NDArray[np.float64]:
    """Weighted RRF: score(d) = sum_j w_j / (k + rank_j(d))."""
    n_docs = ranks_matrix.shape[0]
    n_rankings = ranks_matrix.shape[1]
    scores = np.zeros(n_docs, dtype=np.float64)

    for i in prange(n_docs):
        total = 0.0
        for j in range(n_rankings):
            rank = ranks_matrix[i, j]
            if rank > 0:
                total += weights[j] / (k + rank)
        scores[i] = total

    return scores


def warmup_numba_functions() -> bool:
    """Pre-compile all numba JIT functions to avoid first-call latency.

    Call this during application startup to ensure all numba functions
    are compiled before they're needed in production code paths.

    Returns:
        True if all functions compiled successfully, False otherwise.

    Raises:
        ImportError: If numpy or numba is not available.

    Example:
        >>> from reflectlog.utility.scoring import warmup_numba_functions
        >>> success = warmup_numba_functions()
        >>> if not success:
        ...     print("Numba JIT warmup failed")
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
        _ = compute_weighted_rrf_scores_batch(
            test_ranks, np.array([1.0, 1.0], dtype=np.float64), k=60
        )
        _ = distance_to_similarity_cosine(test_distances)

        return True
    except Exception as e:
        import warnings

        warnings.warn(
            f"Numba JIT warmup failed: {e}. First calls will be slower.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False


# ---------------------------------------------------------------------------
# Reranker normalization (from application/memory/reranking/normalization.py)
# ---------------------------------------------------------------------------


def normalize_reranker_scores(
    scored_results: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Normalize reranker scores to [0, 1] using batch min-max normalization.

    This function transforms scores so that:
    - Best score in batch = 1.0
    - Worst score in batch = 0.0
    - Single result = 1.0 (best by definition)
    - All equal scores = 1.0 (all equally good)

    Args:
        scored_results: List of (document, raw_score) tuples from a reranker.

    Returns:
        List of (document, normalized_score) tuples with scores in [0, 1].

    Example:
        >>> # CrossEncoder typical range (0.001-0.17)
        >>> scored = [("doc1", 0.17), ("doc2", 0.05), ("doc3", 0.001)]
        >>> normalized = normalize_reranker_scores(scored)
        >>> # normalized[0] = ("doc1", 1.0)  # best score
        >>> # normalized[2] = ("doc3", 0.0)  # worst score
    """
    if not scored_results:
        return []

    if len(scored_results) == 1:
        # Single result is best by definition
        return [(scored_results[0][0], 1.0)]

    documents = [doc for doc, _ in scored_results]
    scores: NDArray[np.float64] = np.array(
        [score for _, score in scored_results], dtype=np.float64
    )
    if float(np.max(scores)) == float(np.min(scores)):
        return [(doc, 1.0) for doc in documents]

    normalized = normalize_scores_minmax(scores)

    return [
        (doc, float(norm_score))
        for doc, norm_score in zip(documents, normalized, strict=True)
    ]


def apply_threshold_with_safety_net(
    scored_results: list[tuple[str, float]],
    threshold: float,
    min_results: int = 0,
) -> list[tuple[str, float]]:
    """Apply threshold filtering with optional min_results safety net.

    This function filters results by score threshold, with an optional safety
    net that guarantees at least min_results are returned (if candidates exist).

    Args:
        scored_results: List of (document, score) tuples, MUST be sorted by
            score descending for safety net to work correctly.
        threshold: Minimum score threshold (inclusive). Results with
            score >= threshold are kept.
        min_results: Minimum results to return regardless of threshold.
            Set to 0 (default) to disable safety net and allow empty results.

    Returns:
        Filtered results. If min_results > 0 and all scores are below threshold,
        returns the top min_results instead of empty list.

    Example:
        >>> # With safety net disabled (default)
        >>> scored = [("doc1", 0.4), ("doc2", 0.3)]  # all below 0.5
        >>> filtered = apply_threshold_with_safety_net(scored, 0.5, min_results=0)
        >>> # filtered = []  (all filtered out)

        >>> # With safety net enabled
        >>> filtered = apply_threshold_with_safety_net(scored, 0.5, min_results=1)
        >>> # filtered = [("doc1", 0.4)]  (top 1 returned despite being below threshold)
    """
    if not scored_results:
        return []

    # Filter by threshold
    filtered = [(doc, score) for doc, score in scored_results if score >= threshold]

    # Safety net: return top min_results if we filtered too much
    if min_results > 0:
        if len(filtered) < min_results and len(scored_results) >= min_results:
            # Return top min_results (assumes input is sorted by score desc)
            return scored_results[:min_results]
        elif len(filtered) < min_results:
            # Not enough candidates to meet min_results
            return scored_results  # Return all available

    return filtered


def calculate_recency_factor(
    timestamp_iso: str,
    decay_rate: float,
    now: datetime | None = None,
) -> float:
    """Calculate recency decay factor from an ISO timestamp.

    Uses exponential decay formula: recency_factor = exp(-decay_rate * hours_old)

    Args:
        timestamp_iso: ISO 8601 timestamp string (e.g., "2024-01-15T10:30:00+00:00").
        decay_rate: Decay rate per hour. Default 0.01 gives ~50% decay at 69 hours.
        now: Current time for testing. If None, uses datetime.now(timezone.utc).

    Returns:
        Recency factor in range (0, 1]. Newer memories get factor ≈ 1.0,
        older memories get progressively lower factors.

    Example:
        >>> # Memory created 2 hours ago with default decay rate
        >>> factor = calculate_recency_factor("2024-01-15T08:00:00+00:00", 0.01)
        >>> # factor ≈ 0.98 (exp(-0.01 * 2))

        >>> # Memory created 69 hours ago (half-life point)
        >>> factor = calculate_recency_factor("2024-01-12T17:00:00+00:00", 0.01)
        >>> # factor ≈ 0.50 (exp(-0.01 * 69))
    """
    if now is None:
        now = datetime.now(UTC)

    # Parse ISO timestamp
    try:
        created_at = datetime.fromisoformat(timestamp_iso)
        # Ensure timezone-aware
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
    except ValueError, TypeError:
        # If timestamp is invalid, return factor of 1.0 (no decay)
        return 1.0

    # Calculate hours old
    delta = now - created_at
    hours_old = max(0.0, delta.total_seconds() / 3600)

    # Calculate decay factor: exp(-decay_rate * hours_old)
    recency_factor = math.exp(-decay_rate * hours_old)

    return recency_factor


def apply_recency_decay(
    scored_results: list[tuple[str, float]],
    timestamp_map: dict[str, str],
    decay_rate: float,
    now: datetime | None = None,
) -> list[tuple[str, float]]:
    """Apply recency decay to scored results.

    Multiplies each score by its recency factor based on the document's age.
    Documents without timestamps in the map retain their original scores.

    Args:
        scored_results: List of (document, score) tuples from reranker.
        timestamp_map: Dict mapping document text to ISO timestamp strings.
        decay_rate: Decay rate per hour (e.g., 0.01 for ~50% decay at 69 hours).
        now: Current time for testing. If None, uses datetime.now(timezone.utc).

    Returns:
        List of (document, decayed_score) tuples, sorted by decayed score descending.

    Example:
        >>> scored = [("old_doc", 0.9), ("new_doc", 0.8)]
        >>> timestamps = {
        ...     "old_doc": "2024-01-10T00:00:00+00:00",  # 5 days old
        ...     "new_doc": "2024-01-15T00:00:00+00:00",  # just created
        ... }
        >>> decayed = apply_recency_decay(scored, timestamps, 0.01)
        >>> # new_doc might now rank higher due to old_doc's decay
    """
    if not scored_results:
        return []

    if now is None:
        now = datetime.now(UTC)

    decayed_results: list[tuple[str, float]] = []

    for document, score in scored_results:
        if document in timestamp_map:
            recency_factor = calculate_recency_factor(
                timestamp_map[document], decay_rate, now
            )
            decayed_score = score * recency_factor
        else:
            # No timestamp available, keep original score
            decayed_score = score

        decayed_results.append((document, decayed_score))

    # Sort by decayed score descending
    decayed_results.sort(key=lambda x: x[1], reverse=True)

    return decayed_results
