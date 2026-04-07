"""Scoring, normalization, and recency decay utilities.

This module provides scoring functions used across infrastructure and
application layers. Placed in utility/ (order=0) so infrastructure/
can import without violating the dependency hierarchy.

Functions from numba_utils (JIT-compiled):
    - normalize_scores_minmax: Min-max normalization to [0, 1]
    - distance_to_similarity_cosine: Cosine distance → similarity

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

from numba import jit
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

    # Extract documents and scores
    documents = [doc for doc, _ in scored_results]
    scores: NDArray[np.float64] = np.array(
        [score for _, score in scored_results], dtype=np.float64
    )

    # Use numba-accelerated min-max normalization
    # This handles edge cases like all-equal scores (returns 1.0 for all)
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
