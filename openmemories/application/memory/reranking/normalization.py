"""Score normalization utilities for rerankers.

This module provides batch-level min-max normalization to transform reranker
scores into a consistent [0, 1] range, enabling unified threshold semantics
across different reranking engines (LLM, CrossEncoder).

The key insight is that different rerankers produce fundamentally different
score distributions:
- LLMReranker: Scores cluster in 0.7-0.9 range (prompt-calibrated)
- CrossEncoderReranker: Scores cluster in 0.001-0.17 range (sigmoid-normalized logits)

After batch normalization:
- Best score in batch = 1.0
- Worst score in batch = 0.0
- A threshold of 0.5 consistently means "above median relevance"
"""

import numpy as np
from numpy.typing import NDArray

from ccmemories.application.utils.numba_utils import normalize_scores_minmax


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

    return [(doc, float(norm_score)) for doc, norm_score in zip(documents, normalized)]


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
