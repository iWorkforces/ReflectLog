"""Reranking utilities for score normalization, filtering, and recency decay.

This module provides functions for:
1. Normalizing reranker scores to a consistent [0, 1] range using batch min-max
   normalization, enabling unified threshold semantics across different
   reranking engines (LLM, CrossEncoder).
2. Applying recency decay to scores, giving newer memories an advantage over
   older ones using exponential decay formula.

Example:
    >>> from reflectlog.application.memory.reranking import (
    ...     normalize_reranker_scores,
    ...     apply_threshold_with_safety_net,
    ...     calculate_recency_factor,
    ...     apply_recency_decay,
    ... )
    >>> # Normalize CrossEncoder's low scores (0.001-0.17) to [0, 1]
    >>> scored = [("doc1", 0.17), ("doc2", 0.05), ("doc3", 0.001)]
    >>> normalized = normalize_reranker_scores(scored)
    >>> # normalized: [("doc1", 1.0), ("doc2", 0.29), ("doc3", 0.0)]

    >>> # Apply recency decay
    >>> timestamps = {"doc1": "2024-01-10T00:00:00+00:00"}
    >>> decayed = apply_recency_decay(normalized, timestamps, decay_rate=0.01)
"""

from .normalization import (
    apply_recency_decay,
    apply_threshold_with_safety_net,
    calculate_recency_factor,
    normalize_reranker_scores,
)

__all__ = [
    "normalize_reranker_scores",
    "apply_threshold_with_safety_net",
    "calculate_recency_factor",
    "apply_recency_decay",
]
