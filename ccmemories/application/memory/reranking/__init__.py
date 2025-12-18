"""Reranking utilities for score normalization and filtering.

This module provides functions for normalizing reranker scores to a consistent
[0, 1] range using batch min-max normalization, enabling unified threshold
semantics across different reranking engines (LLM, CrossEncoder).

Example:
    >>> from ccmemories.application.memory.reranking import (
    ...     normalize_reranker_scores,
    ...     apply_threshold_with_safety_net,
    ... )
    >>> # Normalize CrossEncoder's low scores (0.001-0.17) to [0, 1]
    >>> scored = [("doc1", 0.17), ("doc2", 0.05), ("doc3", 0.001)]
    >>> normalized = normalize_reranker_scores(scored)
    >>> # normalized: [("doc1", 1.0), ("doc2", 0.29), ("doc3", 0.0)]
"""

from .normalization import (
    apply_threshold_with_safety_net,
    normalize_reranker_scores,
)

__all__ = [
    "normalize_reranker_scores",
    "apply_threshold_with_safety_net",
]
