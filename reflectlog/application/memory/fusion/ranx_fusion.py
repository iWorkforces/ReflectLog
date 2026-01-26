"""ranx-based fusion engine implementation."""

import logging
import math
from typing import Any

import numpy as np
from ranx import Run
from ranx import fuse as ranx_fuse

from reflectlog.application.utils.numba_utils import normalize_scores_minmax

# Supported fusion methods (unsupervised, no training data required)
# Note: ranx uses 'sum', 'mnz', 'max' instead of 'combsum', 'combmnz', 'combmax'
SUPPORTED_METHODS: set[str] = {"rrf", "sum", "mnz", "max", "bordafuse"}

# Supported normalization strategies
SUPPORTED_NORMALIZATIONS: set[str] = {"min-max", "max", "sum", "zmuv", "rank", "borda"}

# Default normalization per fusion method (applied to inputs before fusion)
# Note: We use None for RRF since it works on ranks, not scores
# Output scores are normalized in _normalize_output_scores()
DEFAULT_NORMALIZATIONS: dict[str, str | None] = {
    "rrf": None,  # RRF uses rank positions, input normalization not needed
    "sum": None,  # CombSUM: Will normalize inputs if scores are incomparable
    "mnz": None,  # CombMNZ: weighted score addition
    "max": None,  # CombMAX: maximum score
    "bordafuse": None,
}


class RanxFusionEngine:
    """Fusion engine using the ranx library.

    This engine supports multiple fusion algorithms and normalization
    strategies provided by the ranx library. It converts between the
    internal (message, score) tuple format and ranx's Run objects.

    Supported methods:
        - rrf: Reciprocal Rank Fusion (default)
        - sum: CombSUM (score addition)
        - mnz: CombMNZ (weighted score addition)
        - max: CombMAX (maximum score)
        - bordafuse: Borda voting fusion

    Supported normalizations:
        - min-max: Min-max normalization
        - max: Max normalization
        - sum: Sum normalization
        - zmuv: Zero-mean unit-variance normalization
        - rank: Rank-based normalization
        - borda: Borda count normalization
    """

    # Fixed query ID for single-query fusion scenario
    _QUERY_ID = "q"

    def __init__(
        self,
        method: str = "rrf",
        normalization: str | None = None,
        rrf_k: int = 60,
        logger: Any | None = None,
    ):
        """Initialize the ranx fusion engine.

        Args:
            method: Fusion algorithm to use. One of: rrf, sum, mnz,
                   max, bordafuse. Defaults to 'rrf'.
            normalization: Score normalization strategy. One of: min-max, max,
                          sum, zmuv, rank, borda. Defaults to auto-select
                          based on method.
            rrf_k: RRF k parameter (only used when method='rrf'). Lower values
                  give more weight to top ranks. Defaults to 60.
            logger: Optional StructuredLogger instance for debug logging.

        Raises:
            ValueError: If method or normalization is not supported.
        """
        if method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported fusion method: '{method}'. "
                f"Supported methods: {sorted(SUPPORTED_METHODS)}"
            )

        if normalization is not None and normalization not in SUPPORTED_NORMALIZATIONS:
            raise ValueError(
                f"Unsupported normalization: '{normalization}'. "
                f"Supported normalizations: {sorted(SUPPORTED_NORMALIZATIONS)}"
            )

        self._method = method
        self._normalization = normalization or DEFAULT_NORMALIZATIONS.get(method)
        self._rrf_k = rrf_k
        self.logger = logger

    @property
    def method(self) -> str:
        """Return the fusion method name."""
        return self._method

    @property
    def normalization(self) -> str | None:
        """Return the normalization strategy."""
        return self._normalization

    @property
    def rrf_k(self) -> int:
        """Return the RRF k parameter."""
        return self._rrf_k

    def _convert_to_run(self, result_set: list[tuple[str, float]], name: str) -> Run:
        """Convert (message, score) tuples to a ranx Run object.

        Handles duplicate documents within the same result set by averaging their
        scores. This provides better fusion quality than keeping only the first
        occurrence, as multiple high-scoring instances from different sources
        indicate stronger relevance.

        Example:
            Input: [("doc1", 0.9), ("doc2", 0.5), ("doc1", 0.7)]
            Output: Run({"q": {"doc1": 0.8, "doc2": 0.5}})
            # doc1 score: (0.9 + 0.7) / 2 = 0.8

        Args:
            result_set: List of (message, score) tuples. May contain duplicate
                       messages with different scores.
            name: Name for the Run object.

        Returns:
            ranx Run object with messages as document IDs. Duplicate messages
            are represented once with their averaged score.
        """
        if not result_set:
            return Run({self._QUERY_ID: {}}, name=name)

        # Use message content as doc_id, original score as value
        # Handle duplicates within a list by averaging their scores
        # This provides better fusion quality than keeping first occurrence
        doc_score_sums: dict[str, float] = {}
        doc_score_counts: dict[str, int] = {}
        for msg, score in result_set:
            if msg in doc_score_sums:
                doc_score_sums[msg] += score
                doc_score_counts[msg] += 1
            else:
                doc_score_sums[msg] = score
                doc_score_counts[msg] = 1

        # Compute averaged scores
        doc_scores: dict[str, float] = {
            msg: doc_score_sums[msg] / doc_score_counts[msg] for msg in doc_score_sums
        }

        return Run({self._QUERY_ID: doc_scores}, name=name)

    def _convert_from_run(self, run: Run) -> list[tuple[str, float]]:
        """Convert a ranx Run object back to (message, score) tuples.

        Args:
            run: ranx Run object after fusion.

        Returns:
            List of (message, fused_score) tuples sorted by score descending.
        """
        if self._QUERY_ID not in run.run:
            return []

        query_results = run.run[self._QUERY_ID]

        # Convert to list of tuples and sort by score descending
        results = [(doc_id, float(score)) for doc_id, score in query_results.items()]
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def _normalize_output_scores(
        self, results: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        """Normalize output scores to 0-1 range using min-max normalization.

        This ensures fusion scores are comparable to threshold values (e.g., 0.8).

        Performance: Uses numba-accelerated normalization for vectorized operations.

        Args:
            results: List of (message, score) tuples.

        Returns:
            List with scores normalized to 0-1 range.
        """
        if not results:
            return results

        # Extract messages and scores separately for vectorized processing
        messages = [msg for msg, _ in results]
        scores = np.array([score for _, score in results], dtype=np.float64)

        # Use numba-accelerated normalization (single-pass min/max + vectorized normalize)
        normalized_scores = normalize_scores_minmax(scores)

        # Reconstruct result tuples with normalized scores
        return [(msg, float(score)) for msg, score in zip(messages, normalized_scores)]

    def fuse(
        self,
        *result_sets: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Fuse multiple ranked lists using the configured algorithm.

        Args:
            *result_sets: Variable number of (message, score) tuple lists.
                         Each list should be sorted by score descending.

        Returns:
            List of (message, fused_score) tuples sorted by score descending.
        """
        # Filter out empty result sets
        non_empty = [rs for rs in result_sets if rs]
        if not non_empty:
            return []

        # Validate score ranges and log unusual inputs (debug level)
        if self.logger and self.logger.isEnabledFor(logging.DEBUG):
            for i, result_set in enumerate(non_empty):
                scores = [score for _, score in result_set]
                if not scores:
                    continue

                # Check for unusual score ranges
                has_negative = any(score < 0 for score in scores)
                has_nan = any(math.isnan(score) for score in scores)
                has_inf = any(math.isinf(score) for score in scores)
                max_score = max(scores)
                min_score = min(scores)

                # Log if any unusual conditions detected
                if has_negative or has_nan or has_inf:
                    self.logger.debug(
                        f"Unusual score range detected in result set {i}",
                        extra={
                            "result_set_index": i,
                            "score_count": len(scores),
                            "min_score": min_score,
                            "max_score": max_score,
                            "has_negative": has_negative,
                            "has_nan": has_nan,
                            "has_inf": has_inf,
                        },
                    )

        # Convert each result set to a ranx Run
        runs = [
            self._convert_to_run(rs, name=f"run_{i}")
            for i, rs in enumerate(result_sets)
            if rs
        ]

        if not runs:
            return []

        # ranx.fuse() requires at least 2 runs - handle single run case
        if len(runs) == 1:
            sorted_results = self._convert_from_run(runs[0])
            normalized_results = self._normalize_output_scores(sorted_results)
            if self.logger:
                self.logger.debug(
                    f"Single result set - no fusion needed: {len(normalized_results)} results",
                    extra={
                        "method": self._method,
                        "input_sets": len(result_sets),
                        "non_empty_sets": 1,
                        "unique_count": len(normalized_results),
                    },
                )
            return normalized_results

        # Build params for ranx.fuse()
        params: dict[str, Any] | None = None
        if self._method == "rrf":
            params = {"k": self._rrf_k}

        # Perform fusion
        combined = ranx_fuse(
            runs=runs,
            norm=self._normalization,
            method=self._method,
            params=params,
        )

        # Convert back to tuple format
        sorted_results = self._convert_from_run(combined)

        # Normalize output scores to 0-1 range for consistent thresholding
        normalized_results = self._normalize_output_scores(sorted_results)

        if self.logger:
            self.logger.debug(
                f"Fusion completed: {len(normalized_results)} unique results "
                f"from {len(non_empty)} result sets",
                extra={
                    "method": self._method,
                    "normalization": self._normalization,
                    "input_sets": len(result_sets),
                    "non_empty_sets": len(non_empty),
                    "unique_count": len(normalized_results),
                },
            )

        return normalized_results
