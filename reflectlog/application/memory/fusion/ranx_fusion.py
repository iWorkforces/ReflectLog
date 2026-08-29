"""ranx-based fusion engine implementation."""

import logging
import math
from typing import TYPE_CHECKING, Any, final, override

if TYPE_CHECKING:
    from reflectlog.core.logging import IStructuredLogger

import numpy as np
from ranx import Run
from ranx import fuse as ranx_fuse

from reflectlog.utility.scoring import (
    compute_rrf_scores_batch,
    compute_weighted_rrf_scores_batch,
    normalize_scores_minmax,
)

from .base import FusionEngine

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


@final
class RanxFusionEngine(FusionEngine):
    """Fusion engine using the ranx library.

    This engine supports multiple fusion algorithms and normalization
    strategies provided by the ranx library. It converts between the
    internal (memory, score) tuple format and ranx's Run objects.

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
        weights: list[float] | None = None,
        logger: IStructuredLogger | None = None,
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
            weights: Optional list of weights for weighted RRF fusion. Must have
                  at least 2 elements if provided. Defaults to None (equal weights).
            logger: Optional StructuredLogger instance for debug logging.

        Raises:
            ValueError: If method or normalization is not supported.
        """
        super().__init__()

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
        self._weights = weights
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

    @property
    def weights(self) -> list[float] | None:
        """Return the fusion weights for weighted RRF."""
        return self._weights

    def _convert_to_run(self, result_set: list[tuple[str, float]], name: str) -> Run:
        """Convert (memory, score) tuples to a ranx Run object.

        Handles duplicate documents within the same result set by averaging their
        scores. This provides better fusion quality than keeping only the first
        occurrence, as multiple high-scoring instances from different sources
        indicate stronger relevance.

        Example:
            Input: [("doc1", 0.9), ("doc2", 0.5), ("doc1", 0.7)]
            Output: Run({"q": {"doc1": 0.8, "doc2": 0.5}})
            # doc1 score: (0.9 + 0.7) / 2 = 0.8

        Args:
            result_set: List of (memory, score) tuples. May contain duplicate
                       memories with different scores.
            name: Name for the Run object.

        Returns:
            ranx Run object with memories as document IDs. Duplicate memories
            are represented once with their averaged score.
        """
        if not result_set:
            return Run({self._QUERY_ID: {}}, name=name)

        # Use memory content as doc_id, original score as value
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
        """Convert a ranx Run object back to (memory, score) tuples.

        Args:
            run: ranx Run object after fusion.

        Returns:
            List of (memory, fused_score) tuples sorted by score descending.
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
            results: List of (memory, score) tuples.

        Returns:
            List with scores normalized to 0-1 range.
        """
        if not results:
            return results

        memories = [msg for msg, _ in results]
        scores = np.array([score for _, score in results], dtype=np.float64)
        # Zero-range RRF (disjoint rank-1 ties) must stay at 1.0 so the
        # default fusion_ranking_threshold=0.8 does not drop every hit.
        # Do not use shared min-max here: that maps a tie to 0.5.
        if len(results) == 1 or float(np.max(scores)) == float(np.min(scores)):
            return [(msg, 1.0) for msg in memories]

        normalized_scores = normalize_scores_minmax(scores)
        return [
            (msg, float(score))
            for msg, score in zip(memories, normalized_scores, strict=True)
        ]

    def _fuse_rrf_numba(
        self, result_sets: list[list[tuple[str, float]]]
    ) -> list[tuple[str, float]]:
        """RRF via interned doc ids and the compiled batch scorer."""
        doc_index: dict[str, int] = {}
        docs: list[str] = []
        for result_set in result_sets:
            for memory, _score in result_set:
                if memory not in doc_index:
                    doc_index[memory] = len(docs)
                    docs.append(memory)

        ranks = np.zeros((len(docs), len(result_sets)), dtype=np.int64)
        for run_idx, result_set in enumerate(result_sets):
            seen: set[int] = set()
            for rank, (memory, _score) in enumerate(result_set, start=1):
                doc_idx = doc_index[memory]
                if doc_idx in seen:
                    continue
                ranks[doc_idx, run_idx] = rank
                seen.add(doc_idx)

        if self._weights is not None:
            if len(self._weights) != len(result_sets):
                raise ValueError(
                    f"RRF weights length {len(self._weights)} does not match "
                    f"{len(result_sets)} result sets"
                )
            weight_arr = np.asarray(self._weights, dtype=np.float64)
            scores = compute_weighted_rrf_scores_batch(
                ranks, weight_arr, k=self._rrf_k
            )
        else:
            scores = compute_rrf_scores_batch(ranks, k=self._rrf_k)
        paired = [(docs[idx], float(scores[idx])) for idx in range(len(docs))]
        paired.sort(key=lambda item: item[1], reverse=True)
        normalized = self._normalize_output_scores(paired)
        if self.logger:
            self.logger.debug(
                f"Fusion completed: {len(normalized)} unique results "
                f"from {len(result_sets)} result sets",
                extra={
                    "method": self._method,
                    "input_sets": len(result_sets),
                    "unique_count": len(normalized),
                },
            )
        return normalized

    @override
    def fuse(
        self,
        *result_sets: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Fuse multiple ranked lists using the configured algorithm.

        Args:
            *result_sets: Variable number of (memory, score) tuple lists.
                         Each list should be sorted by score descending.

        Returns:
            List of (memory, fused_score) tuples sorted by score descending.
        """
        # Filter out empty result sets
        non_empty = [rs for rs in result_sets if rs]
        if not non_empty:
            return []

        if len(non_empty) == 1:
            # Keep original backend scores. Min-max on one list stretches the
            # lowest hit to 0.0, and fusion_ranking_threshold=0.8 then drops it.
            return self._convert_from_run(
                self._convert_to_run(non_empty[0], name="run_0")
            )

        if self._method == "rrf":
            return self._fuse_rrf_numba(non_empty)

        # Validate score ranges and log unusual inputs (debug level)
        if self.logger and self.logger.is_enabled_for(logging.DEBUG):
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
            if self.logger:
                self.logger.debug(
                    f"Single result set - no fusion needed: {len(sorted_results)} results",
                    extra={
                        "method": self._method,
                        "input_sets": len(result_sets),
                        "non_empty_sets": 1,
                        "unique_count": len(sorted_results),
                    },
                )
            return sorted_results

        # Build params for ranx.fuse()
        params: dict[str, Any] | None = None
        if self._method == "rrf":
            params = {"k": self._rrf_k}
        if self._weights is not None:
            params = {} if params is None else params
            params["weights"] = self._weights

        try:
            combined = ranx_fuse(
                runs=runs,
                norm=self._normalization,
                method=self._method,
                params=params,
            )
        except TypeError as fuse_error:
            if params is not None and "weights" in params:
                raise RuntimeError(
                    "Fusion algorithm rejected weights; refusing unweighted fallback"
                ) from fuse_error
            raise

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
