"""Shared post-processing for reranking results.

Extracts the common normalize, threshold, and recency helpers used by
CrossEncoderReranker into a single composition object, avoiding Pydantic
BaseModel MRO issues that prevent mixin-based reuse.

Callers apply the steps themselves. CrossEncoderReranker gates on
pre-decay CE scores, then reorders survivors by recency, then slices top_k.
"""

from typing import TYPE_CHECKING

from reflectlog.utility.scoring import (
    apply_recency_decay,
    apply_threshold_with_safety_net,
    normalize_reranker_scores,
)

if TYPE_CHECKING:
    from reflectlog.core.logging import IStructuredLogger


class RerankerPostProcessor:
    """Shared post-processing pipeline for reranking results.

    Encapsulates the post-processing helpers shared by rerankers:
    1. Batch min-max score normalization
    2. Threshold filtering with safety-net minimum (CE-quality insurance)
    3. Recency decay (exponential time-based reorder)

    Designed for composition: each Pydantic-based reranker holds a
    ``_post_processor`` instance and delegates post-processing to it.
    """

    __slots__ = ("_batch_normalize", "_logger", "_min_results", "_normalize_log_level")

    def __init__(
        self,
        min_results: int,
        batch_normalize: bool,
        logger: IStructuredLogger | None = None,
        normalize_log_level: str = "debug",
    ) -> None:
        self._min_results = min_results
        self._batch_normalize = batch_normalize
        self._logger = logger
        self._normalize_log_level = normalize_log_level

    # ------------------------------------------------------------------
    # Step 1: batch min-max normalization
    # ------------------------------------------------------------------

    def normalize(
        self,
        scores: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Apply batch min-max normalization when enabled.

        Args:
            scores: Scored results from the reranker engine.

        Returns:
            Normalized scores (0-1 range) if batch_normalize is enabled,
            otherwise the original scores unchanged.
        """
        if not self._batch_normalize or not scores:
            return scores

        raw_scores = [s for _, s in scores]
        normalized = normalize_reranker_scores(scores)

        if self._logger:
            message = (
                f"Batch normalization: enabled "
                f"(raw range: {min(raw_scores):.4f}-{max(raw_scores):.4f} "
                f"-> normalized: 0.0-1.0)"
            )
            extra = {
                "batch_normalize": True,
                "raw_min": min(raw_scores),
                "raw_max": max(raw_scores),
            }
            level = self._normalize_log_level
            if level == "debug":
                self._logger.debug(message, extra=extra)
            elif level == "warning":
                self._logger.warning(message, extra=extra)
            elif level == "error":
                self._logger.error(message, extra=extra)
            else:
                self._logger.info(message, extra=extra)

        return normalized

    # ------------------------------------------------------------------
    # Step 2: recency decay
    # ------------------------------------------------------------------

    def apply_decay(
        self,
        scores: list[tuple[str, float]],
        timestamp_map: dict[str, str] | None,
        decay_rate: float,
        enabled: bool,
    ) -> list[tuple[str, float]]:
        """Apply exponential recency decay when conditions are met.

        When decay is applied the returned list is sorted by decayed score
        descending (handled by ``apply_recency_decay``).  When decay is
        *not* applied the caller is responsible for any sorting.

        Args:
            scores: Scored results (post-normalization).
            timestamp_map: Document text → ISO 8601 timestamp mapping.
            decay_rate: Decay rate per hour (e.g. 0.01).
            enabled: Whether recency boost is enabled in config.

        Returns:
            Tuple ``(decayed_scores, decay_was_applied)`` is NOT returned
            for API simplicity.  Instead returns the (possibly decayed)
            scores list.  Callers that need to know whether decay was
            applied can check the same condition used here.
        """
        if not (enabled and decay_rate > 0 and timestamp_map and scores):
            return scores

        pre_decay_scores = [s for _, s in scores]
        decayed = apply_recency_decay(scores, timestamp_map, decay_rate)
        post_decay_scores = [s for _, s in decayed]

        if self._logger:
            self._logger.debug(
                f"Recency decay: applied (rate={decay_rate}), "
                f"score range: {max(pre_decay_scores):.4f}-{min(pre_decay_scores):.4f} -> "
                f"{max(post_decay_scores):.4f}-{min(post_decay_scores):.4f})",
                extra={
                    "recency_decay": True,
                    "decay_rate": decay_rate,
                    "pre_decay_max": max(pre_decay_scores),
                    "post_decay_max": max(post_decay_scores),
                },
            )

        return decayed

    # ------------------------------------------------------------------
    # Step 3: threshold filtering with safety net
    # ------------------------------------------------------------------

    def filter_by_threshold(
        self,
        scores: list[tuple[str, float]],
        threshold: float,
    ) -> list[tuple[str, float]]:
        """Filter results by score threshold with safety-net minimum.

        Args:
            scores: Scored results sorted by score descending.
            threshold: Minimum score to keep (inclusive).

        Returns:
            Filtered results, guaranteed to contain at least
            ``min_results`` entries when enough candidates exist.
        """
        if not scores:
            return []

        return apply_threshold_with_safety_net(
            scores,
            threshold=threshold,
            min_results=self._min_results,
        )
