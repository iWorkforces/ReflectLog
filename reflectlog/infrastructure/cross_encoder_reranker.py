"""Cross-encoder reranker using FlagEmbedding's FlagReranker.

This module provides a local cross-encoder model for fast reranking of search
results after fusion. The local model avoids API cost and keeps search
latency on-device.

Uses FlagEmbedding's FlagReranker which is optimized for BGE reranker models
with built-in FP16 support and score normalization.

Example:
    >>> from reflectlog.infrastructure.cross_encoder_reranker import CrossEncoderConfig, CrossEncoderReranker
    >>> config = CrossEncoderConfig.from_app_config(app_config)
    >>> reranker = CrossEncoderReranker(config=config, logger=logger)
    >>> results = reranker.rerank("Python tutorials", candidates)
"""

from dataclasses import dataclass
import importlib
import threading
from typing import Protocol, final
import warnings

from asyncer import asyncify
from pydantic import BaseModel, ConfigDict, PrivateAttr

from reflectlog.core.config import IAppConfig
from reflectlog.core.logging import IStructuredLogger
from reflectlog.infrastructure.reranker_post_processor import (
    RerankerPostProcessor,
)


@dataclass(frozen=True)
class CrossEncoderConfig:
    """Configuration for cross-encoder reranking using FlagReranker.

    Attributes:
        model_name: Hugging Face model identifier for the cross-encoder.
            Recommended: "BAAI/bge-reranker-v2-m3" (multilingual, high quality).
        enabled: Whether cross-encoder reranking is enabled.
        top_k: Number of top results to return after reranking.
        device: Device for inference ('cpu', 'cuda', 'mps').
        batch_size: Batch size for cross-encoder inference.
        score_threshold: Minimum cross-encoder score to keep (0.0 = keep all).
        use_fp16: Enable FP16 for faster computation with slight quality tradeoff.
        normalize: Apply sigmoid to normalize scores to 0-1 range.
        max_length: Maximum token length for query-document pairs.
    """

    model_name: str = "BAAI/bge-reranker-v2-m3"
    enabled: bool = True
    top_k: int = 20
    device: str = "cpu"
    batch_size: int = 32
    score_threshold: float = 0.5
    use_fp16: bool = True
    normalize: bool = True
    max_length: int = 512
    min_results: int = 1  # Safety net: keep at least the best CE hit
    batch_normalize: bool = True  # Ignored when normalize (sigmoid) is on
    enable_recency_boost: bool = True  # Include memory age in recency decay calculation
    recency_decay_rate: float = 0.001  # Decay rate per hour: exp(-rate * hours_old)

    @classmethod
    def from_config(cls, config: IAppConfig) -> CrossEncoderConfig:
        """Create CrossEncoderConfig from IAppConfig protocol.

        Args:
            config: Configuration satisfying IAppConfig protocol.

        Returns:
            CrossEncoderConfig instance configured from app settings.
        """
        return cls(
            model_name=config.cross_encoder_model,
            enabled=config.reranker_engine == "cross_encoder",
            top_k=config.cross_encoder_top_k,
            device=config.cross_encoder_device,
            batch_size=config.cross_encoder_batch_size,
            score_threshold=config.cross_encoder_score_threshold,
            use_fp16=config.cross_encoder_use_fp16,
            normalize=config.cross_encoder_normalize,
            max_length=config.cross_encoder_max_length,
            min_results=config.reranker_min_results,
            batch_normalize=(
                config.reranker_batch_normalize and not config.cross_encoder_normalize
            ),
            enable_recency_boost=config.enable_recency_boost,
            recency_decay_rate=config.recency_decay_rate,
        )


class FlagRerankerProtocol(Protocol):
    def compute_score(
        self,
        sentence_pairs: list[tuple[str, str]],
        *,
        batch_size: int,
        max_length: int,
        normalize: bool,
    ) -> float | list[float]: ...


@final
class CrossEncoderReranker(BaseModel):
    """Cross-encoder reranker using FlagEmbedding's FlagReranker.

    This class provides fast, local reranking using FlagReranker which is
    optimized for BGE reranker models. It jointly encodes query and document
    pairs and is the only rerank stage in the search pipeline.

    The FlagReranker model is lazily loaded on first use and cached for
    subsequent calls. Loading is thread-safe.

    Attributes:
        config: CrossEncoderConfig with model settings.
        logger: Optional structured logger for debug/info messages.

    Example:
        >>> config = CrossEncoderConfig.from_app_config(app_config)
        >>> reranker = CrossEncoderReranker(config=config, logger=logger)
        >>> # candidates: [(document, fusion_score), ...]
        >>> reranked = reranker.rerank("Python tutorials", candidates)
        >>> # Returns top-k candidates sorted by cross-encoder score
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: CrossEncoderConfig
    logger: IStructuredLogger | None = None

    _model: FlagRerankerProtocol | None = PrivateAttr(default=None)
    _init_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _post_processor: RerankerPostProcessor = PrivateAttr()

    def model_post_init(self, _context: object, /) -> None:
        """Initialize post-processor after Pydantic model init."""
        self._post_processor = RerankerPostProcessor(
            min_results=self.config.min_results,
            batch_normalize=self.config.batch_normalize,
            logger=self.logger,
            normalize_log_level="info",
        )

    @property
    def model(self) -> FlagRerankerProtocol:
        """Get FlagReranker model (thread-safe lazy initialization).

        Returns:
            Initialized FlagReranker model.

        Note:
            The model is loaded on first access and cached. Loading uses a
            lock to ensure thread safety in concurrent environments.
            FlagEmbedding is imported lazily for performance, not to avoid dependency issues.
        """
        if self._model is not None:
            return self._model

        with self._init_lock:
            # Double-check locking pattern
            if self._model is None:
                if self.logger:
                    self.logger.info(
                        f"Loading FlagReranker model: {self.config.model_name}",
                        extra={
                            "device": self.config.device,
                            "use_fp16": self.config.use_fp16,
                        },
                    )

                flag_embedding_module = importlib.import_module("FlagEmbedding")
                flag_reranker_class = flag_embedding_module.FlagReranker

                # Suppress tokenizer optimization warning from transformers
                # ("You're using a XLMRobertaTokenizerFast tokenizer...")
                # This is an internal FlagEmbedding implementation detail
                from transformers import logging as hf_logging

                original_verbosity = hf_logging.get_verbosity()
                hf_logging.set_verbosity_error()

                try:
                    self._model = flag_reranker_class(
                        self.config.model_name,
                        use_fp16=self.config.use_fp16,
                        devices=[self.config.device],
                    )

                    # Suppress "using with `__call__` method is faster" warning
                    # that appears during compute_score() with fast tokenizers.
                    # This warning is informational and not actionable since
                    # FlagReranker handles tokenization internally.
                    self._suppress_fast_tokenizer_warning()
                finally:
                    hf_logging.set_verbosity(original_verbosity)

                if self.logger:
                    self.logger.info(
                        "FlagReranker model loaded successfully",
                        extra={
                            "model": self.config.model_name,
                            "use_fp16": self.config.use_fp16,
                        },
                    )

        # Model is guaranteed to be initialized after lock section
        assert self._model is not None
        return self._model

    def _suppress_fast_tokenizer_warning(self) -> None:
        """Suppress the fast tokenizer padding warning.

        The warning "You're using a XLMRobertaTokenizerFast tokenizer..."
        is informational and not actionable since FlagReranker handles
        tokenization internally. This method suppresses it via two approaches:

        1. Set the tokenizer's deprecation_warnings flag (if accessible)
        2. Add a global warnings filter for this specific message
        """
        # Approach 1: Try to set the tokenizer's deprecation flag directly
        # This is the cleanest solution if the tokenizer is accessible
        if self._model is not None:
            # FlagReranker may store tokenizer in different attributes
            tokenizer = getattr(self._model, "tokenizer", None)
            if tokenizer is None:
                # Try accessing via the model's internal structure
                model_obj = getattr(self._model, "model", None)
                if model_obj is not None:
                    tokenizer = getattr(model_obj, "tokenizer", None)

            if tokenizer is not None and hasattr(tokenizer, "deprecation_warnings"):
                tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
                return

        # Approach 2: Use warnings filter as fallback
        # This catches the warning regardless of tokenizer accessibility
        warnings.filterwarnings(
            "ignore",
            message=r"You're using a \w+TokenizerFast tokenizer.*using the `__call__` method is faster",
            category=UserWarning,
            module=r"transformers\.tokenization_utils_base",
        )

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        timestamp_map: dict[str, str] | None = None,
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Rerank candidates using FlagReranker scores with optional recency decay.

        Scores each candidate document against the query using FlagReranker,
        normalizes, gates on the CE threshold, optionally reorders by recency,
        and then returns the top-k results.

        Args:
            query: Search query string.
            candidates: List of (document, fusion_score) tuples from RRF fusion.
                The fusion_score is preserved as metadata but not used for ranking.
            timestamp_map: Optional mapping of document text to ISO 8601 created_at
                timestamps. Used for recency decay calculation when enabled.

        Returns:
            List of (document, cross_encoder_score) tuples, sorted by score
            descending, filtered by score_threshold, and limited to top_k results.
            Returns original candidates unchanged if reranking is disabled or
            candidates is empty.

        Note:
            - If enabled=False, returns candidates unchanged (pass-through)
            - If candidates is empty, returns empty list
            - With normalize=True, scores are in [0, 1] range (sigmoid applied)
            - With normalize=False, scores can be any real number
            - Recency decay reorders after the score threshold so age
              cannot empty a batch that already passed CE quality
        """
        if not candidates:
            return []

        if not self.config.enabled:
            if self.logger:
                self.logger.debug(
                    "Cross-encoder disabled, returning original candidates",
                    extra={"candidate_count": len(candidates)},
                )
            return candidates

        scored = self._compute_scores(query, candidates)
        scored = self._post_processor.normalize(scored)
        self._log_candidate_scores(scored)
        scored.sort(key=lambda x: x[1], reverse=True)
        # Gate on pre-decay CE quality so recency only reorders survivors.
        # min_results is CE-quality insurance (applied here, before decay).
        # top_k is applied after decay so recency can promote later ranks.
        scored = self._apply_threshold(scored)
        scored = self._apply_recency_reorder(scored, timestamp_map)
        limit = self.config.top_k if top_k is None else max(self.config.top_k, top_k)
        return scored[:limit]

    def _compute_scores(
        self,
        query: str,
        candidates: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Score candidates using FlagReranker and combine with documents."""
        pairs: list[tuple[str, str]] = [(query, doc) for doc, _ in candidates]

        scores = self.model.compute_score(
            pairs,
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
            normalize=self.config.normalize,
        )

        # Handle single-pair case: compute_score returns float instead of list
        if isinstance(scores, (int, float)):
            scores = [scores]

        return [
            (doc, float(score))
            for (doc, _), score in zip(candidates, scores, strict=True)
        ]

    def _apply_recency_reorder(
        self,
        scored: list[tuple[str, float]],
        timestamp_map: dict[str, str] | None,
    ) -> list[tuple[str, float]]:
        """Reorder survivors by recency. Missing or empty timestamps disable decay."""
        decay_enabled = (
            self.config.enable_recency_boost
            and self.config.recency_decay_rate > 0
            and bool(timestamp_map)
            and all(bool(timestamp_map.get(doc)) for doc, _ in scored)
        )
        return self._post_processor.apply_decay(
            scored,
            timestamp_map,
            self.config.recency_decay_rate,
            enabled=decay_enabled,
        )

    def _log_candidate_scores(self, scored: list[tuple[str, float]]) -> None:
        """Log individual candidate scores with keep/filter status."""
        if not self.logger:
            return

        threshold = self.config.score_threshold
        self.logger.debug(
            f"   FlagReranker scoring (threshold: {threshold:.2f}), "
            f"normalize: {self.config.normalize}, batch_norm: {self.config.batch_normalize}):",
            extra={
                "threshold": threshold,
                "normalize": self.config.normalize,
                "batch_normalize": self.config.batch_normalize,
                "candidate_count": len(scored),
            },
        )

        for idx, (doc, score) in enumerate(scored, 1):
            status = "[KEEP]" if score >= threshold else "[FILTER]"
            preview = doc[:60] + "..." if len(doc) > 60 else doc
            self.logger.debug(
                f"      [{idx}] {status} score={score:.4f} -> {preview}",
                extra={
                    "candidate_index": idx,
                    "score": score,
                    "threshold": threshold,
                    "status": "keep" if score >= threshold else "filter",
                    "message_preview": preview,
                },
            )

    def _apply_threshold(
        self,
        scored: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Filter by score threshold (min_results is CE-quality insurance)."""
        pre_filter_count = len(scored)
        scored = self._post_processor.filter_by_threshold(
            scored,
            threshold=self.config.score_threshold,
        )

        if self.logger:
            kept = len(scored)
            filtered = pre_filter_count - kept
            self.logger.info(
                f"   Score threshold filtering: {kept}/{pre_filter_count} "
                f"passed (threshold: {self.config.score_threshold:.2f}), "
                f"min_results: {self.config.min_results})",
                extra={
                    "threshold": self.config.score_threshold,
                    "min_results": self.config.min_results,
                    "passed": kept,
                    "filtered": filtered,
                    "total": pre_filter_count,
                },
            )

        return scored

    async def rerank_async(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        timestamp_map: dict[str, str] | None = None,
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Async wrapper for cross-encoder reranking.

        FlagReranker inference is CPU/GPU-bound, so we run it in a thread
        pool to avoid blocking the event loop.

        Args:
            query: Search query string.
            candidates: List of (document, fusion_score) tuples from RRF fusion.
            timestamp_map: Optional mapping of document text to ISO 8601 created_at
                timestamps. Used for recency decay calculation when enabled.

        Returns:
            List of (document, cross_encoder_score) tuples, sorted by score
            descending, filtered by score_threshold, and limited to top_k results.
        """
        return await asyncify(self.rerank)(
            query, candidates, timestamp_map, top_k
        )
