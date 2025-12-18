"""Cross-encoder reranker using FlagEmbedding's FlagReranker.

This module provides a local cross-encoder model for fast reranking of search
results before passing top candidates to the LLM reranker. This two-stage
approach reduces LLM API costs by limiting the number of candidates that
require expensive LLM scoring.

Uses FlagEmbedding's FlagReranker which is optimized for BGE reranker models
with built-in FP16 support and score normalization.

Example:
    >>> from openmemories.infrastructure import CrossEncoderConfig, CrossEncoderReranker
    >>> config = CrossEncoderConfig.from_app_config(app_config)
    >>> reranker = CrossEncoderReranker(config=config, logger=logger)
    >>> results = reranker.rerank("Python tutorials", candidates)
"""

import threading
import warnings
from dataclasses import dataclass
from typing import Any

from FlagEmbedding import FlagReranker
from pydantic import BaseModel, ConfigDict, PrivateAttr

from openmemories.application.config import Config


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
    score_threshold: float = 0.0
    use_fp16: bool = True
    normalize: bool = True
    max_length: int = 512
    min_results: int = 0  # Safety net: min results to return (0 = disabled)
    batch_normalize: bool = True  # Enable batch min-max normalization

    @classmethod
    def from_app_config(cls, config: Config) -> "CrossEncoderConfig":
        """Create CrossEncoderConfig from application Config.

        Args:
            config: Application configuration object.

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
            batch_normalize=config.reranker_batch_normalize,
        )


class CrossEncoderReranker(BaseModel):
    """Cross-encoder reranker using FlagEmbedding's FlagReranker.

    This class provides fast, local reranking using FlagReranker which is
    optimized for BGE reranker models. It jointly encodes query and document
    pairs and is designed to be used as the first stage in a two-stage
    reranking pipeline, reducing the number of candidates that need expensive
    LLM scoring.

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
    logger: Any = None

    _model: FlagReranker | None = PrivateAttr(default=None)
    _init_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    @property
    def model(self) -> FlagReranker:
        """Get FlagReranker model (thread-safe lazy initialization).

        Returns:
            Initialized FlagReranker model.

        Note:
            The model is loaded on first access and cached. Loading uses a
            lock to ensure thread safety in concurrent environments.
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

                # Suppress tokenizer optimization warning from transformers
                # ("You're using a XLMRobertaTokenizerFast tokenizer...")
                # This is an internal FlagEmbedding implementation detail
                from transformers import logging as hf_logging

                original_verbosity = hf_logging.get_verbosity()
                hf_logging.set_verbosity_error()

                try:
                    self._model = FlagReranker(
                        self.config.model_name,
                        use_fp16=self.config.use_fp16,
                        devices=[self.config.device],
                    )

                    # Suppress "using the `__call__` method is faster" warning
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
    ) -> list[tuple[str, float]]:
        """Rerank candidates using FlagReranker scores.

        Scores each candidate document against the query using FlagReranker,
        applies score threshold filtering, sorts by score descending, and returns
        the top-k results.

        Args:
            query: Search query string.
            candidates: List of (document, fusion_score) tuples from RRF fusion.
                The fusion_score is preserved as metadata but not used for ranking.

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

        # Build query-document pairs for FlagReranker
        pairs: list[tuple[str, str]] = [(query, doc) for doc, _ in candidates]

        # Score all pairs using FlagReranker
        scores = self.model.compute_score(
            pairs,
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
            normalize=self.config.normalize,
        )

        # Handle single-pair case: compute_score returns float instead of list
        if isinstance(scores, (int, float)):
            scores = [scores]

        # Combine documents with their cross-encoder scores
        scored: list[tuple[str, float]] = [
            (doc, float(score))
            for (doc, _), score in zip(candidates, scores, strict=True)
        ]

        # Apply batch normalization if enabled
        if self.config.batch_normalize:
            from openmemories.application.memory.reranking import (
                normalize_reranker_scores,
            )

            raw_scores = [s for _, s in scored]
            scored = normalize_reranker_scores(scored)
            if self.logger:
                self.logger.info(
                    f"   Batch normalization: enabled (raw range: {min(raw_scores):.4f}-{max(raw_scores):.4f} -> normalized: 0.0-1.0)",
                    extra={
                        "batch_normalize": True,
                        "raw_min": min(raw_scores),
                        "raw_max": max(raw_scores),
                    },
                )

        # Log threshold and individual scores for transparency
        if self.logger:
            threshold = self.config.score_threshold
            self.logger.info(
                f"   FlagReranker scoring (threshold: {threshold:.2f}, normalize: {self.config.normalize}, batch_norm: {self.config.batch_normalize}):",
                extra={
                    "threshold": threshold,
                    "normalize": self.config.normalize,
                    "batch_normalize": self.config.batch_normalize,
                    "candidate_count": len(scored),
                },
            )

            # Log each candidate with [KEEP]/[FILTER] status
            for idx, (doc, score) in enumerate(scored, 1):
                status = "[KEEP]" if score >= threshold else "[FILTER]"
                # Truncate long messages for display
                preview = doc[:60] + "..." if len(doc) > 60 else doc
                self.logger.info(
                    f"      [{idx}] {status} score={score:.4f} -> {preview}",
                    extra={
                        "candidate_index": idx,
                        "score": score,
                        "threshold": threshold,
                        "status": "keep" if score >= threshold else "filter",
                        "message_preview": preview,
                    },
                )

        # Sort by score descending before threshold filtering
        scored.sort(key=lambda x: x[1], reverse=True)

        # Apply threshold with optional safety net
        from openmemories.application.memory.reranking import (
            apply_threshold_with_safety_net,
        )

        pre_filter_count = len(scored)
        scored = apply_threshold_with_safety_net(
            scored,
            threshold=self.config.score_threshold,
            min_results=self.config.min_results,
        )

        if self.logger:
            kept = len(scored)
            filtered = pre_filter_count - kept
            self.logger.info(
                f"   Score threshold filtering: {kept}/{pre_filter_count} passed (threshold: {self.config.score_threshold:.2f}, min_results: {self.config.min_results})",
                extra={
                    "threshold": self.config.score_threshold,
                    "min_results": self.config.min_results,
                    "passed": kept,
                    "filtered": filtered,
                    "total": pre_filter_count,
                },
            )

        # Limit to top_k
        result = scored[: self.config.top_k]

        return result

    async def rerank_async(
        self,
        query: str,
        candidates: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Async wrapper for cross-encoder reranking.

        FlagReranker inference is CPU/GPU-bound, so we run it in a thread
        pool to avoid blocking the event loop.

        Args:
            query: Search query string.
            candidates: List of (document, fusion_score) tuples from RRF fusion.

        Returns:
            List of (document, cross_encoder_score) tuples, sorted by score
            descending, filtered by score_threshold, and limited to top_k results.
        """
        from asyncer import asyncify

        return await asyncify(self.rerank)(query, candidates)
