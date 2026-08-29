"""LLM-based document reranker for search results."""

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import re
import time as _time
from typing import Any, Protocol, TypedDict, final, override

import anyio
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from reflectlog.core.config import IAppConfig
from reflectlog.core.logging import IStructuredLogger
from reflectlog.core.prompts import (
    format_scoring_prompt,
    format_scoring_prompt_with_age,
)
from reflectlog.infrastructure.llm_provider_base import (
    BaseOpenAIProvider,
)
from reflectlog.infrastructure.reranker_post_processor import (
    RerankerPostProcessor,
)


def format_memory_age(created_at: str | None) -> str | None:
    """Format a timestamp into a human-readable age string.

    Args:
        created_at: ISO 8601 timestamp string (e.g., "2024-01-15T10:30:00").
                   If None or invalid, returns None.

    Returns:
        Human-readable age string like "2 hours ago", "3 days ago", etc.
        Returns None if timestamp is invalid or not provided.
    """
    if not created_at:
        return None

    try:
        # Parse ISO 8601 timestamp
        created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        now = datetime.now(created_dt.tzinfo) if created_dt.tzinfo else datetime.now()

        delta = now - created_dt
        total_seconds = delta.total_seconds()

        # Handle negative deltas (future timestamps)
        if total_seconds < 0:
            return "just now"

        # Convert to appropriate unit
        if total_seconds < 60:
            return "just now"
        elif total_seconds < 3600:
            minutes = int(total_seconds / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif total_seconds < 86400:
            hours = int(total_seconds / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif total_seconds < 604800:
            days = int(total_seconds / 86400)
            return f"{days} day{'s' if days != 1 else ''} ago"
        elif total_seconds < 2592000:
            weeks = int(total_seconds / 604800)
            return f"{weeks} week{'s' if weeks != 1 else ''} ago"
        else:
            months = int(total_seconds / 2592000)
            return f"{months} month{'s' if months != 1 else ''} ago"

    except ValueError, TypeError:
        return None


class RelevanceScore(BaseModel):
    """Schema for LLM relevance scoring response.

    Used with OpenAI Structured Outputs to guarantee valid JSON responses
    from the LLM reranker.

    Attributes:
        score: Relevance score from 0.0 (no match) to 1.0 (perfect match).
    """

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score from 0.0 (no match) to 1.0 (perfect match)",
    )


class AnthropicRelevanceResponse(TypedDict, total=False):
    score: float


@dataclass(frozen=True)
class LLMRerankerConfig:
    """Configuration for LLM reranking.

    Attributes:
        api_key: OpenRouter API key for authentication.
        base_url: OpenRouter API base URL.
        model: LLM model identifier (e.g., 'x-ai/grok-4.1-fast').
        score_threshold: Minimum relevance score to keep results (0.0-1.0).
        max_concurrency: Maximum parallel LLM calls.
        timeout: HTTP request timeout in seconds.
        min_results: Safety net: min results to return (0 = disabled).
        batch_normalize: Enable batch min-max normalization.
        provider: LLM provider ('openai' or 'anthropic').
        enable_recency_boost: Include memory age in reranking context.
    """

    api_key: str
    base_url: str
    model: str
    score_threshold: float = 0.5
    max_concurrency: int = 5
    timeout: float = 30.0  # 30 seconds default for reranking requests
    min_results: int = 0  # Safety net: min results to return (0 = disabled)
    batch_normalize: bool = True  # Enable batch min-max normalization
    provider: str = "anthropic"  # Provider: "openai" or "anthropic"
    enable_recency_boost: bool = True  # Include memory age in reranking context
    recency_decay_rate: float = 0.01  # Decay rate per hour: exp(-rate * hours_old)

    @classmethod
    def from_config(cls, config: IAppConfig) -> LLMRerankerConfig:
        """Create LLMRerankerConfig from IAppConfig protocol.

        Args:
            config: Configuration satisfying IAppConfig protocol.

        Returns:
            LLMRerankerConfig instance configured from app settings.
        """
        return cls(
            api_key=config.llm_api_key,
            base_url=config.llm_api_base_url,
            model=config.llm_model,
            score_threshold=config.search_score_threshold,
            max_concurrency=config.rerank_max_concurrency,
            min_results=config.reranker_min_results,
            batch_normalize=config.reranker_batch_normalize,
            provider=config.llm_provider,
            enable_recency_boost=config.enable_recency_boost,
            recency_decay_rate=config.recency_decay_rate,
        )


class IRerankerProvider(Protocol):
    """Protocol for reranker LLM providers.

    Defines the interface for LLM providers used in document reranking.
    """

    async def score_document(
        self,
        query: str,
        document: str,
        fallback_score: float,
        memory_age: str | None = None,
    ) -> tuple[str, float]:
        """Score a document's relevance to the query.

        Args:
            query: Search query string.
            document: Document text to score.
            fallback_score: Score to use if LLM call fails.
            memory_age: Human-readable age string (e.g., "2 hours ago").
                       If provided, temporal context is included in the prompt.

        Returns:
            Tuple of (document, score) where score is LLM relevance or fallback.
        """
        ...


@final
class OpenAIRerankerProvider(BaseOpenAIProvider, IRerankerProvider):
    """OpenAI/OpenRouter-based reranking provider.

    Uses AsyncOpenAI client with structured JSON output for reliable parsing.
    Supports fallback to json_object mode for models without structured output.

    Inherits common functionality from BaseOpenAIProvider:
    - AsyncOpenAI client initialization with HTTP/2 support
    - Structured output with fallback to json_object
    - Safe JSON parsing with clamping
    """

    @override
    async def score_document(
        self,
        query: str,
        document: str,
        fallback_score: float,
        memory_age: str | None = None,
    ) -> tuple[str, float]:
        """Score a document's relevance using OpenAI API.

        Args:
            query: Search query string.
            document: Document text to score.
            fallback_score: Score to use if LLM call fails.
            memory_age: Human-readable age string (e.g., "2 hours ago").
                       If provided, temporal context is included in the prompt.

        Returns:
            Tuple of (document, score).
        """
        try:
            # Format the scoring prompt with or without temporal context
            if memory_age:
                prompt = format_scoring_prompt_with_age(
                    query=query, document=document, memory_age=memory_age
                )
            else:
                prompt = format_scoring_prompt(query=query, document=document)

            # Call LLM with structured output (or fallback)
            result = await self._call_llm_with_structured_output(
                prompt=prompt,
                response_schema=RelevanceScore,
                max_tokens=50,  # Only need short JSON response
            )

            # Extract and clamp score using base class helper
            score = self._extract_float_field(result, "score", default=fallback_score)

            if self._logger:
                self._logger.debug(
                    f"LLM scored document: {score:.2f}",
                    extra={
                        "query": query[:50],
                        "document_preview": document[:50],
                        "llm_score": score,
                        "provider": "openai",
                        "memory_age": memory_age,
                    },
                )

            return (document, score)

        except json.JSONDecodeError as e:
            if self._logger:
                self._logger.warning(
                    f"Invalid JSON from LLM, using fallback score: {e}",
                    extra={"error": str(e), "fallback_score": fallback_score},
                )
            return (document, fallback_score)

        except Exception as e:
            if self._logger:
                self._logger.warning(
                    f"LLM scoring failed, using fallback score: {e}",
                    extra={"error": str(e), "fallback_score": fallback_score},
                )
            return (document, fallback_score)


@final
class AnthropicRerankerProvider(IRerankerProvider):
    """Anthropic Claude-based reranking provider.

    Uses Claude Agent SDK via utility module for LLM calls.
    Parses JSON from plain text responses with multiple fallback strategies.
    """

    def __init__(
        self,
        model: str | None = None,
        logger: IStructuredLogger | None = None,
    ):
        """Initialize Anthropic reranker provider.

        Calls init_credentials() to set up OAuth credentials.

        Args:
            model: LLM model identifier (passed to generate_content).
            logger: Optional structured logger.
        """
        super().__init__()

        # Lazy import to avoid dependency issues
        from reflectlog.utility.utility import init_credentials

        _ = init_credentials(verbose=False)
        self._model = model
        self._logger = logger

    def _extract_json_from_response(
        self, response_text: str
    ) -> AnthropicRelevanceResponse:
        """Extract JSON from plain text response.

        Handles multiple response formats:
        1. Pure JSON
        2. Markdown code blocks (```json ... ```)
        3. Embedded JSON in text

        Args:
            response_text: Raw response text from LLM.

        Returns:
            Parsed JSON dictionary.

        Raises:
            ValueError: If JSON extraction fails.
        """
        text = response_text.strip()

        # Strategy 1: Direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Markdown code block
        code_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
        match = re.search(code_block_pattern, text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find embedded JSON object
        json_pattern = r"\{[\s\S]*?\}"
        matches = list(re.finditer(json_pattern, text))
        for match in matches:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue

        raise ValueError(f"Could not extract JSON from response: {text[:200]}")

    @override
    async def score_document(
        self,
        query: str,
        document: str,
        fallback_score: float,
        memory_age: str | None = None,
    ) -> tuple[str, float]:
        """Score a document's relevance using Anthropic Claude.

        Args:
            query: Search query string.
            document: Document text to score.
            fallback_score: Score to use if LLM call fails.
            memory_age: Human-readable age string (e.g., "2 hours ago").
                       If provided, temporal context is included in the prompt.

        Returns:
            Tuple of (document, score).
        """
        # Lazy import to avoid dependency issues
        from reflectlog.utility.utility import generate_content

        try:
            # Format the scoring prompt with or without temporal context
            if memory_age:
                prompt = format_scoring_prompt_with_age(
                    query=query, document=document, memory_age=memory_age
                )
            else:
                prompt = format_scoring_prompt(query=query, document=document)

            # Call generate_content with model parameter
            response_text = await generate_content(
                prompt=prompt,
                model=self._model,
                allowed_tools=[],
            )

            # Parse JSON from response
            result = self._extract_json_from_response(response_text)
            score = float(result.get("score", fallback_score))

            # Clamp score to valid range
            score = max(0.0, min(1.0, score))

            if self._logger:
                self._logger.debug(
                    f"LLM scored document: {score:.2f}",
                    extra={
                        "query": query[:50],
                        "document_preview": document[:50],
                        "llm_score": score,
                        "provider": "anthropic",
                        "memory_age": memory_age,
                    },
                )

            return (document, score)

        except Exception as e:
            if self._logger:
                self._logger.warning(
                    f"Anthropic LLM scoring failed, using fallback score: {e}",
                    extra={
                        "error": str(e),
                        "fallback_score": fallback_score,
                        "provider": "anthropic",
                    },
                )
            return (document, fallback_score)


def create_reranker_provider(
    config: LLMRerankerConfig,
    logger: IStructuredLogger | None = None,
) -> IRerankerProvider:
    """Create a reranker provider based on configuration.

    Factory function that returns the appropriate provider implementation
    based on the provider setting in config.

    Args:
        config: LLMRerankerConfig with provider selection.
        logger: Optional structured logger.

    Returns:
        An IRerankerProvider implementation.

    Raises:
        ValueError: If provider is not supported.
    """
    if config.provider == "openai":
        return OpenAIRerankerProvider(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            timeout=config.timeout,
            logger=logger,
        )
    elif config.provider == "anthropic":
        return AnthropicRerankerProvider(
            model=config.model,
            logger=logger,
        )
    else:
        raise ValueError(
            f"Unsupported provider: '{config.provider}'. "
            f"Valid options: 'openai', 'anthropic'"
        )


@final
class LLMReranker(BaseModel):
    """LLM-based document reranker using configurable LLM providers.

    This class scores search result relevance using an LLM and reranks
    candidates based on their relevance scores. It supports parallel
    scoring with concurrency control and graceful fallback on errors.

    Supports multiple LLM providers via the provider abstraction:
    - 'openai': OpenRouter/OpenAI API with structured JSON output
    - 'anthropic': Claude Agent SDK via utility module

    Attributes:
        config: LLMRerankerConfig with API credentials and settings.
        logger: Optional structured logger for debug/info messages.

    Example:
        >>> config = LLMRerankerConfig.from_app_config(app_config)
        >>> reranker = LLMReranker(config=config, logger=logger)
        >>> results = await reranker.rerank("Python tutorials", candidates)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: LLMRerankerConfig
    logger: IStructuredLogger | None = None

    _provider: IRerankerProvider | None = PrivateAttr(default=None)
    _post_processor: RerankerPostProcessor = PrivateAttr()

    def __init__(self, **data: Any):
        """Initialize LLMReranker with appropriate provider."""
        super().__init__(**data)

        # Initialize provider based on configuration
        self._provider = create_reranker_provider(self.config, self.logger)

        # Shared post-processing pipeline (composition over inheritance)
        self._post_processor = RerankerPostProcessor(
            min_results=self.config.min_results,
            batch_normalize=self.config.batch_normalize,
            logger=self.logger,
        )

        self._rerank_cache: dict[str, tuple[float, float]] = {}
        self._rerank_cache_ttl = 60.0
        self._rerank_cache_max = 256

    def _compute_cache_key(
        self, query: str, document: str, memory_age: str | None
    ) -> str:
        """Compute deterministic cache key for a query-document pair.

        Args:
            query: Search query string.
            document: Document text.
            memory_age: Optional age string included so recency-aware
                scores are not reused for a different age.

        Returns:
            MD5 hash hex string as cache key.
        """
        raw = f"{query}||{document}||{memory_age or ''}"
        return hashlib.md5(raw.encode()).hexdigest()

    async def _score_single(
        self,
        query: str,
        document: str,
        fallback_score: float,
        memory_age: str | None = None,
    ) -> tuple[str, float]:
        """Score a single document's relevance to the query.

        Delegates to the configured provider for actual scoring.
        Uses TTL cache to avoid redundant API calls for identical queries.

        Args:
            query: Search query string.
            document: Document text to score.
            fallback_score: Score to use if LLM call fails.
            memory_age: Human-readable age string (e.g., "2 hours ago").
                       If provided and recency boost is enabled, temporal
                       context is included in the prompt.

        Returns:
            Tuple of (document, score) where score is LLM relevance or fallback.
        """
        if self._provider is None:
            return (document, fallback_score)

        # Check cache first
        cache_key = self._compute_cache_key(query, document, memory_age)
        cached = self._rerank_cache.get(cache_key)
        now = _time.time()
        if cached is not None:
            cached_score, cached_at = cached
            if now - cached_at < self._rerank_cache_ttl:
                return (document, cached_score)
            del self._rerank_cache[cache_key]

        # Only pass memory_age if recency boost is enabled
        effective_age = memory_age if self.config.enable_recency_boost else None
        _, score = await self._provider.score_document(
            query, document, fallback_score, effective_age
        )

        if len(self._rerank_cache) >= self._rerank_cache_max:
            oldest_key = next(iter(self._rerank_cache), None)
            if oldest_key is not None:
                del self._rerank_cache[oldest_key]
        self._rerank_cache[cache_key] = (score, now)

        return (document, score)

    async def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        timestamp_map: dict[str, str] | None = None,
    ) -> list[tuple[str, float]]:
        """Rerank candidates by LLM relevance scores.

        Scores each candidate document against the query using the LLM,
        filters by score threshold, and returns results sorted by
        relevance score descending.

        Args:
            query: Search query string.
            candidates: List of (memory, fusion_score) tuples from
                RRF fusion. The fusion_score is used as fallback if
                LLM scoring fails for a document.
            timestamp_map: Optional mapping from memory text to ISO 8601
                timestamp string. Used to provide temporal context to the
                LLM when enable_recency_boost is True.

        Returns:
            Reranked list of (memory, llm_score) tuples, filtered by
            score_threshold and sorted by score descending.
        """
        if not candidates:
            return []

        scored_results = await self._score_all_candidates(
            query, candidates, timestamp_map
        )
        scored_results = self._apply_post_processing(scored_results, timestamp_map)

        pre_filter_count = len(scored_results)
        scored_results = self._post_processor.filter_by_threshold(
            scored_results,
            threshold=self.config.score_threshold,
        )

        if self.logger:
            self.logger.debug(
                f"Reranking complete: {len(scored_results)}/{pre_filter_count} "
                f"passed threshold",
                extra={
                    "query": query[:50],
                    "input_count": len(candidates),
                    "output_count": len(scored_results),
                    "threshold": self.config.score_threshold,
                    "min_results": self.config.min_results,
                    "batch_normalize": self.config.batch_normalize,
                    "provider": self.config.provider,
                },
            )

        return scored_results

    async def _score_all_candidates(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        timestamp_map: dict[str, str] | None,
    ) -> list[tuple[str, float]]:
        """Score all candidates in parallel with concurrency-limited LLM calls."""
        semaphore = anyio.Semaphore(self.config.max_concurrency)
        results: list[tuple[str, float] | None] = [None] * len(candidates)

        async def score_with_semaphore(
            idx: int, doc: str, fallback: float, memory_age: str | None
        ) -> None:
            async with semaphore:
                results[idx] = await self._score_single(
                    query, doc, fallback, memory_age
                )

        async with anyio.create_task_group() as tg:
            for idx, (document, fusion_score) in enumerate(candidates):
                memory_age = None
                if (
                    self.config.enable_recency_boost
                    and timestamp_map
                    and document in timestamp_map
                ):
                    memory_age = format_memory_age(timestamp_map[document])
                tg.start_soon(
                    score_with_semaphore, idx, document, fusion_score, memory_age
                )

        return [r for r in results if r is not None]

    def _apply_post_processing(
        self,
        scored_results: list[tuple[str, float]],
        timestamp_map: dict[str, str] | None,
    ) -> list[tuple[str, float]]:
        """Apply normalization, recency decay, and sorting."""
        scored_results = self._post_processor.normalize(scored_results)

        decay_enabled = (
            self.config.enable_recency_boost
            and self.config.recency_decay_rate > 0
            and bool(timestamp_map)
        )
        scored_results = self._post_processor.apply_decay(
            scored_results,
            timestamp_map,
            self.config.recency_decay_rate,
            enabled=decay_enabled,
        )

        if not decay_enabled:
            scored_results.sort(key=lambda x: x[1], reverse=True)

        return scored_results
