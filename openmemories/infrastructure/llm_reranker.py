"""LLM-based document reranker for search results."""

import json
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import anyio
from openai import AsyncOpenAI, DefaultAioHttpClient
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from openmemories.application.config import SCORING_PROMPT, Config


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
    """

    api_key: str
    base_url: str
    model: str
    score_threshold: float = 0.5
    max_concurrency: int = 5
    timeout: float = 30.0  # 30 seconds default for reranking requests
    min_results: int = 0  # Safety net: min results to return (0 = disabled)
    batch_normalize: bool = True  # Enable batch min-max normalization

    @classmethod
    def from_app_config(cls, config: Config) -> "LLMRerankerConfig":
        """Create LLMRerankerConfig from application Config.

        Args:
            config: Application configuration object.

        Returns:
            LLMRerankerConfig instance configured from app settings.
        """
        return cls(
            api_key=config.openrouter_api_key.get_secret_value(),
            base_url=config.openrouter_base_url,
            model=config.llm_model,
            score_threshold=config.search_score_threshold,
            max_concurrency=config.rerank_max_concurrency,
            min_results=config.reranker_min_results,
            batch_normalize=config.reranker_batch_normalize,
        )


class LLMReranker(BaseModel):
    """LLM-based document reranker using OpenRouter API.

    This class scores search result relevance using an LLM and reranks
    candidates based on their relevance scores. It supports parallel
    scoring with concurrency control and graceful fallback on errors.

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
    logger: Any = None

    _client: AsyncOpenAI | None = PrivateAttr(default=None)

    def __init__(self, **data: Any):
        """Initialize LLMReranker with AsyncOpenAI client."""
        super().__init__(**data)

        # Initialize async OpenAI client with HTTP/2 support and timeout
        self._client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            http_client=DefaultAioHttpClient(),
            timeout=self.config.timeout,
        )

    async def _call_llm_with_structured_output(
        self,
        prompt: str,
    ) -> Any:
        """Call LLM with structured output, falling back to json_object if unsupported.

        Args:
            prompt: The formatted scoring prompt.

        Returns:
            The API response object.

        Raises:
            Exception: If both structured output and fallback fail.
        """
        if self._client is None:
            raise RuntimeError("LLM reranker client is not initialized.")

        # Build structured output response format using Pydantic schema
        structured_response_format: dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": "relevance_score",
                "strict": True,
                "schema": RelevanceScore.model_json_schema(),
            },
        }

        try:
            # Try structured outputs first (guaranteed schema compliance)
            return await self._client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,  # Deterministic scoring
                max_tokens=50,  # Only need short JSON response
                response_format=structured_response_format,
            )
        except Exception as e:
            error_msg = str(e).lower()
            # Fallback to simple json_object mode for unsupported models
            if "json_schema" in error_msg or "structured" in error_msg:
                if self.logger:
                    self.logger.warning(
                        "Model doesn't support structured outputs, falling back to json_object",
                        extra={"model": self.config.model, "error": str(e)},
                    )
                return await self._client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=50,
                    response_format={"type": "json_object"},
                )
            raise

    async def _score_single(
        self,
        query: str,
        document: str,
        fallback_score: float,
    ) -> Tuple[str, float]:
        """Score a single document's relevance to the query.

        Uses OpenAI Structured Outputs with a Pydantic schema to guarantee
        valid JSON responses. Falls back to json_object mode if the model
        doesn't support structured outputs.

        Args:
            query: Search query string.
            document: Document text to score.
            fallback_score: Score to use if LLM call fails.

        Returns:
            Tuple of (document, score) where score is LLM relevance or fallback.
        """
        try:
            # Format the scoring prompt
            prompt = SCORING_PROMPT.format(query=query, document=document)

            # Call LLM with structured output (or fallback)
            response = await self._call_llm_with_structured_output(prompt)

            # Parse JSON response - schema guarantees valid JSON with score field
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Empty response from LLM")

            result = json.loads(content)
            score = float(result.get("score", fallback_score))

            # Clamp score to valid range (defensive, schema already constrains)
            score = max(0.0, min(1.0, score))

            if self.logger:
                self.logger.debug(
                    f"LLM scored document: {score:.2f}",
                    extra={
                        "query": query[:50],
                        "document_preview": document[:50],
                        "llm_score": score,
                    },
                )

            return (document, score)

        except json.JSONDecodeError as e:
            if self.logger:
                self.logger.warning(
                    f"Invalid JSON from LLM, using fallback score: {e}",
                    extra={"error": str(e), "fallback_score": fallback_score},
                )
            return (document, fallback_score)

        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"LLM scoring failed, using fallback score: {e}",
                    extra={"error": str(e), "fallback_score": fallback_score},
                )
            return (document, fallback_score)

    async def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """Rerank candidates by LLM relevance scores.

        Scores each candidate document against the query using the LLM,
        filters by score threshold, and returns results sorted by
        relevance score descending.

        Args:
            query: Search query string.
            candidates: List of (message, fusion_score) tuples from
                RRF fusion. The fusion_score is used as fallback if
                LLM scoring fails for a document.

        Returns:
            Reranked list of (message, llm_score) tuples, filtered by
            score_threshold and sorted by score descending.
        """
        if not candidates:
            return []

        # Use semaphore to limit concurrent LLM calls
        semaphore = anyio.Semaphore(self.config.max_concurrency)
        results: List[Optional[Tuple[str, float]]] = [None] * len(candidates)

        async def score_with_semaphore(idx: int, doc: str, fallback: float) -> None:
            """Score document with semaphore for concurrency control."""
            async with semaphore:
                results[idx] = await self._score_single(query, doc, fallback)

        # Score all candidates in parallel with concurrency limit
        async with anyio.create_task_group() as tg:
            for idx, (document, fusion_score) in enumerate(candidates):
                tg.start_soon(score_with_semaphore, idx, document, fusion_score)

        # Collect all scored results (filter out None results)
        scored_results: List[Tuple[str, float]] = []
        for result in results:
            if result is not None:
                scored_results.append(result)

        # Apply batch normalization if enabled
        if self.config.batch_normalize and scored_results:
            from openmemories.application.memory.reranking import (
                normalize_reranker_scores,
            )

            raw_scores = [s for _, s in scored_results]
            scored_results = normalize_reranker_scores(scored_results)
            if self.logger:
                self.logger.debug(
                    f"Batch normalization: enabled (raw range: {min(raw_scores):.4f}-{max(raw_scores):.4f} -> normalized: 0.0-1.0)",
                    extra={
                        "batch_normalize": True,
                        "raw_min": min(raw_scores),
                        "raw_max": max(raw_scores),
                    },
                )

        # Sort by score descending
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Apply threshold with optional safety net
        from openmemories.application.memory.reranking import (
            apply_threshold_with_safety_net,
        )

        pre_filter_count = len(scored_results)
        scored_results = apply_threshold_with_safety_net(
            scored_results,
            threshold=self.config.score_threshold,
            min_results=self.config.min_results,
        )

        if self.logger:
            self.logger.debug(
                f"Reranking complete: {len(scored_results)}/{pre_filter_count} passed threshold",
                extra={
                    "query": query[:50],
                    "input_count": len(candidates),
                    "output_count": len(scored_results),
                    "threshold": self.config.score_threshold,
                    "min_results": self.config.min_results,
                    "batch_normalize": self.config.batch_normalize,
                },
            )

        return scored_results
