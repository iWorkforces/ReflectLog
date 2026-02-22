"""Search pipeline with pluggable stages.

This module provides the SearchPipeline class that orchestrates the
4-step search process with pluggable stages for:
1. Backend execution (semantic, full-text)
2. Result fusion (RRF, concatenation)
3. Threshold filtering
4. Reranking (LLM, cross-encoder, or none)
"""

from dataclasses import dataclass
import logging
from typing import Protocol

import anyio

from reflectlog.application.config import Config
from reflectlog.core.logging import IStructuredLogger
from reflectlog.core.reranking import IReranker
from reflectlog.core.search import (
    ISearchBackend,
    ISearchResult,
)
from reflectlog.core.search import (
    SearchContext as CoreSearchContext,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchPipelineConfig:
    """Configuration for search pipeline stages."""

    enable_hybrid_search: bool
    enable_rrf_fusion: bool
    reranker_engine: str
    fusion_ranking_threshold: float
    search_limit: int


class IBackendExecutor(Protocol):
    """Protocol for search backend executor stages."""

    async def execute(
        self,
        context: CoreSearchContext,
    ) -> dict[str, list[ISearchResult]]:
        """Execute search across configured backends.

        Args:
            context: Search context with query and configuration.

        Returns:
            Dict mapping backend name to results.
        """
        ...


class IFusionStage(Protocol):
    """Protocol for result fusion stages."""

    def fuse(
        self,
        results: dict[str, list[ISearchResult]],
        limit: int,
    ) -> list[ISearchResult]:
        """Fuse results from multiple backends.

        Args:
            results: Dict mapping backend name to results.
            limit: Maximum results to return.

        Returns:
            Fused and ranked results.
        """
        ...


class IFilterStage(Protocol):
    """Protocol for result filtering stages."""

    def filter(
        self,
        results: list[ISearchResult],
        threshold: float,
    ) -> list[ISearchResult]:
        """Filter results by score threshold.

        Args:
            results: Results to filter.
            threshold: Minimum score to include.

        Returns:
            Filtered results.
        """
        ...


class DefaultBackendExecutor:
    """Default backend executor for semantic + full-text search."""

    def __init__(
        self,
        semantic_backend: ISearchBackend,
        fulltext_backend: ISearchBackend | None,
    ) -> None:
        self._semantic = semantic_backend
        self._fulltext = fulltext_backend

    async def execute(
        self,
        context: CoreSearchContext,
    ) -> dict[str, list[ISearchResult]]:
        """Execute parallel search across backends."""
        results: dict[str, list[ISearchResult]] = {}

        async with anyio.create_task_group() as tg:
            # Semantic search
            async def search_semantic() -> None:
                semantic_results = await self._semantic.search(
                    query=context.query,
                    project_id=context.project_id,
                    limit=context.overfetch_limit,
                )
                results["semantic"] = semantic_results

            tg.start_soon(search_semantic)

            # Full-text search (if enabled)
            fulltext_backend = self._fulltext
            if fulltext_backend is not None and context.enable_hybrid_search:

                async def search_fulltext() -> None:
                    fulltext_results = await fulltext_backend.search(
                        query=context.query,
                        project_id=context.project_id,
                        limit=context.overfetch_limit,
                    )
                    results["fulltext"] = fulltext_results

                tg.start_soon(search_fulltext)

        return results


class RRFFusionStage:
    """Reciprocal Rank Fusion implementation."""

    def __init__(self, k: float = 60.0) -> None:
        self._k = k

    def fuse(
        self,
        results: dict[str, list[ISearchResult]],
        limit: int,
    ) -> list[ISearchResult]:
        """Apply RRF to combine rankings."""
        if not results:
            return []

        # Build rankings for each backend
        rankings: dict[str, dict[str, float]] = {}
        for backend_name, backend_results in results.items():
            rankings[backend_name] = {}
            for rank, result in enumerate(backend_results, 1):
                rankings[backend_name][result.content] = 1.0 / (self._k + rank)

        # Aggregate scores across backends
        aggregated: dict[str, float] = {}
        for _backend_name, backend_rankings in rankings.items():
            for content, score in backend_rankings.items():
                if content not in aggregated:
                    aggregated[content] = 0.0
                aggregated[content] += score

        # Sort by aggregate score
        sorted_results = sorted(
            aggregated.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Return top results with original scores
        top_results: list[ISearchResult] = []
        seen: set[str] = set()
        for content, rrf_score in sorted_results:
            if len(top_results) >= limit:
                break
            if content in seen:
                continue
            seen.add(content)
            # Find original score from first backend
            original_score = 0.0
            for backend_results in results.values():
                for result in backend_results:
                    if result.content == content:
                        original_score = result.score
                        break
            top_results.append(
                ISearchResult(
                    content=content,
                    score=rrf_score,
                    memory_id="",
                    metadata={"original_score": original_score},
                )
            )

        return top_results


class ConcatenationFusionStage:
    """Simple concatenation fusion (no fusion, just combine)."""

    def fuse(
        self,
        results: dict[str, list[ISearchResult]],
        limit: int,
    ) -> list[ISearchResult]:
        """Concatenate results from backends."""
        if not results:
            return []

        # Combine all results
        all_results: list[ISearchResult] = []
        for backend_results in results.values():
            all_results.extend(backend_results)

        # Deduplicate by content
        seen: set[str] = set()
        unique_results: list[ISearchResult] = []
        for result in all_results:
            if result.content not in seen:
                seen.add(result.content)
                unique_results.append(result)

        # Sort by score
        unique_results.sort(key=lambda x: x.score, reverse=True)

        return unique_results[:limit]


class ThresholdFilterStage:
    """Filter results by score threshold."""

    def filter(
        self,
        results: list[ISearchResult],
        threshold: float,
    ) -> list[ISearchResult]:
        """Filter results below threshold."""
        return [r for r in results if r.score >= threshold]


class NoopRerankerStage:
    """No-op reranker (pass-through)."""

    async def rerank(
        self,
        query: str,
        results: list[ISearchResult],
    ) -> list[ISearchResult]:
        """Return results unchanged."""
        return results


class SearchPipeline:
    """Modular search pipeline with pluggable stages.

    This pipeline orchestrates the search process with configurable stages:
    1. Backend execution (semantic, full-text)
    2. Result fusion (RRF, concatenation)
    3. Threshold filtering
    4. Reranking (LLM, cross-encoder, or none)

    Example:
        pipeline = SearchPipeline(
            backend_executor=DefaultBackendExecutor(semantic, fulltext),
            fusion_stage=RRFFusionStage(k=60),
            filter_stage=ThresholdFilterStage(),
            reranker_stage=NoopRerankerStage(),
        )
        results = await pipeline.execute(context)
    """

    def __init__(
        self,
        backend_executor: IBackendExecutor,
        fusion_stage: IFusionStage,
        filter_stage: IFilterStage,
        reranker_stage: IReranker | None,
        config: SearchPipelineConfig,
        logger: IStructuredLogger | None,
    ):
        """Initialize search pipeline.

        Args:
            backend_executor: Stage for executing backend searches.
            fusion_stage: Stage for fusing results.
            filter_stage: Stage for filtering results.
            reranker_stage: Stage for reranking (optional).
            config: Pipeline configuration.
            logger: Structured logger.
        """
        if logger is None:
            raise ValueError("logger is required")

        self._backend_executor = backend_executor
        self._fusion_stage = fusion_stage
        self._filter_stage = filter_stage
        self._reranker = reranker_stage
        self._config = config
        self._logger: IStructuredLogger = logger

    async def execute(
        self,
        query: str,
        project_id: str,
        limit: int | None = None,
    ) -> list[str]:
        """Execute the full search pipeline.

        Args:
            query: Search query.
            project_id: Project identifier.
            limit: Maximum results (uses config default if None).

        Returns:
            List of matching memory strings.
        """
        if limit is None:
            limit = self._config.search_limit

        # Calculate overfetch based on index size
        overfetch_limit = self._calculate_overfetch(limit)

        # Create search context
        context = CoreSearchContext(
            query=query,
            limit=limit,
            overfetch_limit=overfetch_limit,
            enable_hybrid_search=self._config.enable_hybrid_search,
            enable_rrf_fusion=self._config.enable_rrf_fusion,
            reranker_engine=self._config.reranker_engine,
            project_id=project_id,
        )

        # Step 1: Execute backend searches
        backend_results = await self._backend_executor.execute(context)

        # Step 2: Fuse results
        fused_results = self._fusion_stage.fuse(backend_results, limit)

        # Step 3: Filter by threshold
        if self._config.enable_rrf_fusion:
            filtered_results = self._filter_stage.filter(
                fused_results,
                self._config.fusion_ranking_threshold,
            )
        else:
            filtered_results = fused_results

        # Step 4: Rerank if configured
        if self._reranker is not None:
            memories = [r.content for r in filtered_results]
            if memories:
                reranked = await self._reranker.rerank(query, memories)
                final_results = [
                    ISearchResult(
                        content=r.content,
                        score=r.score,
                        memory_id="",
                        metadata=r.metadata,
                    )
                    for r in reranked
                ]
            else:
                final_results = filtered_results
        else:
            final_results = filtered_results

        return [r.content for r in final_results[:limit]]

    def _calculate_overfetch(self, limit: int) -> int:
        """Calculate adaptive overfetch limit based on limit."""
        if limit <= 10:
            return min(limit * 3, 50)
        elif limit <= 50:
            return min(limit * 2, 100)
        else:
            return int(limit * 1.5)


def create_default_pipeline(
    semantic_backend: ISearchBackend,
    fulltext_backend: ISearchBackend | None,
    reranker: IReranker | None,
    config: Config,
    logger: IStructuredLogger | None,
) -> SearchPipeline:
    """Create a search pipeline with default configuration.

    Args:
        semantic_backend: Semantic search backend (USearch).
        fulltext_backend: Full-text search backend (Tantivy) or None.
        reranker: Reranker instance or None.
        config: Application configuration.
        logger: Structured logger.

    Returns:
        Configured SearchPipeline instance.
    """
    # Backend executor
    backend_executor = DefaultBackendExecutor(semantic_backend, fulltext_backend)

    # Fusion stage
    if config.enable_rrf_fusion:
        fusion_stage = RRFFusionStage(k=config.fusion_rrf_k)
    else:
        fusion_stage = ConcatenationFusionStage()

    # Filter stage
    filter_stage = ThresholdFilterStage()

    # Pipeline config
    pipeline_config = SearchPipelineConfig(
        enable_hybrid_search=config.enable_hybrid_search,
        enable_rrf_fusion=config.enable_rrf_fusion,
        reranker_engine=config.reranker_engine,
        fusion_ranking_threshold=config.fusion_ranking_threshold,
        search_limit=config.search_limit,
    )

    return SearchPipeline(
        backend_executor=backend_executor,
        fusion_stage=fusion_stage,
        filter_stage=filter_stage,
        reranker_stage=reranker,
        config=pipeline_config,
        logger=logger,
    )
