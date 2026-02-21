'''Search pipeline strategies for hybrid semantic + full-text search.

This module extracts the search logic from MemoryManager into a separate
concern-focused module. It implements the 4-step search pipeline:
1. Parallel Search (USearch + Tantivy)
2. RRF Fusion or Concatenation
3. Fusion Threshold Filtering (when RRF enabled)
4. Reranking (LLM or CrossEncoder)
'''

from dataclasses import dataclass
import math
import time
from typing import Any, cast

from asyncer import asyncify, create_task_group

from ..config import Config
from ..exceptions import SearchError
from ..types import ISemanticSearchEngine
from ..utils import StructuredLogger, format_fusion_score_status
from ..utils.validation import truncate_memory
from .fusion import FusionEngine

# Constants for magic numbers (documented for maintainability)
MIN_OVERFETCH_LIMIT = 20  # Minimum docs to fetch for better fusion quality
TANTIVY_SCORE_DIVISOR = 10.0  # Tantivy BM25 scores typically 0-10+, normalize to 0-1
LOG_QUERY_TRUNCATE_LENGTH = 100


@dataclass
class SearchContext:
    '''Context object for search pipeline execution.

    Attributes:
        query: The search query string.
        limit: Maximum number of results to return.
        overfetch_limit: Number of candidates to fetch for better fusion quality.
        enable_hybrid_search: Whether Tantivy full-text search is enabled.
        enable_rrf_fusion: Whether RRF fusion is enabled (vs concatenation).
        reranker_engine: The reranker engine to use ("llm", "cross_encoder", or "none").
        project_id: Project identifier for logging.
    '''

    query: str
    limit: int
    overfetch_limit: int
    enable_hybrid_search: bool
    enable_rrf_fusion: bool
    reranker_engine: str
    project_id: str


@dataclass
class SearchResult:
    '''Result of search pipeline execution.

    Attributes:
        memories: List of memory strings.
        timestamp_map: Mapping from memory to created_at timestamp.
        semantic_results: Original semantic search results.
        tantivy_results: Original Tantivy search results.
    '''

    memories: list[str]
    timestamp_map: dict[str, str]
    semantic_results: list[tuple[str, float, str]]
    tantivy_results: list[tuple[str, float]]


class SearchPipeline:
    """Hybrid search pipeline orchestrator.

    Implements a 4-step search pipeline:
    1. Parallel semantic + full-text search
    2. RRF fusion or result concatenation
    3. Fusion threshold filtering (when RRF enabled)
    4. Reranking (LLM or CrossEncoder)

    Thread-safe: Uses MemoryManager's RLock for state changes.
    """

    def __init__(
        self,
        semantic_engine: ISemanticSearchEngine,
        tantivy_engine: Any,  # TantivyEngine | None
        fusion_engine: FusionEngine,
        config: Config,
        logger: StructuredLogger,
        memory_manager: Any,  # MemoryManager for lazy reranker fetching
    ):
        '''Initialize search pipeline.

        Args:
            semantic_engine: USearchEngine for semantic search.
            tantivy_engine: TantivyEngine for full-text search (optional).
            fusion_engine: RanxFusionEngine for result fusion.
            config: Application configuration.
            logger: Structured logger instance.
            memory_manager: MemoryManager instance for lazy reranker fetching.
        '''
        super().__init__()
        self._semantic_engine = semantic_engine
        self._tantivy_engine = tantivy_engine
        self._fusion_engine = fusion_engine
        self.config = config
        self.logger = logger
        self._memory_manager = memory_manager

    async def execute(self, context: SearchContext) -> SearchResult:
        '''Execute the full search pipeline.

        Args:
            context: Search context with query, limit, and configuration.

        Returns:
            SearchResult with memories and metadata.

        Raises:
            SearchError: If search operation fails.
        '''
        try:
            # Handle non-hybrid search (semantic only)
            if not context.enable_hybrid_search:
                return await self._execute_semantic_only(context)

            # Execute 4-step hybrid search pipeline
            return await self._execute_hybrid_search(context)

        except Exception as e:
            self.logger.error(
                'Search pipeline failed',
                extra={
                    'project_id': context.project_id,
                    'query': context.query,
                    'error': str(e),
                },
            )
            raise SearchError(f'Failed to execute search: {e}') from e

    async def _execute_semantic_only(self, context: SearchContext) -> SearchResult:
        '''Execute semantic-only search (hybrid disabled).'''
        self.logger.info(
            'SEARCH MODE: Semantic only (hybrid disabled)',
            extra={'mode': 'semantic'},
        )

        results = self._semantic_engine.search(
            query=context.query,
            project_id=context.project_id,
            limit=context.limit,
        )

        # Build timestamp map and memory list
        timestamp_map = {msg: created_at for msg, _, created_at in results}
        memories = [msg for msg, _, _ in results]

        return SearchResult(
            memories=memories,
            timestamp_map=timestamp_map,
            semantic_results=results,
            tantivy_results=[],
        )

    async def _execute_hybrid_search(self, context: SearchContext) -> SearchResult:
        '''Execute 4-step hybrid search pipeline.'''
        # Step 1: Parallel Search
        semantic_results, tantivy_results = await self._step1_parallel_search(context)

        # Build timestamp map from semantic results
        timestamp_map: dict[str, str] = {
            msg: created_at for msg, _, created_at in semantic_results
        }

        # Log results
        self._log_search_results(context, semantic_results, tantivy_results)

        # Step 2: RRF Fusion or Concatenation
        hybrid_results = await self._step2_fusion_or_concatenate(
            context, semantic_results, tantivy_results
        )

        # Step 3: Fusion threshold filtering (only when RRF enabled)
        if context.enable_rrf_fusion:
            hybrid_results = await self._step3_fusion_threshold(context, hybrid_results)
            # Handle case where all results were filtered out
            if not hybrid_results:
                return SearchResult(
                    memories=[],
                    timestamp_map={},
                    semantic_results=semantic_results,
                    tantivy_results=tantivy_results,
                )

        # Step 4: Reranking (or skip if 0-1 results)
        rerank_step_num = 4 if context.enable_rrf_fusion else 3
        if len(hybrid_results) <= 1:
            self._log_skip_reranking(len(hybrid_results), rerank_step_num)
        else:
            hybrid_results = await self._step4_reranking(
                context, hybrid_results, timestamp_map, rerank_step_num
            )

        # Extract final memories
        memories = [msg for msg, _ in hybrid_results[: context.limit]]

        # Final summary
        self.logger.info(
            f'SEARCH COMPLETE: {len(memories)} result(s) returned',
            extra={
                'project_id': context.project_id,
                'query': context.query,
                'result_count': len(memories),
                'top_score': hybrid_results[0][1] if hybrid_results else 0.0,
            },
        )

        return SearchResult(
            memories=memories,
            timestamp_map=timestamp_map,
            semantic_results=semantic_results,
            tantivy_results=tantivy_results,
        )

    async def _step1_parallel_search(
        self, context: SearchContext
    ) -> tuple[list[tuple[str, float, str]], list[tuple[str, float]]]:
        '''Step 1: Execute parallel search on both engines.'''
        self.logger.info(
            'STEP 1: Executing parallel search engines...',
            extra={'step': 'parallel_search'},
        )

        # Pre-initialize engines to prevent race conditions
        self._semantic_engine.ensure_initialized()
        if self._tantivy_engine is not None:
            self._tantivy_engine.ensure_initialized()

        # Run both searches in parallel
        soon_semantic = None
        soon_tantivy = None
        async with create_task_group() as tg:
            soon_semantic = tg.soonify(asyncify(self._search_semantic))(
                context.query, context.overfetch_limit, context.project_id
            )
            soon_tantivy = tg.soonify(asyncify(self._search_tantivy))(
                context.query, context.overfetch_limit, context.project_id
            )

        assert soon_semantic is not None
        assert soon_tantivy is not None

        semantic_results = soon_semantic.value or []
        tantivy_results = soon_tantivy.value or []

        self.logger.info(
            f'Both engines completed (semantic: {len(semantic_results)}, tantivy: {len(tantivy_results)})',
            extra={
                'project_id': context.project_id,
                'query': context.query[:LOG_QUERY_TRUNCATE_LENGTH],
                'semantic_count': len(semantic_results),
                'tantivy_count': len(tantivy_results),
            },
        )

        return semantic_results, tantivy_results

    def _search_semantic(
        self, query: str, limit: int, project_id: str
    ) -> list[tuple[str, float, str]]:
        '''Execute semantic search on USearchEngine.

        Falls back to empty list on error, relying on Tantivy results.
        '''
        try:
            results = self._semantic_engine.search(
                query=query,
                project_id=project_id,
                limit=limit,
            )
            return results
        except Exception as e:
            # Enhanced diagnostic logging for semantic search fallback
            self.logger.warning(
                'Semantic search failed - falling back to Tantivy full-text only',
                extra={
                    'project_id': project_id,
                    'query': query[:LOG_QUERY_TRUNCATE_LENGTH],
                    'error_type': type(e).__name__,
                    'error': str(e),
                    'fallback_behavior': 'empty_semantic_results',
                    'note': 'Search will continue with Tantivy results only',
                },
            )
            return []

    def _search_tantivy(
        self, query: str, limit: int, project_id: str
    ) -> list[tuple[str, float]]:
        '''Execute full-text search on Tantivy engine.'''
        if self._tantivy_engine is None:
            return []
        return self._tantivy_engine.search(query, project_id, limit)

    async def _step2_fusion_or_concatenate(
        self,
        context: SearchContext,
        semantic_results: list[tuple[str, float, str]],
        tantivy_results: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        '''Step 2: Combine results using RRF fusion or concatenation.'''
        # Convert semantic_results from 3-tuples to 2-tuples for fusion
        semantic_results_2tuple = [(msg, score) for msg, score, _ in semantic_results]

        if context.enable_rrf_fusion:
            return await self._fuse_rrf(
                context, semantic_results_2tuple, tantivy_results
            )
        else:
            return self._concatenate_results(
                semantic_results_2tuple, tantivy_results, context.overfetch_limit
            )

    async def _fuse_rrf(
        self,
        context: SearchContext,
        semantic_results: list[tuple[str, float]],
        tantivy_results: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        '''Fuse results using RRF algorithm.'''
        self.logger.info(
            'STEP 2: RRF Fusion (combining results)...',
            extra={'step': 'fusion', 'mode': 'rrf'},
        )

        hybrid_results = self._fusion_engine.fuse(semantic_results, tantivy_results)

        if hybrid_results:
            top_rrf = hybrid_results[0][1] if hybrid_results else 0.0
            self.logger.info(
                f'Combined {len(hybrid_results)} unique result(s) using {self._fusion_engine.method.upper()} algorithm',
                extra={
                    'project_id': context.project_id,
                    'query': context.query[:LOG_QUERY_TRUNCATE_LENGTH],
                    'engine': 'fusion',
                    'result_count': len(hybrid_results),
                    'method': self._fusion_engine.method,
                    'top_rrf_score': top_rrf,
                },
            )

        return hybrid_results

    def _concatenate_results(
        self,
        semantic_results: list[tuple[str, float]],
        tantivy_results: list[tuple[str, float]],
        limit: int,
    ) -> list[tuple[str, float]]:
        '''Concatenate semantic + tantivy results without RRF fusion.'''
        self.logger.info(
            'STEP 2: Concatenate results (RRF fusion disabled)...',
            extra={'step': 'concatenate', 'mode': 'concatenate'},
        )

        seen_memories: set[str] = set()
        combined: list[tuple[str, float]] = []

        # Add semantic results first (higher priority)
        for msg, score in semantic_results:
            if len(combined) >= limit:
                break
            if msg not in seen_memories:
                seen_memories.add(msg)
                combined.append((msg, score))

        semantic_count = len(combined)

        # Add tantivy results (skip duplicates)
        for msg, score in tantivy_results:
            if len(combined) >= limit:
                break
            if msg not in seen_memories:
                seen_memories.add(msg)
                combined.append((msg, score))

        tantivy_added = len(combined) - semantic_count
        duplicates_skipped = len(tantivy_results) - tantivy_added

        self.logger.info(
            f'Combined {len(combined)} result(s): {semantic_count} semantic + {tantivy_added} tantivy ({duplicates_skipped} duplicates skipped)',
            extra={
                'semantic_count': semantic_count,
                'tantivy_added': tantivy_added,
                'duplicates_skipped': duplicates_skipped,
                'total_count': len(combined),
            },
        )

        return combined

    async def _step3_fusion_threshold(
        self, context: SearchContext, results: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        '''Step 3: Filter by fusion threshold.'''
        self.logger.info(
            f'STEP 3: Filtering (threshold >= {self.config.fusion_ranking_threshold})...',
            extra={
                'step': 'filtering',
                'threshold': self.config.fusion_ranking_threshold,
            },
        )

        return self._filter_by_fusion_threshold(
            results, self.config.fusion_ranking_threshold, context.query
        )

    def _filter_by_fusion_threshold(
        self, results: list[tuple[str, float]], threshold: float, query: str
    ) -> list[tuple[str, float]]:
        '''Filter fusion results by fusion score threshold.'''
        if not results:
            return results

        # Apply threshold filter
        filtered = [(msg, score) for msg, score in results if score >= threshold]
        filtered_out = len(results) - len(filtered)

        # Log summary
        self.logger.info(
            f'Kept {len(filtered)}/{len(results)} result(s), filtered {filtered_out}',
            extra={
                'fusion_threshold': threshold,
                'kept_count': len(filtered),
                'filtered_count': filtered_out,
            },
        )

        # Show filtered results with status
        for idx, (memory, score) in enumerate(results[: min(5, len(results))], 1):
            status, interpretation = format_fusion_score_status(score, threshold)
            preview = truncate_memory(memory, max_length=50)
            status_indicator = '[KEEP]' if score >= threshold else '[FILTER]'
            self.logger.info(
                f'[{idx}] {status_indicator} score={score:.4f} ({interpretation}) -> {preview}',
                extra={'fusion_score': score, 'fusion_status': status},
            )

        return filtered

    def _get_reranker(self):
        """Get the appropriate reranker (lazy loading via memory_manager).

        Returns:
            Tuple of (reranker_type, reranker_instance) where type is 'llm', 'cross_encoder', or None.
        """
        if self._memory_manager is None:
            return None, None

        reranker = self._memory_manager.get_reranker()
        if self.config.reranker_engine == 'llm' and reranker is not None:
            return 'llm', reranker
        elif self.config.reranker_engine == 'cross_encoder' and reranker is not None:
            return 'cross_encoder', reranker
        return None, None

    async def _step4_reranking(
        self,
        context: SearchContext,
        results: list[tuple[str, float]],
        timestamp_map: dict[str, str],
        step_num: int,
    ) -> list[tuple[str, float]]:
        '''Step 4: Rerank results using LLM or CrossEncoder.'''
        reranker_type, reranker = self._get_reranker()

        if reranker_type == 'llm':
            return await self._rerank_llm(
                context, results, timestamp_map, step_num, reranker
            )
        elif reranker_type == 'cross_encoder':
            return await self._rerank_cross_encoder(
                context, results, step_num, reranker
            )
        else:
            # No reranking configured
            return results

    async def _rerank_llm(
        self,
        context: SearchContext,
        results: list[tuple[str, float]],
        timestamp_map: dict[str, str],
        step_num: int,
        llm_reranker: Any | None = None,  # Optional parameter to use provided reranker
    ) -> list[tuple[str, float]]:
        '''Rerank using LLM.'''
        # Use provided reranker or fetch via _get_reranker
        if llm_reranker is None:
            _, llm_reranker = self._get_reranker()
            if llm_reranker is None:
                return results

        self.logger.info(
            f'STEP {step_num}: LLM Reranking ({len(results)} candidates)...',
            extra={
                'step': 'reranking',
                'step_num': step_num,
                'engine': 'llm',
                'candidate_count': len(results),
            },
        )

        rerank_start = time.time()
        pre_rerank_count = len(results)

        reranked_results = await llm_reranker.rerank(
            context.query, results, timestamp_map
        )
        results = cast(list[tuple[str, float]], reranked_results)

        rerank_duration = (time.time() - rerank_start) * 1000

        self.logger.info(
            f'Kept {len(results)}/{pre_rerank_count} result(s) after LLM scoring',
            extra={
                'kept_count': len(results),
                'filtered_count': pre_rerank_count - len(results),
                'threshold': self.config.search_score_threshold,
            },
        )
        self.logger.info(
            f'LLM reranking completed in {rerank_duration:.0f}ms',
            extra={'rerank_duration_ms': rerank_duration},
        )

        return results

    async def _rerank_cross_encoder(
        self,
        context: SearchContext,
        results: list[tuple[str, float]],
        step_num: int,
        cross_encoder_reranker: Any
        | None = None,  # Optional parameter to use provided reranker
    ) -> list[tuple[str, float]]:
        '''Rerank using CrossEncoder.'''
        # Use provided reranker or fetch via _get_reranker
        if cross_encoder_reranker is None:
            _, cross_encoder_reranker = self._get_reranker()
            if cross_encoder_reranker is None:
                return results

        self.logger.info(
            f'STEP {step_num}: CrossEncoder Reranking ({len(results)} candidates)...',
            extra={
                'step': 'reranking',
                'step_num': step_num,
                'engine': 'cross_encoder',
                'candidate_count': len(results),
            },
        )

        rerank_start = time.time()
        pre_rerank_count = len(results)

        reranked_results = await cross_encoder_reranker.rerank_async(
            context.query, results
        )
        results = cast(list[tuple[str, float]], reranked_results)

        rerank_duration = (time.time() - rerank_start) * 1000

        self.logger.info(
            f'Kept {len(results)}/{pre_rerank_count} result(s) after CrossEncoder scoring',
            extra={
                'kept_count': len(results),
                'filtered_count': pre_rerank_count - len(results),
                'model': self.config.cross_encoder_model,
            },
        )
        self.logger.info(
            f'CrossEncoder reranking completed in {rerank_duration:.0f}ms',
            extra={'rerank_duration_ms': rerank_duration},
        )

        return results

    def _log_skip_reranking(self, result_count: int, step_num: int) -> None:
        '''Log when reranking is skipped due to 0-1 results.'''
        if result_count == 1:
            self.logger.info(
                f'STEP {step_num}: Reranking skipped (single result - reranking unnecessary)',
                extra={
                    'step': 'reranking_skip',
                    'step_num': step_num,
                    'result_count': 1,
                    'reason': 'single result - reranking unnecessary',
                },
            )

    def _log_search_results(
        self,
        context: SearchContext,
        semantic_results: list[tuple[str, float, str]],
        tantivy_results: list[tuple[str, float]],
    ) -> None:
        '''Log search results from both engines.'''
        self.logger.info('─' * 50, extra={'section': 'search_engines'})

        # Log USearch results
        self.logger.info('USEARCH SEMANTIC ENGINE:', extra={'engine': 'usearch'})
        if semantic_results:
            top_score = semantic_results[0][1] if semantic_results else 0.0
            self.logger.info(
                f'Found {len(semantic_results)} result(s), best score: {top_score:.4f}',
                extra={'result_count': len(semantic_results), 'top_score': top_score},
            )
            for idx, (memory, score, _) in enumerate(
                semantic_results[: min(3, len(semantic_results))], 1
            ):
                preview = truncate_memory(memory, max_length=60)
                self.logger.info(
                    f'[{idx}] score={score:.4f} → {preview}',
                    extra={'result_index': idx, 'score': score},
                )
        else:
            self.logger.info('No results found', extra={'result_count': 0})

        # Log Tantivy results
        self.logger.info('TANTIVY FULL-TEXT ENGINE:', extra={'engine': 'tantivy'})
        if tantivy_results:
            top_score = tantivy_results[0][1] if tantivy_results else 0.0
            self.logger.info(
                f'Found {len(tantivy_results)} result(s), best BM25 score: {top_score:.4f}',
                extra={'result_count': len(tantivy_results), 'top_score': top_score},
            )
            for idx, (memory, score) in enumerate(
                tantivy_results[: min(3, len(tantivy_results))], 1
            ):
                preview = truncate_memory(memory, max_length=60)
                self.logger.info(
                    f'[{idx}] score={score:.4f} → {preview}',
                    extra={'result_index': idx, 'score': score},
                )
        else:
            self.logger.info('No results found', extra={'result_count': 0})

        self.logger.info('─' * 50, extra={'section': 'fusion'})


def calculate_adaptive_overfetch(limit: int, index_size: int, config: Config) -> int:
    '''Calculate adaptive overfetch limit based on index size.

    For small indexes, we use a higher multiplier to ensure diversity.
    For large indexes, we use a lower multiplier since there are enough
    documents for good fusion quality.

    The multiplier scales logarithmically:
    - Index size <= 100: max multiplier (3.0x default)
    - Index size >= 10,000: min multiplier (1.5x default)
    - Between: logarithmic interpolation

    Args:
        limit: The base search limit.
        index_size: Current size of the search index.
        config: Application configuration.

    Returns:
        Calculated overfetch limit (minimum MIN_OVERFETCH_LIMIT).
    '''
    # If adaptive is disabled or index is empty, use static multiplier
    if not config.overfetch_adaptive or index_size == 0:
        return max(limit * config.overfetch_multiplier, MIN_OVERFETCH_LIMIT)

    # Bounds for index size (log scale interpolation)
    small_index = 100  # At or below this: use max multiplier
    large_index = 10000  # At or above this: use min multiplier

    min_mult = config.overfetch_min_multiplier
    max_mult = config.overfetch_max_multiplier

    if index_size <= small_index:
        multiplier = max_mult
    elif index_size >= large_index:
        multiplier = min_mult
    else:
        # Logarithmic interpolation between bounds
        log_small = math.log(small_index)
        log_large = math.log(large_index)
        log_current = math.log(index_size)

        # Normalize to [0, 1] range
        t = (log_current - log_small) / (log_large - log_small)

        # Linear interpolation from max_mult to min_mult
        multiplier = max_mult - t * (max_mult - min_mult)

    overfetch_limit = int(limit * multiplier)
    return max(overfetch_limit, MIN_OVERFETCH_LIMIT)
