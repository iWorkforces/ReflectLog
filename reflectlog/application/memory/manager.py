"""Memory management wrapper for hybrid search integration.

This module provides the MemoryManager class that orchestrates all memory
operations. It combines semantic vector search (USearch) with full-text
search (Tantivy) using Reciprocal Rank Fusion (RRF) for intelligent
result ranking.

Protocol Compliance:
    MemoryManager implements the IMemoryManager protocol from core/memory.py.
    This enables:
    - Dependency injection with mock implementations for testing
    - Runtime component substitution (different backends)
    - Type-safe interaction with the memory system

Example:
    from reflectlog.core.memory import IMemoryManager
    from reflectlog.core.config_adapters import ConfigAdapter

    # Protocol-based usage
    manager: IMemoryManager = MemoryManager(config, logger)

    # With dependency injection
    mock_backend = create_mock_backend()
    manager = MemoryManager(
        config=config,
        logger=logger,
        semantic_backend=mock_backend,
    )
"""

import threading
import time
from typing import Any, final

from asyncer import asyncify

from reflectlog.application.constants import LOG_ADD_MEMORY_PREVIEW_LIMIT
from reflectlog.core.exceptions import (
    ConfigurationError,
    InconsistentStateError,
    SearchError,
    StorageError,
)
from reflectlog.infrastructure.cached_embeddings import CachedEmbeddings
from reflectlog.infrastructure.cross_encoder_reranker import (
    CrossEncoderConfig,
    CrossEncoderReranker,
)
from reflectlog.infrastructure.qwen3_embedding import LangchainQwenEmbeddings
from reflectlog.infrastructure.smart_replacer import SmartReplacer, SmartReplacerConfig
from reflectlog.infrastructure.tantivy_engine import TantivyConfig, TantivyEngine
from reflectlog.infrastructure.usearch_engine import USearchConfig, USearchEngine

from ...core.config_adapters import ConfigAdapter
from ...core.logging import IStructuredLogger
from ...core.types import ISemanticSearchEngine, ReplacementTransition
from ..config.settings import Config
from ..utils.validation import (
    truncate_memory,
)
from .add_phases import (
    AddPipeline,
    AddResult,
    DuplicateDetectionPhase,
    SmartReplacementPhase,
    StoragePhase,
)
from .fusion import create_fusion_engine
from .fusion.base import FusionEngine
from .match_utils import has_exact_match
from .replacement_recovery import (
    reconcile_pending_replacements as apply_pending_replacements,
)
from .search_strategies import (
    SearchContext,
    SearchPipeline,
    calculate_adaptive_overfetch,
)


@final
class MemoryManager:
    """Manages memory storage and retrieval using USearch with SQLite backend."""

    def __init__(self, config: Config, logger: IStructuredLogger | None):
        """Initialize Hybrid MemoryManager with USearch semantic
        & Tantivy full-text engines.

        Args:
            config: Application configuration.
            logger: Structured logger instance.
        """
        super().__init__()
        if logger is None:
            raise ValueError("logger is required")

        self.config = config
        self.logger: IStructuredLogger = logger
        self.workspace_id = config.workspace_id

        self._init_locks()
        self.is_hybrid_search = self.config.enable_hybrid_search
        self._init_semantic_engine()
        self._init_search_engine()
        self._init_fusion_engine()
        self._init_rerankers()
        self._init_smart_replacer()
        self._init_pipelines()
        self._log_configuration()
        _ = self.reconcile_pending_replacements()

        if config.eager_initialization:
            self._eager_initialize_engines()

    def _init_locks(self) -> None:
        """Create all threading locks and startup metrics placeholder.

        Lock hierarchy (outer to inner) — ALWAYS follow this order
        to prevent deadlocks:
          1. _write_lock — for all write operations across async/sync boundary
          2. _lock — for read operations and inner critical sections

        Usage pattern:
          - Read operations: use _lock only
          - Write operations: acquire _write_lock first, then _lock if needed
          - Never acquire _lock before _write_lock (risk of deadlock)
        """
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self.startup_metrics: dict[str, float] | None = None
        self._reranker_lock = threading.RLock()
        self._smart_replacer_lock = threading.RLock()

    def _init_semantic_engine(self) -> None:
        """Create USearch semantic engine with optional embedding cache."""
        config = self.config
        usearch_config = USearchConfig.from_config(ConfigAdapter(config))
        base_embedder = LangchainQwenEmbeddings(
            config={
                "model": config.embedding_model,
                "embedding_dims": config.qwen_embedding_dims
                if config.embedder_provider == "langchain"
                else config.embedding_dims,
                "api_key": config.openrouter_api_key.get_secret_value(),
                "openai_base_url": config.openrouter_base_url,
                "batch_size": config.embedding_batch_size,
                "max_concurrent_batches": config.embedding_max_concurrent_batches,
            }
        )

        # Wrap embedder with LRU cache for query embeddings (reduces API calls)
        if config.embedding_cache_enabled:
            embedder = CachedEmbeddings(
                embedder=base_embedder,
                cache_size=config.embedding_cache_size,
                enabled=True,
                logger=self.logger,
            )
        else:
            embedder = base_embedder

        self._semantic_engine: ISemanticSearchEngine = USearchEngine(
            usearch_config, embedder=embedder, logger=self.logger
        )

    def _init_search_engine(self) -> None:
        """Create Tantivy full-text engine when hybrid search is enabled."""
        self._tantivy_engine: TantivyEngine | None = None
        if self.is_hybrid_search:
            tantivy_config = TantivyConfig(
                workspace_id=self.workspace_id,
                index_path=self.config.tantivy_index_path_template.format(
                    workspace_id=self.workspace_id
                ).lower(),
                normalize_scores=self.config.tantivy_normalize_scores,
                soft_delete_enabled=self.config.tantivy_soft_delete_enabled,
                compaction_threshold_ratio=self.config.tantivy_compaction_threshold_ratio,
                compaction_max_tombstones=self.config.tantivy_compaction_max_tombstones,
                tombstone_ttl_days=self.config.tantivy_tombstone_ttl_days,
            )
            self._tantivy_engine = TantivyEngine(tantivy_config, logger=self.logger)

    def _init_fusion_engine(self) -> None:
        """Create fusion engine for hybrid ranking."""
        fusion_weights = self.config.fusion_weights
        self._fusion_engine: FusionEngine = create_fusion_engine(
            method=self.config.fusion_method,
            normalization=self.config.fusion_normalization,
            rrf_k=self.config.fusion_rrf_k,
            weights=fusion_weights if isinstance(fusion_weights, list) else None,
            logger=self.logger,
        )

    def _init_rerankers(self) -> None:
        """Set up reranker references for lazy initialization via properties.

        The cross-encoder is created on first search to avoid startup overhead.
        """
        self._cross_encoder_reranker: CrossEncoderReranker | None = None

        config = self.config
        if config.reranker_engine == "cross_encoder":
            self.logger.info(
                f"CrossEncoder reranker configured (lazy init) "
                f"[model={config.cross_encoder_model}]",
                extra={
                    "reranker_engine": "cross_encoder",
                    "model": config.cross_encoder_model,
                    "device": config.cross_encoder_device,
                },
            )
        elif config.reranker_engine == "none":
            self.logger.info(
                "Reranking disabled (RERANKER_ENGINE=none)",
                extra={"reranker_engine": "none"},
            )
        else:
            raise ConfigurationError(
                f"Invalid RERANKER_ENGINE: '{config.reranker_engine}'. "
                "Valid options: cross_encoder, none"
            )

    def _init_smart_replacer(self) -> None:
        """Set up smart replacer reference for lazy initialization via property.

        SmartReplacer is created on first add operation
        to avoid startup overhead.
        """
        self._smart_replacer: SmartReplacer | None = None

        config = self.config
        if config.enable_smart_replace:
            self.logger.info(
                f"SmartReplacer configured (lazy init) [model={config.llm_model}, "
                f"threshold={config.smart_replace_threshold}]",
                extra={
                    "smart_replacer": "enabled",
                    "model": config.llm_model,
                    "threshold": config.smart_replace_threshold,
                },
            )
        else:
            self.logger.info(
                "Smart memory replacement disabled (ENABLE_SMART_REPLACE=false)",
                extra={"smart_replacer": "disabled"},
            )

    def _init_pipelines(self) -> None:
        """Create SearchPipeline and AddPipeline with their phase components."""
        self._search_pipeline = SearchPipeline(
            semantic_engine=self._semantic_engine,
            tantivy_engine=self._tantivy_engine,
            fusion_engine=self._fusion_engine,
            config=self.config,
            logger=self.logger,
            memory_manager=self,  # Pass self for lazy reranker fetching
        )

        self._duplicate_detection_phase = DuplicateDetectionPhase(
            semantic_engine=self._semantic_engine,
            tantivy_engine=self._tantivy_engine,
            config=self.config,
            logger=self.logger,
        )
        self._smart_replacement_phase = SmartReplacementPhase(
            semantic_engine=self._semantic_engine,
            config=self.config,
            logger=self.logger,
            memory_manager=self,  # Pass self for lazy SmartReplacer fetching
        )
        self._storage_phase = StoragePhase(
            semantic_engine=self._semantic_engine,
            tantivy_engine=self._tantivy_engine,
            config=self.config,
            logger=self.logger,
            write_lock=self._write_lock,
            lock=self._lock,
        )
        self._add_pipeline = AddPipeline(
            duplicate_detection_phase=self._duplicate_detection_phase,
            smart_replacement_phase=self._smart_replacement_phase,
            storage_phase=self._storage_phase,
            config=self.config,
            logger=self.logger,
        )

    def reconcile_pending_replacements(self) -> int:
        """Finish replacements interrupted by a previous process stop.

        Called at startup and at the start of the next add persist.
        Not invoked by health_check. Acquires ``_write_lock`` then ``_lock``.
        """
        return apply_pending_replacements(
            semantic_engine=self._semantic_engine,
            tantivy_engine=self._tantivy_engine,
            write_lock=self._write_lock,
            lock=self._lock,
            logger=self.logger,
        )

    def pending_replacement_count(self) -> int:
        """Return how many replacement transitions are still pending."""
        store = getattr(self._semantic_engine, "memory_store", None)
        list_pending = getattr(store, "list_pending_transitions", None)
        if store is None or not callable(list_pending):
            return 0
        pending = list_pending()
        if not isinstance(pending, list):
            return 0
        return sum(
            1 for row in pending if isinstance(row, ReplacementTransition)
        )

    def _log_configuration(self) -> None:
        """Log the final configuration state after initialization."""
        self.logger.info(
            f"Initialized Hybrid MemoryManager [workspace_id={self.workspace_id}, "
            f"semantic_backend=usearch, "
            f"hybrid_search={self.is_hybrid_search}, "
            f"embedding_model={self.config.embedding_model}, "
        )
        self.logger.info(
            f"tantivy_index={self.config.tantivy_index_path_template.format(workspace_id=self.workspace_id)}, "
            f"embedder={self.config.embedder_provider}]",
        )

    def _eager_initialize_engines(self) -> None:
        """Pre-warm storage engines for faster first operation.

        This method forces initialization of lazy-loaded resources based on
        granular configuration settings:
        - USearch index and libSQL connection (when eager_initialize_search_engines)
        - Tantivy index, writer, and searcher (when eager_initialize_search_engines)
        - Reranker (when eager_initialize_reranker)
        - SmartReplacer (when eager_initialize_smart_replacer)

        Granular settings take precedence over the general EAGER_INITIALIZATION flag.
        If granular settings are None (not explicitly configured), falls back to
        EAGER_INITIALIZATION for search engines only (rerankers are lazy by default).

        Useful for reducing first-request latency in production deployments.
        Called during __init__ when enabled.
        """
        start_time = time.time()

        # Determine which components to initialize
        # Priority: granular setting > general eager initialization
        # > default (search only)
        should_init_search = (
            self.config.eager_initialize_search_engines
            if self.config.eager_initialize_search_engines is not None
            else self.config.eager_initialization
        )
        should_init_reranker = (
            self.config.eager_initialize_reranker
            if self.config.eager_initialize_reranker is not None
            else False  # Rerankers are lazy by default
        )
        should_init_smart_replacer = (
            self.config.eager_initialize_smart_replacer
            if self.config.eager_initialize_smart_replacer is not None
            else False  # SmartReplacer is lazy by default
        )

        engines_initialized: list[str] = []

        # Pre-warm search engines if configured
        if should_init_search:
            self.logger.info(
                "Starting eager search engine initialization...",
                extra={"workspace_id": self.workspace_id},
            )

            # Pre-warm USearch semantic engine
            self._semantic_engine.ensure_initialized()
            engines_initialized.append("usearch")

            # Pre-warm Tantivy full-text engine
            if self._tantivy_engine is not None:
                self._tantivy_engine.ensure_initialized()
                engines_initialized.append("tantivy")

        # Pre-warm reranker if explicitly configured (lazy by default)
        if should_init_reranker:
            # Validate that reranker engine type is supported
            if self.config.reranker_engine != "cross_encoder":
                raise ValueError(
                    f"Invalid reranker_engine for eager initialization: "
                    f"{self.config.reranker_engine!r}. "
                    f"Must be 'cross_encoder', or set "
                    f"eager_initialize_reranker=false for lazy loading."
                )

            self.logger.info(
                "Starting eager reranker initialization...",
                extra={
                    "workspace_id": self.workspace_id,
                    "reranker_engine": self.config.reranker_engine,
                },
            )
            reranker = self.get_reranker()
            if reranker is not None:
                model = getattr(reranker, "model", None)
                if model is not None:
                    _ = model
                engines_initialized.append(f"reranker_{self.config.reranker_engine}")
            else:
                self.logger.warning(
                    "Eager reranker initialization requested but no reranker configured",
                    extra={
                        "workspace_id": self.workspace_id,
                        "reranker_engine": self.config.reranker_engine,
                    },
                )

        # Pre-warm SmartReplacer if explicitly configured (lazy by default)
        if should_init_smart_replacer:
            # Validate that smart replacement is enabled
            if not self.config.enable_smart_replace:
                raise ValueError(
                    "Eager SmartReplacer initialization requested but "
                    "enable_smart_replace is disabled. Set enable_smart_replace=true "
                    "or set eager_initialize_smart_replacer=false for lazy loading."
                )

            self.logger.info(
                "Starting eager SmartReplacer initialization...",
                extra={"workspace_id": self.workspace_id},
            )
            _ = self.smart_replacer
            engines_initialized.append("smart_replacer")

        elapsed_ms = (time.time() - start_time) * 1000

        if engines_initialized:
            self.logger.info(
                f"Eager initialization complete [{elapsed_ms:.1f}ms]",
                extra={
                    "workspace_id": self.workspace_id,
                    "elapsed_ms": elapsed_ms,
                    "engines_initialized": engines_initialized,
                },
            )
        else:
            self.logger.info(
                "Eager initialization skipped (all components set to lazy loading)",
                extra={"workspace_id": self.workspace_id},
            )

    @property
    def cross_encoder_reranker(self) -> CrossEncoderReranker | None:
        """Get CrossEncoder reranker (lazy initialization with thread-safety).

        Returns:
            CrossEncoderReranker instance if configured, None otherwise.

        Raises:
            RuntimeError: If reranker_engine is 'cross_encoder' but initialization fails.
        """
        # Fast path: already initialized or not configured
        if (
            self._cross_encoder_reranker is not None
            or self.config.reranker_engine != "cross_encoder"
        ):
            return self._cross_encoder_reranker

        # Slow path: need to initialize with lock
        with self._reranker_lock:
            # Double-check after acquiring lock
            if (
                self._cross_encoder_reranker is not None
                or self.config.reranker_engine != "cross_encoder"
            ):
                return self._cross_encoder_reranker

            # Initialize CrossEncoder reranker
            ce_config = CrossEncoderConfig.from_config(ConfigAdapter(self.config))
            self._cross_encoder_reranker = CrossEncoderReranker(
                config=ce_config, logger=self.logger
            )
            self.logger.info(
                f"Lazy initialized CrossEncoder reranker [model={self.config.cross_encoder_model}]",
                extra={
                    "reranker_engine": "cross_encoder",
                    "model": self.config.cross_encoder_model,
                    "device": self.config.cross_encoder_device,
                },
            )
            return self._cross_encoder_reranker

    @property
    def smart_replacer(self) -> SmartReplacer | None:
        """Get SmartReplacer (lazy initialization with thread-safety).

        Returns:
            SmartReplacer instance if smart replacement is enabled, None otherwise.

        Raises:
            RuntimeError: If ENABLE_SMART_REPLACE=true but initialization fails.
        """
        # Fast path: already initialized or not configured
        if self._smart_replacer is not None or not self.config.enable_smart_replace:
            return self._smart_replacer

        # Slow path: need to initialize with lock
        with self._smart_replacer_lock:
            # Double-check after acquiring lock
            if self._smart_replacer is not None or not self.config.enable_smart_replace:
                return self._smart_replacer

            # Initialize SmartReplacer
            smart_replacer_config = SmartReplacerConfig.from_config(
                ConfigAdapter(self.config)
            )
            self._smart_replacer = SmartReplacer(
                config=smart_replacer_config, logger=self.logger
            )
            self.logger.info(
                f"Lazy initialized SmartReplacer [model={self.config.llm_model}]",
                extra={
                    "smart_replacer": "enabled",
                    "model": self.config.llm_model,
                },
            )
            return self._smart_replacer

    def get_reranker(self) -> CrossEncoderReranker | None:
        """Get the configured cross-encoder reranker, or None if disabled."""
        if self.config.reranker_engine == "cross_encoder":
            return self.cross_encoder_reranker
        return None

    def _add_memory(self, memory: str) -> bool:
        """Add a single memory to BOTH USearch semantic and Tantivy full-text engines if not duplicate.

        Args:
            memory: The memory to store.

        Returns:
            True if the memory was stored, False if it was skipped as a duplicate.

        Raises:
            RuntimeError: If storage operation fails.
        """
        if self.config.deduplicate_memories and self._has_exact_match(memory):
            self.logger.info(
                "Duplicate memory detected, skipping storage",
                extra={
                    "workspace_id": self.workspace_id,
                    "memory_preview": memory[:200],
                },
            )
            return False

        try:
            # 1. Add to USearch semantic engine
            self._semantic_engine.add(
                workspace_id=self.workspace_id,
                content=memory,
                infer=self.config.enable_llm_infer,
            )

            # 2. Add to Tantivy full-text search engine
            if self._tantivy_engine is not None:
                self._tantivy_engine.add(self.workspace_id, memory)

            self.logger.debug(
                "Memory added to hybrid storage",
                extra={
                    "workspace_id": self.workspace_id,
                    "memory_length": len(memory),
                    "engines": ["semantic", "tantivy"],
                },
            )
            return True

        except Exception as e:
            raise StorageError(f"Failed to add memory to hybrid storage: {e}") from e

    def _add_message(self, message: str) -> bool:
        return self._add_memory(message)

    def add_memories(self, memories: list[str]) -> int:
        """Add multiple memories to memory store (thread-safe).

        Thread-safe: Uses RLock to ensure consistent state during batch addition.

        Args:
            memories: List of memories to store.

        Returns:
            Number of memories actually stored (duplicates skipped).

        Raises:
            RuntimeError: If storage operation fails.
        """
        memories_to_add: list[str] = []
        seen_memories: set[str] = set()
        with self._lock:

            log_limit = min(len(memories), LOG_ADD_MEMORY_PREVIEW_LIMIT)
            for idx, memory in enumerate(memories, 1):
                if idx <= log_limit:
                    preview = truncate_memory(memory, max_length=60)
                    self.logger.info(
                        f"  ⏳ [{idx}/{len(memories)}] Processing: {preview}",
                        extra={
                            "memory_index": idx,
                            "total_memories": len(memories),
                            "memory_length": len(memory),
                        },
                    )
                if memory in seen_memories:
                    if idx <= log_limit:
                        self.logger.info(
                            "    Skipped (duplicate in batch)",
                            extra={
                                "memory_index": idx,
                                "reason": "batch_duplicate",
                            },
                        )
                    continue
                seen_memories.add(memory)

                if self.config.deduplicate_memories and self._has_exact_match(memory):
                    if idx <= log_limit:
                        self.logger.info(
                            "    Skipped (duplicate detected)",
                            extra={"memory_index": idx, "reason": "duplicate"},
                        )
                    continue

                memories_to_add.append(memory)
            if len(memories) > log_limit:
                self.logger.info(
                    f"  ... {len(memories) - log_limit} more memory(s) omitted from logs",
                    extra={
                        "omitted_count": len(memories) - log_limit,
                        "total_memories": len(memories),
                    },
                )

        if not memories_to_add:
            return 0

        embedder = getattr(self._semantic_engine, "embedder", None)
        embed_documents = getattr(embedder, "embed_documents", None)
        vectors: list[list[float]] | None = None
        if callable(embed_documents):
            computed = embed_documents(memories_to_add)
            if isinstance(computed, list):
                typed_vectors: list[list[float]] = []
                valid = True
                for item in computed:
                    if not isinstance(item, list):
                        valid = False
                        break
                    typed_vectors.append(item)
                if valid:
                    if len(typed_vectors) != len(memories_to_add) or any(
                        not item for item in typed_vectors
                    ):
                        raise StorageError(
                            "Embedding batch size mismatch or empty vector"
                        )
                    vectors = typed_vectors

        with self._write_lock, self._lock:
            if vectors is None:
                inserted_memories = self._semantic_engine.add_batch(
                    workspace_id=self.workspace_id,
                    contents=memories_to_add,
                    infer=self.config.enable_llm_infer,
                )
            else:
                inserted_memories = self._semantic_engine.add_batch(
                    workspace_id=self.workspace_id,
                    contents=memories_to_add,
                    infer=self.config.enable_llm_infer,
                    vectors=vectors,
                )

            if self._tantivy_engine is not None:
                add_batch = type(self._tantivy_engine).__dict__.get("add_batch")
                if callable(add_batch):
                    _ = add_batch(
                        self._tantivy_engine, self.workspace_id, inserted_memories
                    )
                else:
                    for memory in inserted_memories:
                        self._tantivy_engine.add(self.workspace_id, memory)

            stored_count = len(inserted_memories)
            if stored_count != len(memories_to_add):
                self.logger.warning(
                    "    Skipped during batch insert",
                    extra={
                        "reason": "batch_insert_skipped",
                        "expected_count": len(memories_to_add),
                        "stored_count": stored_count,
                    },
                )

            if self._tantivy_engine is not None:
                self._tantivy_engine.commit()
                self.logger.info(
                    "  Tantivy index committed",
                    extra={"engine": "tantivy"},
                )

            self._semantic_engine.commit()
            self.logger.info(
                "  USearch index committed",
                extra={"engine": "usearch"},
            )

            return stored_count

    def add_messages(self, messages: list[str]) -> int:
        return self.add_memories(messages)

    async def add_memories_async(
        self, memories: list[str], dry_run: bool = False
    ) -> AddResult:
        """Add multiple memories with phased parallel processing (Sprint 2.2).

        Uses a 3-phase approach for optimal performance via AddPipeline:
        - Phase 1 (Parallel): Duplicate detection for all memories
        - Phase 2 (Parallel): Smart replacement detection for non-duplicates
        - Phase 3 (Sequential): Database writes to avoid SQLite corruption

        This approach provides 5-8x speedup over sequential processing for
        multiple memories by maximizing I/O parallelism in phases 1-2 while
        maintaining data consistency in phase 3.

        Args:
            memories: List of memories to store.
            dry_run: If True, only check for replacements without making changes.
                Returns what WOULD happen without actually storing or deleting.

        Returns:
            AddResult with detailed information about stored, skipped, and replaced
            memories, including full replacement details.

        Raises:
            RuntimeError: If storage operation fails (not raised in dry_run mode).
        """
        return await self._add_pipeline.execute(memories, dry_run)

    async def add_messages_async(
        self, messages: list[str], dry_run: bool = False
    ) -> AddResult:
        return await self.add_memories_async(messages, dry_run)

    def get_all(self) -> list[str]:
        """Retrieve all stored memories with cross-engine consistency check (thread-safe).

        Thread-safe: Uses RLock to ensure consistent state during retrieval.

        Returns:
            List of all memories from USearchEngine (source of truth).

        Raises:
            RuntimeError: If retrieval operation fails.
        """
        try:
            with self._lock:
                memories = self._semantic_engine.get_all(workspace_id=self.workspace_id)
            self.logger.info(
                f"Retrieved {len(memories)} memories (USearchEngine={len(memories)})",
                extra={
                    "workspace_id": self.workspace_id,
                    "count": len(memories),
                },
            )
            return memories
        except Exception as e:
            raise StorageError(f"Failed to retrieve memories: {e}") from e

    async def search(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[str]:
        """Hybrid semantic + full-text search with RRF ranking (async).

        This async method uses the SearchPipeline to execute a 4-step search:
        1. Parallel semantic + full-text search
        2. RRF fusion or concatenation
        3. Fusion threshold filtering (when RRF enabled)
        4. Reranking (CrossEncoder)

        Args:
            query: Search query string.
            limit: Maximum number of results (uses config default if None).

        Returns:
            List of hybrid-ranked memories from both engines.

        Raises:
            SearchError: If search operation fails.

        Note:
            Cancelling this await does not abort native USearch or Tantivy
            work already running in a worker thread.
        """
        # Use defaults from config if not provided
        if limit is None:
            limit = self.config.search_limit

        # Index restore / len() can block; keep it off the event loop.
        index_size: int = await asyncify(self._semantic_index_size)()
        overfetch_limit = calculate_adaptive_overfetch(limit, index_size, self.config)

        # Create search context
        context = SearchContext(
            query=query,
            limit=limit,
            overfetch_limit=overfetch_limit,
            enable_hybrid_search=self.is_hybrid_search,
            enable_rrf_fusion=self.config.enable_rrf_fusion,
            reranker_engine=self.config.reranker_engine,
            workspace_id=self.workspace_id,
        )

        # Execute search pipeline
        result = await self._search_pipeline.execute(context)

        return result.memories

    def _semantic_index_size(self) -> int:
        """Return the semantic index size, or 0 if it cannot be read."""
        try:
            raw_index = getattr(self._semantic_engine, "_index", None)
            if raw_index is not None:
                return len(raw_index)
            return 0
        except Exception:
            return 0

    def search_engine_status(self) -> dict[str, str]:
        """Return public readiness of search engines for health checks.

        Values are ``initialized`` (warmed), ``pending`` (constructed but
        not yet ``ensure_initialized``), ``not_initialized``, or ``disabled``.
        """
        return {
            "semantic_engine": self._engine_readiness(
                self._semantic_engine, absent="not_initialized"
            ),
            "tantivy_engine": self._engine_readiness(
                self._tantivy_engine, absent="disabled"
            ),
        }

    def _engine_readiness(self, engine: object | None, *, absent: str) -> str:
        if engine is None:
            return absent
        is_ready = type(engine).__dict__.get("is_ready")
        try:
            if callable(is_ready) and is_ready(engine):
                return "initialized"
        except Exception:
            return "pending"
        return "pending"

    def search_for_removal(
        self, query: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Search for memories to potentially remove using direct database lookup.

        Uses O(log n) indexed database lookup instead of O(n) iteration through
        all memories. This is a significant performance improvement for large
        collections.

        Args:
            query: The exact memory content to find for removal.
            limit: Maximum number of candidates (uses config default if None).
                   Note: Since we're looking for exact match, we'll return at most 1.

        Returns:
            List of candidate records with 'id', 'memory', and 'score' fields.
            Returns at most 1 candidate since exact match is unique.

        Raises:
            SearchError: If the lookup fails.
        """
        # Note: limit is kept for API compatibility but exact match returns at most 1
        _ = limit  # Unused, kept for API compatibility

        try:
            candidates: list[dict[str, Any]] = []

            # Direct database lookup - O(log n) instead of O(n) get_all() + iteration
            mem_id = self.get_id_by_content(query)

            if mem_id is not None:
                # Found exact match - return as candidate
                candidates.append(
                    {
                        "id": str(mem_id),  # Use database ID directly
                        "memory": query,
                        "score": 1.0,  # Exact match = perfect score
                    }
                )

            self.logger.debug(
                f"Direct lookup removal candidates: {len(candidates)}",
                extra={"workspace_id": self.workspace_id, "query": query[:50]},
            )
            return candidates

        except Exception as e:
            raise SearchError(f"Failed to search for removal: {e}") from e

    def delete_by_id(self, memory_id: str) -> None:
        """Delete a memory entry by its ID.

        Args:
            memory_id: The ID of the memory to delete.

        Raises:
            StorageError: If deletion fails.
        """
        with self._write_lock:
            try:
                self._semantic_engine.delete(memory_id=memory_id)
            except Exception as e:
                raise StorageError(f"Failed to delete memory: {e}") from e

    def delete_by_memory(self, memory: str) -> bool:
        """Delete a memory entry by its exact memory content (thread-safe).

        Thread-safe: Uses RLock to ensure consistent state during deletion.

        This method looks up the memory in the database and deletes it from
        both USearch (semantic) and Tantivy (full-text) engines atomically.
        If Tantivy deletion fails after USearch deletion succeeds, the operation
        raises an error to alert about the inconsistent state.

        Args:
            memory: The exact memory content to delete.

        Returns:
            True if the memory was found and deleted, False if not found.

        Raises:
            RuntimeError: If deletion fails or results in inconsistent state.
        """
        with self._write_lock, self._lock:
            try:
                # 1. Look up the SQLite ID from the memory content
                mem_id = self.get_id_by_content(memory)

                if mem_id is None:
                    self.logger.debug(
                        "Memory not found for deletion",
                        extra={
                            "workspace_id": self.workspace_id,
                            "memory_preview": memory[:50],
                        },
                    )
                    return False

                # 2. Delete from USearch semantic engine using the numeric ID
                self._semantic_engine.delete(memory_id=str(mem_id))

                # 3. Delete from Tantivy full-text engine
                # If this fails after USearch deletion, we have inconsistent state
                if self._tantivy_engine is not None:
                    try:
                        deleted = self._tantivy_engine.delete(
                            self.workspace_id, memory, verify_exists=True
                        )
                        if not deleted:
                            raise InconsistentStateError(
                                "USearch deletion succeeded but Tantivy "
                                f"did not delete {memory!r}"
                            )
                    except InconsistentStateError:
                        raise
                    except Exception as tantivy_error:
                        # Log critical error - state is now inconsistent
                        self.logger.error(
                            "Tantivy deletion failed after USearch deletion - INCONSISTENT STATE",
                            extra={
                                "workspace_id": self.workspace_id,
                                "memory_id": mem_id,
                                "error": str(tantivy_error),
                            },
                        )
                        raise InconsistentStateError(
                            "USearch deletion succeeded but Tantivy deletion failed: "
                            f"{tantivy_error}"
                        ) from tantivy_error

                self.logger.debug(
                    "Memory deleted from hybrid storage",
                    extra={
                        "workspace_id": self.workspace_id,
                        "memory_id": mem_id,
                        "engines": ["usearch", "tantivy"]
                        if self._tantivy_engine
                        else ["usearch"],
                    },
                )

                return True

            except InconsistentStateError:
                raise  # Re-raise inconsistent state errors with full context
            except Exception as e:
                raise StorageError(f"Failed to delete memory: {e}") from e

    def delete_by_message(self, message: str) -> bool:
        return self.delete_by_memory(message)

    def delete_memories(self, memories: list[str]) -> list[str]:
        """Delete many memories under one write lock and one Tantivy commit.

        Returns:
            Contents that were found and deleted.
        """
        unique = list(dict.fromkeys(memories))
        if not unique:
            return []
        with self._write_lock, self._lock:
            found: list[tuple[int, str]] = []
            for memory in unique:
                mem_id = self.get_id_by_content(memory)
                if mem_id is not None:
                    found.append((mem_id, memory))
            contents = [memory for _mem_id, memory in found]
            for mem_id, _memory in found:
                self._semantic_engine.delete(memory_id=str(mem_id))
            if found and self._tantivy_engine is not None:
                try:
                    delete_batch = type(self._tantivy_engine).__dict__.get(
                        "delete_batch"
                    )
                    if callable(delete_batch):
                        deleted_count = delete_batch(
                            self._tantivy_engine,
                            self.workspace_id,
                            contents,
                            verify_exists=True,
                        )
                        if not isinstance(deleted_count, int):
                            raise InconsistentStateError(
                                "USearch deletion succeeded but Tantivy "
                                f"delete_batch returned {type(deleted_count).__name__}"
                            )
                        if deleted_count < len(contents):
                            raise InconsistentStateError(
                                "USearch deletion succeeded but Tantivy "
                                f"deleted {deleted_count}/{len(contents)}"
                            )
                    else:
                        for content in contents:
                            deleted = self._tantivy_engine.delete(
                                self.workspace_id, content, verify_exists=True
                            )
                            if not deleted:
                                raise InconsistentStateError(
                                    "USearch deletion succeeded but Tantivy "
                                    f"did not delete {content!r}"
                                )
                except InconsistentStateError:
                    raise
                except Exception as tantivy_error:
                    raise InconsistentStateError(
                        "USearch deletion succeeded but Tantivy deletion failed: "
                        f"{tantivy_error}"
                    ) from tantivy_error
            if found:
                self._semantic_engine.commit()
            return contents

    def get_id_by_content(self, content: str) -> int | None:
        """Get SQLite ID for exact memory content match.

        Uses get_id_by_content() when available, with fallback to a
        legacy ID lookup API for compatibility.
        """
        return self._semantic_engine.get_id_by_content(self.workspace_id, content)

    def get_id_by_message(self, message: str) -> int | None:
        return self.get_id_by_content(message)

    def _has_exact_match(self, content: str) -> bool:
        """Check whether the exact memory already exists in storage.

        Uses the unique SQLite (workspace_id, content) index. Tantivy is not
        consulted for identity because stemming cannot do exact match.
        """
        return has_exact_match(
            semantic_engine=self._semantic_engine,
            tantivy_engine=self._tantivy_engine,
            workspace_id=self.workspace_id,
            content=content,
            logger=self.logger,
        )

    def close(self) -> None:
        """Close all resources and persist data to disk (thread-safe).

        This method ensures all data is properly saved before shutdown:
        1. Commits and closes Tantivy full-text engine (if enabled)
        2. Commits and closes USearch semantic engine

        Thread-safe: Uses RLock to ensure consistent state during shutdown.

        Should be called during graceful shutdown (e.g., on SIGINT/SIGTERM)
        to prevent data loss.
        """
        with self._write_lock, self._lock:
            self.logger.info(
                "Closing MemoryManager - persisting data to disk...",
                extra={"workspace_id": self.workspace_id},
            )

            # 1. Close Tantivy engine first (if enabled)
            if self._tantivy_engine is not None:
                try:
                    # Use flush() to commit and wait for all merge threads
                    self._tantivy_engine.flush()
                    self._tantivy_engine.close()
                    self.logger.info(
                        "Tantivy engine closed successfully",
                        extra={"workspace_id": self.workspace_id, "engine": "tantivy"},
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error closing Tantivy engine: {e}",
                        extra={
                            "workspace_id": self.workspace_id,
                            "engine": "tantivy",
                            "error": str(e),
                        },
                    )

            # 2. Close USearch semantic engine
            try:
                # Commit to save USearch index to disk
                self._semantic_engine.commit()
                self._semantic_engine.close()
                self.logger.info(
                    "USearch engine closed successfully",
                    extra={"workspace_id": self.workspace_id, "engine": "usearch"},
                )
            except Exception as e:
                self.logger.error(
                    f"Error closing USearch engine: {e}",
                    extra={
                        "workspace_id": self.workspace_id,
                        "engine": "usearch",
                        "error": str(e),
                    },
                )

            self.logger.info(
                "MemoryManager closed - all data persisted",
                extra={"workspace_id": self.workspace_id},
            )
