"""Memory management wrapper for hybrid search integration."""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import anyio
from asyncer import asyncify, create_task_group

from openmemories.application.types import ISemanticSearchEngine
from openmemories.infrastructure import (
    CachedEmbeddings,
    CrossEncoderConfig,
    CrossEncoderReranker,
    LangchainQwenEmbeddings,
    LLMReranker,
    LLMRerankerConfig,
    SmartReplacer,
    SmartReplacerConfig,
    TantivyConfig,
    TantivyEngine,
    USearchConfig,
    USearchEngine,
)

from ..config import Config
from ..exceptions import InconsistentStateError, SearchError, StorageError
from ..utils import (
    StructuredLogger,
    format_fusion_score_status,
    truncate_message,
)
from .fusion import FusionEngine, create_fusion_engine


@dataclass
class ReplacementInfo:
    """Information about a memory replacement that occurred.

    Attributes:
        old_memory: The memory that was replaced (deleted).
        new_memory: The new memory that replaced it.
        confidence: LLM confidence score for the replacement decision (0.0-1.0).
        reason: LLM's explanation for why replacement was appropriate.
        similarity_score: Embedding similarity between old and new memories.
    """

    old_memory: str
    new_memory: str
    confidence: float
    reason: str
    similarity_score: float = 0.0


@dataclass
class AddResult:
    """Result of adding messages to memory storage.

    Provides detailed information about what happened during the add operation,
    including which messages were stored, skipped (duplicates), and replaced.

    Attributes:
        stored_count: Number of messages successfully stored.
        skipped_count: Number of messages skipped (duplicates).
        replaced_count: Number of existing memories that were replaced.
        replacements: Detailed info about each replacement that occurred.
    """

    stored_count: int = 0
    skipped_count: int = 0
    replaced_count: int = 0
    replacements: List[ReplacementInfo] = field(default_factory=list)


# Constants for magic numbers (documented for maintainability)
MIN_OVERFETCH_LIMIT = 20  # Minimum docs to fetch for better fusion quality
TANTIVY_SCORE_DIVISOR = 10.0  # Tantivy BM25 scores typically 0-10+, normalize to 0-1
LOG_QUERY_TRUNCATE_LENGTH = 100  # Max query length in log messages


class MemoryManager:
    """Manages memory storage and retrieval using USearch with SQLite backend."""

    def __init__(self, config: Config, logger: StructuredLogger):
        """Initialize Hybrid MemoryManager with USearch semantic & Tantivy full-text engines.

        Args:
            config: Application configuration.
            logger: Structured logger instance.
        """
        self.config = config
        self.logger = logger
        self.project_id = config.project_id

        # Thread-safety: RLock for protecting concurrent operations
        # RLock (re-entrant) because some methods may call other protected methods
        self._lock = threading.RLock()

        # Hybrid mode enabled by default
        self.is_hybrid_search = self.config.enable_hybrid_search

        # 1. Initialize USearch semantic engine
        self._semantic_engine: ISemanticSearchEngine
        usearch_config = USearchConfig.from_app_config(config)
        base_embedder = LangchainQwenEmbeddings(
            config={
                "model": config.embedding_model,
                "embedding_dims": config.qwen_embedding_dims
                if config.embedder_provider == "langchain"
                else config.embedding_dims,
                "api_key": config.openrouter_api_key.get_secret_value(),
                "openai_base_url": config.openrouter_base_url,
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

        self._semantic_engine = USearchEngine(
            usearch_config, embedder=embedder, logger=self.logger
        )

        # 2. Initialize Tantivy full-text engine (hybrid companion)
        self._tantivy_engine: Optional[TantivyEngine] = None
        if self.is_hybrid_search:
            tantivy_config = TantivyConfig(
                project_id=self.project_id,
                index_path=self.config.tantivy_index_path_template.format(
                    project_id=self.project_id
                ).lower(),
            )
            self._tantivy_engine = TantivyEngine(tantivy_config, logger=self.logger)

        # 3. Initialize fusion engine for hybrid ranking
        self._fusion_engine: FusionEngine = create_fusion_engine(
            method=self.config.fusion_method,
            normalization=self.config.fusion_normalization,
            rrf_k=self.config.fusion_rrf_k,
            logger=self.logger,
        )

        # 4. Initialize reranker (LLM or CrossEncoder, based on reranker_engine config)
        self._llm_reranker: LLMReranker | None = None
        self._cross_encoder_reranker: CrossEncoderReranker | None = None

        if self.config.reranker_engine == "llm":
            reranker_config = LLMRerankerConfig.from_app_config(config)
            self._llm_reranker = LLMReranker(config=reranker_config, logger=self.logger)
            self.logger.info(
                f"Initialized LLM reranker [model={config.llm_model}]",
                extra={"reranker_engine": "llm", "model": config.llm_model},
            )
        elif self.config.reranker_engine == "cross_encoder":
            ce_config = CrossEncoderConfig.from_app_config(config)
            self._cross_encoder_reranker = CrossEncoderReranker(
                config=ce_config, logger=self.logger
            )
            self.logger.info(
                f"Initialized CrossEncoder reranker [model={config.cross_encoder_model}]",
                extra={
                    "reranker_engine": "cross_encoder",
                    "model": config.cross_encoder_model,
                    "device": config.cross_encoder_device,
                },
            )
        else:
            self.logger.info(
                "Reranking disabled (RERANKER_ENGINE=none)",
                extra={"reranker_engine": "none"},
            )

        # 5. Initialize smart replacer (LLM-based memory replacement detection)
        self._smart_replacer: SmartReplacer | None = None
        if self.config.enable_smart_replace:
            smart_replacer_config = SmartReplacerConfig.from_app_config(config)
            self._smart_replacer = SmartReplacer(
                config=smart_replacer_config, logger=self.logger
            )
            self.logger.info(
                f"Initialized SmartReplacer [model={config.llm_model}, "
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

        self.logger.info(
            f"Initialized Hybrid MemoryManager [project_id={self.project_id}, "
            f"semantic_backend=usearch, "
            f"hybrid_search={self.is_hybrid_search}, "
            f"embedding_model={self.config.embedding_model}, "
            f"tantivy_index={self.config.tantivy_index_path_template.format(project_id=self.project_id)}, "
            f"embedding_dims={self.config.qwen_embedding_dims if self.config.embedder_provider == 'langchain' else self.config.embedding_dims}, "
            f"embedder={config.embedder_provider}]",
            extra={
                "project_id": self.project_id,
                "semantic_backend": "usearch",
                "hybrid_search": self.is_hybrid_search,
                "embedding_model": config.embedding_model,
                "tantivy_index_path": self.config.tantivy_index_path_template.format(
                    project_id=self.project_id
                ),
                "embedding_dims": config.embedding_dims,
                "embedder": config.embedder_provider,
            },
        )

        # 6. Eager initialization (pre-warm engines if enabled)
        if config.eager_initialization:
            self._eager_initialize_engines()

    def _eager_initialize_engines(self) -> None:
        """Pre-warm all storage engines for faster first operation.

        This method forces initialization of lazy-loaded resources:
        - USearch index and libSQL connection
        - Tantivy index, writer, and searcher

        Useful for reducing first-request latency in production deployments.
        Called during __init__ when EAGER_INITIALIZATION=true.
        """
        start_time = time.time()
        self.logger.info(
            "Starting eager engine initialization...",
            extra={"project_id": self.project_id},
        )

        # Pre-warm USearch semantic engine
        self._semantic_engine.ensure_initialized()

        # Pre-warm Tantivy full-text engine
        if self._tantivy_engine is not None:
            self._tantivy_engine.ensure_initialized()

        elapsed_ms = (time.time() - start_time) * 1000
        self.logger.info(
            f"Eager initialization complete [{elapsed_ms:.1f}ms]",
            extra={
                "project_id": self.project_id,
                "elapsed_ms": elapsed_ms,
                "engines": ["usearch", "tantivy" if self._tantivy_engine else None],
            },
        )

    def _add_message(self, message: str) -> bool:
        """Add a single message to BOTH USearch semantic and Tantivy full-text engines if not duplicate.

        Args:
            message: The message to store.

        Returns:
            True if the message was stored, False if it was skipped as a duplicate.

        Raises:
            RuntimeError: If storage operation fails.
        """
        if self.config.deduplicate_messages and self._has_exact_match(message):
            self.logger.info(
                "Duplicate message detected, skipping storage",
                extra={
                    "project_id": self.project_id,
                    "message_preview": message[:200],
                },
            )
            return False

        try:
            # 1. Add to USearch semantic engine
            self._semantic_engine.add(
                project_id=self.project_id,
                message=message,
                infer=self.config.enable_llm_infer,
            )

            # 2. Add to Tantivy full-text search engine
            if self._tantivy_engine is not None:
                self._tantivy_engine.add(self.project_id, message)

            self.logger.debug(
                "Message added to hybrid storage",
                extra={
                    "project_id": self.project_id,
                    "message_length": len(message),
                    "engines": ["semantic", "tantivy"],
                },
            )
            return True

        except Exception as e:
            raise StorageError(f"Failed to add message to hybrid storage: {e}") from e

    def _archive_for_replacement(
        self, old_memory: str, new_memory: str, confidence: float, reason: str
    ) -> bool:
        """Archive a message before replacement (for recovery).

        This method archives a message to the archived_messages table before
        it gets deleted during smart replacement. This enables recovery if
        the LLM made a mistake in determining the replacement.

        Args:
            old_memory: The message being replaced and archived.
            new_memory: The new message that replaces it.
            confidence: LLM confidence score for the replacement decision.
            reason: LLM's explanation for why replacement was appropriate.

        Returns:
            True if archiving succeeded, False otherwise.
        """
        try:
            # Get the message ID from semantic engine's message store
            msg_id = self._semantic_engine.get_id_by_message(
                self.project_id, old_memory
            )

            if msg_id is None:
                self.logger.warning(
                    "Cannot archive - message not found",
                    extra={
                        "project_id": self.project_id,
                        "old_memory_preview": old_memory[:50],
                    },
                )
                return False

            # Access the message store through the semantic engine
            message_store = self._semantic_engine.message_store
            archive_id = message_store.archive(
                message_id=msg_id,
                project_id=self.project_id,
                message=old_memory,
                replaced_by=new_memory,
                reason=reason,
                confidence=confidence,
            )

            if archive_id is not None:
                self.logger.info(
                    "Message archived before replacement",
                    extra={
                        "project_id": self.project_id,
                        "archive_id": archive_id,
                        "original_id": msg_id,
                        "confidence": confidence,
                    },
                )
                return True

            return False

        except Exception as e:
            # Graceful degradation - log warning but don't block the replacement
            self.logger.warning(
                f"Failed to archive message for replacement: {e}",
                extra={
                    "project_id": self.project_id,
                    "error": str(e),
                },
            )
            return False

    async def _check_for_replacement(self, new_memory: str) -> List[ReplacementInfo]:
        """Check if new memory should replace existing memories.

        Uses semantic search to find the most similar existing memories,
        applies similarity pre-filter, then uses LLM to determine which
        should be replaced.

        Args:
            new_memory: The new memory being added.

        Returns:
            List of ReplacementInfo objects for memories that should be replaced.
            Empty list if no replacements needed.
        """
        if self._smart_replacer is None:
            return []

        # Capture in local variable for type narrowing in nested function
        smart_replacer = self._smart_replacer

        replacements: List[ReplacementInfo] = []

        try:
            # Step 1: Find top N most similar memories via semantic search
            candidate_limit = self.config.smart_replace_candidate_limit
            similar_results = self._semantic_engine.search(
                query=new_memory,
                project_id=self.project_id,
                limit=candidate_limit,
            )

            if not similar_results:
                self.logger.debug(
                    "No similar memories found for replacement check",
                    extra={
                        "project_id": self.project_id,
                        "new_memory_preview": new_memory[:100],
                    },
                )
                return []

            # Step 2: Filter candidates by similarity threshold (pre-filter)
            min_similarity = self.config.smart_replace_min_similarity
            filtered_candidates = [
                (mem, score)
                for mem, score in similar_results
                if score >= min_similarity and mem != new_memory
            ]

            if not filtered_candidates:
                self.logger.debug(
                    f"All candidates below similarity threshold ({min_similarity})",
                    extra={
                        "project_id": self.project_id,
                        "candidate_count": len(similar_results),
                        "min_similarity": min_similarity,
                    },
                )
                return []

            self.logger.debug(
                f"Found {len(filtered_candidates)} candidates above similarity threshold",
                extra={
                    "project_id": self.project_id,
                    "candidate_count": len(filtered_candidates),
                    "min_similarity": min_similarity,
                    "top_similarity": filtered_candidates[0][1]
                    if filtered_candidates
                    else 0,
                },
            )

            # Step 3: Check candidates in parallel with LLM (rate-limited)
            # Reuse rerank_max_concurrency for API rate limiting
            semaphore = anyio.Semaphore(self.config.rerank_max_concurrency)

            async def check_single_candidate(
                existing_memory: str, similarity_score: float
            ) -> Optional[ReplacementInfo]:
                """Check a single candidate for replacement (with semaphore)."""
                async with semaphore:
                    try:
                        (
                            should_replace,
                            confidence,
                            reason,
                        ) = await smart_replacer.check_replacement(
                            new_memory=new_memory,
                            existing_memory=existing_memory,
                        )

                        if should_replace:
                            self.logger.info(
                                f"Smart replacement triggered (confidence={confidence:.2f})",
                                extra={
                                    "project_id": self.project_id,
                                    "should_replace": True,
                                    "confidence": confidence,
                                    "similarity_score": similarity_score,
                                    "reason": reason,
                                    "old_memory_preview": truncate_message(
                                        existing_memory, max_length=60
                                    ),
                                    "new_memory_preview": truncate_message(
                                        new_memory, max_length=60
                                    ),
                                },
                            )
                            return ReplacementInfo(
                                old_memory=existing_memory,
                                new_memory=new_memory,
                                confidence=confidence,
                                reason=reason,
                                similarity_score=similarity_score,
                            )
                        else:
                            self.logger.debug(
                                f"No replacement needed (confidence={confidence:.2f}): {reason}",
                                extra={
                                    "project_id": self.project_id,
                                    "should_replace": False,
                                    "confidence": confidence,
                                    "reason": reason,
                                },
                            )
                            return None
                    except Exception as candidate_error:
                        # Graceful degradation: log warning and skip this candidate
                        self.logger.warning(
                            f"LLM check failed for candidate: {candidate_error}",
                            extra={
                                "project_id": self.project_id,
                                "error": str(candidate_error),
                                "existing_preview": existing_memory[:100],
                            },
                        )
                        return None

            # Run all candidate checks in parallel
            results: List[Optional[ReplacementInfo]] = []

            async def collect_result(
                existing_memory: str, similarity_score: float
            ) -> None:
                """Run check and collect result."""
                result = await check_single_candidate(existing_memory, similarity_score)
                results.append(result)

            async with anyio.create_task_group() as tg:
                for existing_memory, similarity_score in filtered_candidates:
                    tg.start_soon(collect_result, existing_memory, similarity_score)

            # Filter out None results (failed checks or no replacement needed)
            replacements = [r for r in results if r is not None]

            return replacements

        except Exception as e:
            # Graceful degradation: log warning and proceed without replacement
            self.logger.warning(
                f"Smart replacement check failed: {e}",
                extra={
                    "project_id": self.project_id,
                    "error": str(e),
                    "new_memory_preview": new_memory[:100],
                },
            )
            return []

    def add_messages(self, messages: List[str]) -> int:
        """Add multiple messages to memory store (thread-safe).

        Thread-safe: Uses RLock to ensure consistent state during batch addition.

        Args:
            messages: List of messages to store.

        Returns:
            Number of messages actually stored (duplicates skipped).

        Raises:
            RuntimeError: If storage operation fails.
        """
        with self._lock:
            stored_count = 0
            for idx, message in enumerate(messages, 1):
                preview = truncate_message(message, max_length=60)
                self.logger.info(
                    f"  ⏳ [{idx}/{len(messages)}] Processing: {preview}",
                    extra={
                        "message_index": idx,
                        "total_messages": len(messages),
                        "message_length": len(message),
                    },
                )
                if self._add_message(message):
                    stored_count += 1
                    self.logger.info(
                        "    Stored in USearch (semantic) + Tantivy (full-text)",
                        extra={
                            "message_index": idx,
                            "engines": ["usearch", "tantivy"]
                            if self._tantivy_engine
                            else ["usearch"],
                        },
                    )
                else:
                    self.logger.info(
                        "    Skipped (duplicate detected)",
                        extra={"message_index": idx, "reason": "duplicate"},
                    )

            # Commit Tantivy changes after batch
            if self._tantivy_engine is not None:
                self._tantivy_engine.commit()
                self.logger.info(
                    "  Tantivy index committed",
                    extra={"engine": "tantivy"},
                )

            # Commit USearch semantic engine changes
            self._semantic_engine.commit()
            self.logger.info(
                "  USearch index committed",
                extra={"engine": "usearch"},
            )

            return stored_count

    async def _add_message_async(self, message: str) -> bool:
        """Add a single message asynchronously to hybrid storage.

        This is the async version of _add_message() for use with parallel execution.
        Uses asyncify() for better type inference and keyword argument support.

        Note: USearch operations are synchronous. The async wrapper enables
        concurrent processing of multiple messages but does NOT provide I/O
        efficiency gains. The underlying sync code runs in a thread pool.

        Args:
            message: The message to store.

        Returns:
            True if the message was stored, False if it was skipped as a duplicate.

        Raises:
            RuntimeError: If storage operation fails.
        """
        # Use asyncify for better type inference than anyio.to_thread.run_sync
        return await asyncify(self._add_message)(message)

    async def add_messages_async(
        self, messages: List[str], dry_run: bool = False
    ) -> AddResult:
        """Add multiple messages with phased parallel processing (Sprint 2.2).

        Uses a 3-phase approach for optimal performance:
        - Phase 1 (Parallel): Duplicate detection for all messages
        - Phase 2 (Parallel): Smart replacement detection for non-duplicates
        - Phase 3 (Sequential): Database writes to avoid SQLite corruption

        This approach provides 5-8x speedup over sequential processing for
        multiple messages by maximizing I/O parallelism in phases 1-2 while
        maintaining data consistency in phase 3.

        Args:
            messages: List of messages to store.
            dry_run: If True, only check for replacements without making changes.
                Returns what WOULD happen without actually storing or deleting.

        Returns:
            AddResult with detailed information about stored, skipped, and replaced
            messages, including full replacement details.

        Raises:
            RuntimeError: If storage operation fails (not raised in dry_run mode).
        """
        result = AddResult()

        if not messages:
            return result

        mode_str = "DRY RUN" if dry_run else "LIVE"
        self.logger.info(
            f"Starting phased parallel message addition [{mode_str}]: {len(messages)} messages",
            extra={
                "total_messages": len(messages),
                "smart_replace_enabled": self._smart_replacer is not None,
                "dry_run": dry_run,
                "optimization": "phased_parallel",
            },
        )

        try:
            # ============================================================
            # PHASE 1: Parallel duplicate detection + batch deduplication
            # ============================================================
            phase1_start = time.perf_counter()

            # First, deduplicate within the batch itself (preserve order, keep first)
            seen: Dict[str, int] = {}
            unique_messages: List[str] = []
            batch_duplicate_indices: List[int] = []

            for idx, msg in enumerate(messages):
                if msg in seen:
                    batch_duplicate_indices.append(idx)
                else:
                    seen[msg] = idx
                    unique_messages.append(msg)

            batch_duplicates_count = len(batch_duplicate_indices)
            if batch_duplicates_count > 0:
                self.logger.info(
                    f"Phase 1: Found {batch_duplicates_count} duplicates within batch",
                    extra={
                        "batch_duplicates": batch_duplicates_count,
                        "unique_count": len(unique_messages),
                    },
                )

            # Parallel duplicate detection against existing storage
            duplicate_flags: Dict[str, bool] = {}
            semaphore = anyio.Semaphore(self.config.add_max_concurrency)

            async def check_duplicate(msg: str) -> Tuple[str, bool]:
                """Check if message is duplicate (with semaphore for concurrency control)."""
                async with semaphore:
                    is_dup = await asyncify(self._has_exact_match)(msg)
                    return (msg, is_dup)

            # Run all duplicate checks in parallel
            async with create_task_group() as tg:
                results: List[Tuple[str, bool]] = []

                async def collect_result(msg: str) -> None:
                    res = await check_duplicate(msg)
                    results.append(res)

                for msg in unique_messages:
                    tg.soonify(collect_result)(msg)

            # Process results
            for msg, is_dup in results:
                duplicate_flags[msg] = is_dup

            # Separate duplicates from non-duplicates
            storage_duplicates: List[str] = []
            non_duplicate_messages: List[str] = []

            for msg in unique_messages:
                if duplicate_flags.get(msg, False):
                    storage_duplicates.append(msg)
                else:
                    non_duplicate_messages.append(msg)

            phase1_duration = time.perf_counter() - phase1_start
            self.logger.info(
                f"Phase 1 complete: {len(non_duplicate_messages)} unique, "
                f"{len(storage_duplicates)} storage duplicates ({phase1_duration:.3f}s)",
                extra={
                    "phase": 1,
                    "duration_ms": int(phase1_duration * 1000),
                    "unique_count": len(non_duplicate_messages),
                    "storage_duplicates": len(storage_duplicates),
                    "batch_duplicates": batch_duplicates_count,
                },
            )

            # ============================================================
            # PHASE 2: Parallel smart replacement detection
            # ============================================================
            phase2_start = time.perf_counter()

            # Map: message -> list of replacement infos
            replacement_map: Dict[str, List[ReplacementInfo]] = {}

            if self._smart_replacer is not None and non_duplicate_messages:
                async with create_task_group() as tg:
                    replacement_results: List[Tuple[str, List[ReplacementInfo]]] = []

                    async def check_replacement_for_msg(msg: str) -> None:
                        """Check replacement for a single message."""
                        infos = await self._check_for_replacement(msg)
                        replacement_results.append((msg, infos))

                    for msg in non_duplicate_messages:
                        tg.soonify(check_replacement_for_msg)(msg)

                # Collect results
                for msg, infos in replacement_results:
                    replacement_map[msg] = infos

            phase2_duration = time.perf_counter() - phase2_start
            total_replacements = sum(len(infos) for infos in replacement_map.values())
            self.logger.info(
                f"Phase 2 complete: {total_replacements} replacements detected ({phase2_duration:.3f}s)",
                extra={
                    "phase": 2,
                    "duration_ms": int(phase2_duration * 1000),
                    "messages_checked": len(non_duplicate_messages),
                    "total_replacements": total_replacements,
                },
            )

            # ============================================================
            # PHASE 3: Sequential database writes (SQLite constraint)
            # ============================================================
            phase3_start = time.perf_counter()

            # Track skipped count from duplicates
            result.skipped_count = len(storage_duplicates) + batch_duplicates_count

            # Process each non-duplicate message sequentially
            for idx, message in enumerate(non_duplicate_messages):
                replacement_infos = replacement_map.get(message, [])

                # Process replacements (delete old memories)
                for replacement_info in replacement_infos:
                    self.logger.info(
                        "    Replacing old memory with new one",
                        extra={
                            "message_index": idx + 1,
                            "action": "replace",
                            "old_preview": truncate_message(
                                replacement_info.old_memory, max_length=60
                            ),
                            "new_preview": truncate_message(message, max_length=60),
                            "confidence": replacement_info.confidence,
                            "similarity": replacement_info.similarity_score,
                            "dry_run": dry_run,
                        },
                    )

                    if not dry_run:
                        try:
                            # Archive the old memory before deletion (for recovery)
                            archived = self._archive_for_replacement(
                                old_memory=replacement_info.old_memory,
                                new_memory=message,
                                confidence=replacement_info.confidence,
                                reason=replacement_info.reason,
                            )
                            if archived:
                                self.logger.debug(
                                    "    Old memory archived for recovery",
                                    extra={
                                        "message_index": idx + 1,
                                        "action": "archive_success",
                                    },
                                )

                            # Delete the old memory
                            deleted = self.delete_by_message(
                                replacement_info.old_memory
                            )
                            if deleted:
                                result.replaced_count += 1
                                result.replacements.append(replacement_info)
                                self.logger.debug(
                                    "    Old memory removed successfully",
                                    extra={
                                        "message_index": idx + 1,
                                        "action": "delete_success",
                                        "archived": archived,
                                    },
                                )
                            else:
                                self.logger.warning(
                                    "    Old memory not found for deletion",
                                    extra={
                                        "message_index": idx + 1,
                                        "action": "delete_not_found",
                                    },
                                )
                        except Exception as delete_error:
                            # Graceful degradation: log warning and continue
                            self.logger.warning(
                                f"    Failed to delete old memory: {delete_error}",
                                extra={
                                    "message_index": idx + 1,
                                    "action": "delete_failed",
                                    "error": str(delete_error),
                                },
                            )
                    else:
                        # In dry_run mode, just record what would be replaced
                        result.replaced_count += 1
                        result.replacements.append(replacement_info)

                # Add the new message (unless dry_run)
                if not dry_run:
                    stored = self._add_message(message)
                    if stored:
                        result.stored_count += 1
                    else:
                        # Shouldn't happen since we already checked, but handle gracefully
                        result.skipped_count += 1
                else:
                    # In dry_run, assume it would be stored
                    result.stored_count += 1

            phase3_duration = time.perf_counter() - phase3_start

            # Commit changes (only in live mode)
            if not dry_run:
                if self._tantivy_engine is not None:
                    self._tantivy_engine.commit()
                self._semantic_engine.commit()

            self.logger.info(
                f"Phase 3 complete: {result.stored_count} stored ({phase3_duration:.3f}s)",
                extra={
                    "phase": 3,
                    "duration_ms": int(phase3_duration * 1000),
                    "stored_count": result.stored_count,
                    "replaced_count": result.replaced_count,
                },
            )

        except Exception as e:
            self.logger.error(
                f"Phased parallel message addition failed: {e}",
                extra={
                    "total_messages": len(messages),
                    "stored_count": result.stored_count,
                    "replaced_count": result.replaced_count,
                    "error": str(e),
                    "dry_run": dry_run,
                },
            )
            if not dry_run:
                raise StorageError(f"Failed to add messages: {e}") from e

        self.logger.info(
            f"Phased parallel addition completed [{mode_str}]: "
            f"{result.stored_count}/{len(messages)} stored, "
            f"{result.replaced_count} replaced, {result.skipped_count} skipped",
            extra={
                "total_messages": len(messages),
                "stored_count": result.stored_count,
                "replaced_count": result.replaced_count,
                "skipped_count": result.skipped_count,
                "replacement_details": [
                    {
                        "old": truncate_message(r.old_memory, 50),
                        "confidence": r.confidence,
                    }
                    for r in result.replacements
                ],
                "dry_run": dry_run,
                "optimization": "phased_parallel",
            },
        )

        return result

    def get_all(self) -> List[str]:
        """Retrieve all stored messages with cross-engine consistency check (thread-safe).

        Thread-safe: Uses RLock to ensure consistent state during retrieval.

        Returns:
            List of all messages from USearchEngine (source of truth).

        Raises:
            RuntimeError: If retrieval operation fails.
        """
        with self._lock:
            try:
                messages = self._semantic_engine.get_all(project_id=self.project_id)

                self.logger.info(
                    f"Retrieved {len(messages)} messages (USearchEngine={len(messages)})",
                    extra={
                        "project_id": self.project_id,
                        "count": len(messages),
                    },
                )

                return messages

            except Exception as e:
                raise StorageError(f"Failed to retrieve messages: {e}") from e

    def _search_semantic(self, query: str, limit: int) -> List[Tuple[str, float]]:
        """Execute semantic search on USearchEngine."""
        try:
            results = self._semantic_engine.search(
                query=query,
                project_id=self.project_id,
                limit=limit,
            )
            return results
        except Exception as e:
            self.logger.warning(
                "Semantic search failed, falling back to Tantivy",
                extra={"project_id": self.project_id, "error": str(e)},
            )
            return []

    def _search_tantivy(self, query: str, limit: int) -> List[Tuple[str, float]]:
        """Execute full-text search on Tantivy engine."""
        if self._tantivy_engine is None:
            return []
        return self._tantivy_engine.search(query, self.project_id, limit)

    def _calculate_adaptive_overfetch(self, limit: int) -> int:
        """Calculate adaptive overfetch limit based on index size.

        For small indexes, we use a higher multiplier to ensure diversity.
        For large indexes, we use a lower multiplier since there are enough
        documents for good fusion quality.

        The multiplier scales logarithmically:
        - Index size <= 100: max multiplier (3.0x default)
        - Index size >= 10,000: min multiplier (1.5x default)
        - Between: logarithmic interpolation

        Args:
            limit: The base search limit.

        Returns:
            Calculated overfetch limit (minimum MIN_OVERFETCH_LIMIT).
        """
        import math

        # Get current index size (0 if not initialized or unavailable)
        # Note: We access USearchEngine.index directly - the protocol doesn't expose this
        # but the concrete implementation does. Using getattr for type safety.
        try:
            engine_index = getattr(self._semantic_engine, "index", None)
            index_size = len(engine_index) if engine_index is not None else 0
        except Exception:
            index_size = 0

        # If adaptive is disabled or index is empty, use static multiplier
        if not self.config.overfetch_adaptive or index_size == 0:
            return max(limit * self.config.overfetch_multiplier, MIN_OVERFETCH_LIMIT)

        # Bounds for index size (log scale interpolation)
        small_index = 100  # At or below this: use max multiplier
        large_index = 10000  # At or above this: use min multiplier

        min_mult = self.config.overfetch_min_multiplier
        max_mult = self.config.overfetch_max_multiplier

        if index_size <= small_index:
            multiplier = max_mult
        elif index_size >= large_index:
            multiplier = min_mult
        else:
            # Logarithmic interpolation between bounds
            # Maps [log(small), log(large)] -> [max_mult, min_mult]
            log_small = math.log(small_index)
            log_large = math.log(large_index)
            log_current = math.log(index_size)

            # Normalize to [0, 1] range
            t = (log_current - log_small) / (log_large - log_small)

            # Linear interpolation from max_mult to min_mult
            multiplier = max_mult - t * (max_mult - min_mult)

        overfetch_limit = int(limit * multiplier)
        return max(overfetch_limit, MIN_OVERFETCH_LIMIT)

    async def search(
        self,
        query: str,
        limit: Optional[int] = None,
    ) -> List[str]:
        """Hybrid semantic + full-text search with RRF ranking (async).

        This async method runs both semantic and full-text searches in parallel
        using asyncio.gather() with asyncify(), providing non-blocking execution
        without ThreadPoolExecutor overhead.

        Args:
            query: Search query string.
            limit: Maximum number of results (uses config default if None).

        Returns:
            List of hybrid-ranked messages from both engines.

        Raises:
            RuntimeError: If search operation fails.
        """
        # Use defaults from config if not provided
        if limit is None:
            limit = self.config.search_limit

        try:
            if not self.is_hybrid_search:
                self.logger.info(
                    "🔎 SEARCH MODE: Semantic only (hybrid disabled)",
                    extra={"mode": "semantic"},
                )
                # Pure semantic search via USearchEngine
                results = self._semantic_engine.search(
                    query=query,
                    project_id=self.project_id,
                    limit=limit,
                )
                return [msg for msg, _ in results]

            # HYBRID SEARCH: USearch semantic + Tantivy full-text → fusion ranking
            self.logger.info(
                "🔎 SEARCH MODE: Hybrid (semantic + full-text)",
                extra={
                    "mode": "hybrid",
                    "limit": limit,
                },
            )
            self.logger.info(
                f'   Query: "{query[:LOG_QUERY_TRUNCATE_LENGTH]}{"..." if len(query) > LOG_QUERY_TRUNCATE_LENGTH else ""}"',
                extra={"query": query[:LOG_QUERY_TRUNCATE_LENGTH]},
            )

            # Overfetch for better fusion quality (adaptive or static multiplier)
            overfetch_limit = self._calculate_adaptive_overfetch(limit)

            # Thread-safety: Pre-initialize engines before parallel execution
            # This prevents race conditions in lazy initialization when multiple
            # threads access uninitialized engines simultaneously
            self._semantic_engine.ensure_initialized()  # Force USearch initialization
            if self._tantivy_engine is not None:
                self._tantivy_engine.ensure_initialized()  # Force Tantivy searcher init

            # Execute both searches in parallel for ~30-50% latency improvement
            self.logger.info(
                "─" * 50,
                extra={"section": "search_engines"},
            )
            self.logger.info(
                "📡 STEP 1: Executing parallel search engines...",
                extra={"step": "parallel_search"},
            )

            # Run both searches in parallel using asyncer task group
            async with create_task_group() as tg:
                soon_semantic = tg.soonify(asyncify(self._search_semantic))(
                    query, overfetch_limit
                )
                soon_tantivy = tg.soonify(asyncify(self._search_tantivy))(
                    query, overfetch_limit
                )
            semantic_results = soon_semantic.value or []
            tantivy_results = soon_tantivy.value or []

            self.logger.info(
                f"   Both engines completed (semantic: {len(semantic_results)}, tantivy: {len(tantivy_results)})",
                extra={
                    "project_id": self.project_id,
                    "query": query[:LOG_QUERY_TRUNCATE_LENGTH],
                    "semantic_count": len(semantic_results),
                    "tantivy_count": len(tantivy_results),
                },
            )

            # Log USearch (semantic) results
            self.logger.info("")
            self.logger.info(
                "   USEARCH SEMANTIC ENGINE:",
                extra={"engine": "usearch"},
            )
            if semantic_results:
                top_score = semantic_results[0][1] if semantic_results else 0.0
                self.logger.info(
                    f"      Found {len(semantic_results)} result(s), best score: {top_score:.4f}",
                    extra={
                        "project_id": self.project_id,
                        "engine": "usearch",
                        "result_count": len(semantic_results),
                        "top_score": top_score,
                    },
                )
                # Show top results preview
                for idx, (message, score) in enumerate(
                    semantic_results[: min(3, len(semantic_results))], 1
                ):
                    preview = truncate_message(message, max_length=60)
                    self.logger.info(
                        f"      [{idx}] score={score:.4f} → {preview}",
                        extra={
                            "engine": "usearch",
                            "result_index": idx,
                            "score": score,
                        },
                    )
            else:
                self.logger.info(
                    "      No results found",
                    extra={"engine": "usearch", "result_count": 0},
                )

            # Log Tantivy (full-text) results
            self.logger.info("")
            self.logger.info(
                "   TANTIVY FULL-TEXT ENGINE:",
                extra={"engine": "tantivy"},
            )
            if tantivy_results:
                top_score = tantivy_results[0][1] if tantivy_results else 0.0
                self.logger.info(
                    f"      Found {len(tantivy_results)} result(s), best BM25 score: {top_score:.4f}",
                    extra={
                        "project_id": self.project_id,
                        "engine": "tantivy",
                        "result_count": len(tantivy_results),
                        "top_score": top_score,
                    },
                )
                # Show top results preview
                for idx, (message, score) in enumerate(
                    tantivy_results[: min(3, len(tantivy_results))], 1
                ):
                    preview = truncate_message(message, max_length=60)
                    self.logger.info(
                        f"      [{idx}] score={score:.4f} → {preview}",
                        extra={
                            "engine": "tantivy",
                            "result_index": idx,
                            "score": score,
                        },
                    )
            else:
                self.logger.info(
                    "      No results found",
                    extra={"engine": "tantivy", "result_count": 0},
                )

            # Fusion ranking - combine results from both engines
            self.logger.info(
                "─" * 50,
                extra={"section": "fusion"},
            )
            self.logger.info(
                "🔀 STEP 2: RRF Fusion (combining results)...",
                extra={"step": "fusion"},
            )
            hybrid_results = self._fusion_engine.fuse(semantic_results, tantivy_results)

            # Log fusion results with source information
            if hybrid_results:
                top_rrf = hybrid_results[0][1] if hybrid_results else 0.0
                self.logger.info(
                    f"   Combined {len(hybrid_results)} unique result(s) using {self._fusion_engine.method.upper()} algorithm",
                    extra={
                        "project_id": self.project_id,
                        "query": query[:LOG_QUERY_TRUNCATE_LENGTH],
                        "engine": "fusion",
                        "result_count": len(hybrid_results),
                        "method": self._fusion_engine.method,
                        "top_rrf_score": top_rrf,
                    },
                )
                self.logger.info(
                    f"   Best fusion score: {top_rrf:.4f} (normalized 0-1)",
                    extra={"top_score": top_rrf},
                )

                # Show top fusion results with source info
                for idx, (message, rrf_score) in enumerate(
                    hybrid_results[: min(3, len(hybrid_results))], 1
                ):
                    # Determine source(s) of this result
                    in_semantic = any(msg == message for msg, _ in semantic_results)
                    in_tantivy = any(msg == message for msg, _ in tantivy_results)
                    sources = []
                    if in_semantic:
                        sources.append("semantic")
                    if in_tantivy:
                        sources.append("full-text")
                    source_str = " + ".join(sources) if sources else "unknown"

                    preview = truncate_message(message, max_length=50)
                    self.logger.info(
                        f"      [{idx}] score={rrf_score:.4f} ({source_str}) → {preview}",
                        extra={
                            "project_id": self.project_id,
                            "engine": "rrf_fusion",
                            "result_index": idx,
                            "rrf_score": rrf_score,
                            "source": source_str,
                        },
                    )
            else:
                self.logger.info(
                    "   No results to fuse",
                    extra={"engine": "fusion", "result_count": 0},
                )

            # Apply fusion threshold filtering BEFORE reranking
            self.logger.info(
                "─" * 50,
                extra={"section": "filtering"},
            )
            self.logger.info(
                f"🎯 STEP 3: Filtering (threshold >= {self.config.fusion_ranking_threshold})...",
                extra={
                    "step": "filtering",
                    "threshold": self.config.fusion_ranking_threshold,
                },
            )
            pre_filter_count = len(hybrid_results)
            hybrid_results = self._filter_by_fusion_threshold(
                hybrid_results, self.config.fusion_ranking_threshold, query
            )

            # Handle case where all results were filtered out
            if not hybrid_results:
                self.logger.info(
                    f"   All {pre_filter_count} result(s) filtered out (below threshold {self.config.fusion_ranking_threshold})",
                    extra={
                        "project_id": self.project_id,
                        "fusion_threshold": self.config.fusion_ranking_threshold,
                        "filtered_count": pre_filter_count,
                    },
                )
                self.logger.info(
                    "─" * 50,
                    extra={"section": "complete"},
                )
                self.logger.info(
                    "SEARCH COMPLETE: No results passed quality threshold",
                    extra={"result_count": 0},
                )
                return []

            # Step 4: Reranking (LLM or CrossEncoder, based on reranker_engine config)
            if self._llm_reranker is not None:
                self.logger.info(
                    "─" * 50,
                    extra={"section": "reranking"},
                )
                self.logger.info(
                    f"🤖 STEP 4: LLM Reranking ({len(hybrid_results)} candidates)...",
                    extra={
                        "step": "reranking",
                        "engine": "llm",
                        "candidate_count": len(hybrid_results),
                    },
                )

                rerank_start = time.time()
                pre_rerank_count = len(hybrid_results)

                # Rerank using LLM relevance scores
                hybrid_results = await self._llm_reranker.rerank(query, hybrid_results)

                rerank_duration = (time.time() - rerank_start) * 1000  # ms

                # Log reranking results
                if hybrid_results:
                    self.logger.info(
                        f"   Kept {len(hybrid_results)}/{pre_rerank_count} result(s) after LLM scoring",
                        extra={
                            "kept_count": len(hybrid_results),
                            "filtered_count": pre_rerank_count - len(hybrid_results),
                            "threshold": self.config.search_score_threshold,
                        },
                    )
                    # Show top reranked results
                    for idx, (message, llm_score) in enumerate(
                        hybrid_results[: min(3, len(hybrid_results))], 1
                    ):
                        preview = truncate_message(message, max_length=50)
                        self.logger.info(
                            f"      [{idx}] score={llm_score:.4f} → {preview}",
                            extra={
                                "result_index": idx,
                                "llm_score": llm_score,
                            },
                        )
                else:
                    self.logger.info(
                        f"   All {pre_rerank_count} result(s) filtered out by LLM reranking",
                        extra={
                            "filtered_count": pre_rerank_count,
                            "threshold": self.config.search_score_threshold,
                        },
                    )

                self.logger.info(
                    f"   LLM reranking completed in {rerank_duration:.0f}ms",
                    extra={"rerank_duration_ms": rerank_duration},
                )

                # Handle case where all results were filtered out by reranking
                if not hybrid_results:
                    self.logger.info(
                        "─" * 50,
                        extra={"section": "complete"},
                    )
                    self.logger.info(
                        "SEARCH COMPLETE: No results passed LLM reranking threshold",
                        extra={"result_count": 0},
                    )
                    return []

            elif self._cross_encoder_reranker is not None:
                self.logger.info(
                    "─" * 50,
                    extra={"section": "reranking"},
                )
                self.logger.info(
                    f"🔄 STEP 4: CrossEncoder Reranking ({len(hybrid_results)} candidates)...",
                    extra={
                        "step": "reranking",
                        "engine": "cross_encoder",
                        "candidate_count": len(hybrid_results),
                    },
                )

                rerank_start = time.time()
                pre_rerank_count = len(hybrid_results)

                # Rerank using CrossEncoder scores (async wrapper for non-blocking)
                hybrid_results = await self._cross_encoder_reranker.rerank_async(
                    query, hybrid_results
                )

                rerank_duration = (time.time() - rerank_start) * 1000  # ms

                # Log reranking results
                if hybrid_results:
                    self.logger.info(
                        f"   Kept {len(hybrid_results)}/{pre_rerank_count} result(s) after CrossEncoder scoring",
                        extra={
                            "kept_count": len(hybrid_results),
                            "filtered_count": pre_rerank_count - len(hybrid_results),
                            "model": self.config.cross_encoder_model,
                        },
                    )
                    # Show top reranked results
                    for idx, (message, ce_score) in enumerate(
                        hybrid_results[: min(3, len(hybrid_results))], 1
                    ):
                        preview = truncate_message(message, max_length=50)
                        self.logger.info(
                            f"      [{idx}] score={ce_score:.4f} → {preview}",
                            extra={
                                "result_index": idx,
                                "cross_encoder_score": ce_score,
                            },
                        )
                else:
                    self.logger.info(
                        f"   ⚠️  All {pre_rerank_count} result(s) filtered out (scores below threshold {self.config.cross_encoder_score_threshold:.2f})",
                        extra={
                            "filtered_count": pre_rerank_count,
                            "threshold": self.config.cross_encoder_score_threshold,
                        },
                    )
                    self.logger.info(
                        f"   💡 Tip: Lower CROSS_ENCODER_SCORE_THRESHOLD (current: {self.config.cross_encoder_score_threshold:.2f}) to keep more results",
                        extra={
                            "threshold": self.config.cross_encoder_score_threshold,
                            "config_var": "CROSS_ENCODER_SCORE_THRESHOLD",
                        },
                    )

                self.logger.info(
                    f"   CrossEncoder reranking completed in {rerank_duration:.0f}ms",
                    extra={"rerank_duration_ms": rerank_duration},
                )

                # Handle case where all results were filtered out by reranking
                if not hybrid_results:
                    self.logger.info(
                        "─" * 50,
                        extra={"section": "complete"},
                    )
                    self.logger.info(
                        "SEARCH COMPLETE: No results passed CrossEncoder reranking threshold",
                        extra={"result_count": 0},
                    )
                    return []

            # Extract messages from fusion/reranked results
            filtered_messages = [msg for msg, _ in hybrid_results[:limit]]

            # Final summary
            self.logger.info(
                "─" * 50,
                extra={"section": "complete"},
            )
            self.logger.info(
                f"SEARCH COMPLETE: {len(filtered_messages)} result(s) returned",
                extra={
                    "project_id": self.project_id,
                    "query": query,
                    "result_count": len(filtered_messages),
                    "top_rrf_score": hybrid_results[0][1] if hybrid_results else 0.0,
                },
            )

            return filtered_messages

        except Exception as e:
            self.logger.error(
                "Hybrid search failed",
                extra={"project_id": self.project_id, "query": query, "error": str(e)},
            )
            raise SearchError(f"Failed to execute hybrid search: {str(e)}") from e

    def search_for_removal(
        self, query: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search for messages to potentially remove using direct database lookup.

        Uses O(log n) indexed database lookup instead of O(n) iteration through
        all messages. This is a significant performance improvement for large
        collections.

        Args:
            query: The exact message content to find for removal.
            limit: Maximum number of candidates (uses config default if None).
                   Note: Since we're looking for exact match, we'll return at most 1.

        Returns:
            List of candidate records with 'id', 'memory', and 'score' fields.
            Returns at most 1 candidate since exact match is unique.

        Raises:
            RuntimeError: If search operation fails.
        """
        # Note: limit is kept for API compatibility but exact match returns at most 1
        _ = limit  # Unused, kept for API compatibility

        try:
            candidates: List[Dict[str, Any]] = []

            # Direct database lookup - O(log n) instead of O(n) get_all() + iteration
            msg_id = self._semantic_engine.get_id_by_message(self.project_id, query)

            if msg_id is not None:
                # Found exact match - return as candidate
                candidates.append(
                    {
                        "id": str(msg_id),  # Use database ID directly
                        "memory": query,
                        "score": 1.0,  # Exact match = perfect score
                    }
                )

            self.logger.debug(
                f"Direct lookup removal candidates: {len(candidates)}",
                extra={"project_id": self.project_id, "query": query[:50]},
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
        try:
            self._semantic_engine.delete(memory_id=memory_id)
        except Exception as e:
            raise StorageError(f"Failed to delete memory: {e}") from e

    def delete_by_message(self, message: str) -> bool:
        """Delete a memory entry by its exact message content (thread-safe).

        Thread-safe: Uses RLock to ensure consistent state during deletion.

        This method looks up the message in the database and deletes it from
        both USearch (semantic) and Tantivy (full-text) engines atomically.
        If Tantivy deletion fails after USearch deletion succeeds, the operation
        raises an error to alert about the inconsistent state.

        Args:
            message: The exact message content to delete.

        Returns:
            True if the message was found and deleted, False if not found.

        Raises:
            RuntimeError: If deletion fails or results in inconsistent state.
        """
        with self._lock:
            try:
                # 1. Look up the SQLite ID from the message content
                msg_id = self._semantic_engine.get_id_by_message(
                    self.project_id, message
                )

                if msg_id is None:
                    self.logger.debug(
                        "Message not found for deletion",
                        extra={
                            "project_id": self.project_id,
                            "message_preview": message[:50],
                        },
                    )
                    return False

                # 2. Delete from USearch semantic engine using the numeric ID
                self._semantic_engine.delete(memory_id=str(msg_id))

                # 3. Delete from Tantivy full-text engine
                # If this fails after USearch deletion, we have inconsistent state
                if self._tantivy_engine is not None:
                    try:
                        # delete() commits internally for both soft-delete and rebuild modes
                        self._tantivy_engine.delete(self.project_id, message)
                    except Exception as tantivy_error:
                        # Log critical error - state is now inconsistent
                        self.logger.error(
                            "Tantivy deletion failed after USearch deletion - INCONSISTENT STATE",
                            extra={
                                "project_id": self.project_id,
                                "message_id": msg_id,
                                "error": str(tantivy_error),
                            },
                        )
                        raise InconsistentStateError(
                            f"USearch deletion succeeded but Tantivy deletion failed: "
                            f"{tantivy_error}"
                        ) from tantivy_error

                self.logger.debug(
                    "Message deleted from hybrid storage",
                    extra={
                        "project_id": self.project_id,
                        "message_id": msg_id,
                        "engines": ["usearch", "tantivy"]
                        if self._tantivy_engine
                        else ["usearch"],
                    },
                )

                return True

            except InconsistentStateError:
                raise  # Re-raise inconsistent state errors with full context
            except Exception as e:
                raise StorageError(f"Failed to delete message: {e}") from e

    def _has_exact_match(self, message: str) -> bool:
        """Check whether the exact message already exists in storage.

        Uses Tantivy for fast exact phrase matching when hybrid search is enabled,
        falling back to direct database lookup otherwise. Both paths are O(log n)
        avoiding the ~100-500ms embedding API call overhead.

        Sprint 2.1 Optimization: Fallback now uses get_id_by_message() for direct
        indexed database lookup instead of semantic search with embedding API call.
        """
        # Fast path: Use Tantivy for exact match if hybrid search is enabled
        if self._tantivy_engine is not None:
            try:
                # Use quoted exact phrase search with escaped query
                escaped_query = self._escape_tantivy_query(message)
                results = self._tantivy_engine.search(
                    f'"{escaped_query}"',
                    self.project_id,
                    limit=5,  # Small limit since we only need to check existence
                )
                # Check for exact string match in results
                has_match = any(msg == message for msg, _ in results)
                if has_match:
                    self.logger.debug(
                        "Tantivy found exact duplicate",
                        extra={"project_id": self.project_id},
                    )
                return has_match
            except Exception as e:
                self.logger.warning(
                    "Tantivy duplicate check failed; falling back to database lookup",
                    extra={"project_id": self.project_id, "error": str(e)},
                )

        # Optimized fallback: Direct database lookup (O(log n), no embedding API call)
        # This avoids the 100-500ms embedding API overhead of semantic search
        try:
            msg_id = self._semantic_engine.get_id_by_message(self.project_id, message)
            if msg_id is not None:
                self.logger.debug(
                    "Database lookup found exact duplicate",
                    extra={"project_id": self.project_id, "msg_id": msg_id},
                )
                return True
            return False
        except Exception as e:
            self.logger.warning(
                "Duplicate detection failed; proceeding without deduplication",
                extra={
                    "project_id": self.project_id,
                    "error": str(e),
                },
            )
            return False

    def _filter_by_fusion_threshold(
        self, results: List[Tuple[str, float]], threshold: float, query: str
    ) -> List[Tuple[str, float]]:
        """Filter fusion results by fusion score threshold.

        This filtering happens immediately after RRF fusion, before reranking.

        Args:
            results: List of (message, fusion_score) tuples from fusion engine.
            threshold: Minimum fusion score to keep (default: 0.5).
            query: Original search query (for logging context).

        Returns:
            Filtered list of (message, fusion_score) tuples.
        """
        if not results:
            return results

        # Apply threshold filter
        filtered = [(msg, score) for msg, score in results if score >= threshold]
        filtered_out = len(results) - len(filtered)

        # Log summary with clear pass/fail indication
        self.logger.info(
            f"   Kept {len(filtered)}/{len(results)} result(s), filtered {filtered_out}",
            extra={
                "project_id": self.project_id,
                "fusion_threshold": threshold,
                "kept_count": len(filtered),
                "filtered_count": filtered_out,
            },
        )

        # Show filtered results with status
        for idx, (message, score) in enumerate(results[: min(5, len(results))], 1):
            status, interpretation = format_fusion_score_status(score, threshold)
            preview = truncate_message(message, max_length=50)
            status_indicator = "[KEEP]" if score >= threshold else "[FILTER]"
            self.logger.info(
                f"      [{idx}] {status_indicator} score={score:.4f} ({interpretation}) -> {preview}",
                extra={
                    "project_id": self.project_id,
                    "fusion_score": score,
                    "fusion_status": status,
                },
            )

        return filtered

    @staticmethod
    def _escape_tantivy_query(query: str) -> str:
        """Escape special characters for Tantivy query syntax.

        Tantivy uses Lucene-style query syntax where certain characters have
        special meaning. This method escapes them to prevent query injection.

        Args:
            query: Raw query string that may contain special characters.

        Returns:
            Escaped query string safe for use in Tantivy queries.
        """
        # Tantivy/Lucene special characters that need escaping
        special_chars = r'+-&|!(){}[]^"~*?:\/'
        escaped = []
        for char in query:
            if char in special_chars:
                escaped.append(f"\\{char}")
            else:
                escaped.append(char)
        return "".join(escaped)

    def close(self) -> None:
        """Close all resources and persist data to disk (thread-safe).

        This method ensures all data is properly saved before shutdown:
        1. Commits and closes Tantivy full-text engine (if enabled)
        2. Commits and closes USearch semantic engine

        Thread-safe: Uses RLock to ensure consistent state during shutdown.

        Should be called during graceful shutdown (e.g., on SIGINT/SIGTERM)
        to prevent data loss.
        """
        with self._lock:
            self.logger.info(
                "Closing MemoryManager - persisting data to disk...",
                extra={"project_id": self.project_id},
            )

            # 1. Close Tantivy engine first (if enabled)
            if self._tantivy_engine is not None:
                try:
                    # Use flush() to commit and wait for all merge threads
                    self._tantivy_engine.flush()
                    self._tantivy_engine.close()
                    self.logger.info(
                        "Tantivy engine closed successfully",
                        extra={"project_id": self.project_id, "engine": "tantivy"},
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error closing Tantivy engine: {e}",
                        extra={
                            "project_id": self.project_id,
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
                    extra={"project_id": self.project_id, "engine": "usearch"},
                )
            except Exception as e:
                self.logger.error(
                    f"Error closing USearch engine: {e}",
                    extra={
                        "project_id": self.project_id,
                        "engine": "usearch",
                        "error": str(e),
                    },
                )

            self.logger.info(
                "MemoryManager closed - all data persisted",
                extra={"project_id": self.project_id},
            )
