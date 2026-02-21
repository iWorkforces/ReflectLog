'''Add operation phases for the 3-phase parallel add pipeline.

This module extracts the add operation logic from MemoryManager into
separate phase classes. The 3 phases are:
1. Duplicate Detection (parallel)
2. Smart Replacement Detection (parallel)
3. Sequential Storage

Each phase is implemented as a separate class that takes inputs and
produces outputs for the next phase.
'''

from dataclasses import dataclass, field
import threading
import time
from typing import Any

import anyio
from asyncer import asyncify, create_task_group

from ..config import Config
from ..exceptions import StorageError
from ..types import ISemanticSearchEngine
from ..utils import StructuredLogger, truncate_memory
from .match_utils import has_exact_match


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
    '''Result of adding memories to memory storage.

    Provides detailed information about what happened during the add operation,
    including which memories were stored, skipped (duplicates), and replaced.

    Attributes:
        stored_count: Number of memories successfully stored.
        skipped_count: Number of memories skipped (duplicates).
        replaced_count: Number of existing memories that were replaced.
        replacements: Detailed info about each replacement that occurred.
    '''

    stored_count: int = 0
    skipped_count: int = 0
    replaced_count: int = 0
    replacements: list[ReplacementInfo] = field(default_factory=list)


@dataclass
class Phase1Result:
    '''Result of Phase 1: Duplicate Detection.

    Attributes:
        unique_memories: Memories that are unique within the batch.
        storage_duplicates: Memories that already exist in storage.
        batch_duplicates_count: Count of duplicates within the batch.
        duration: Time taken for phase 1 execution.
    '''

    unique_memories: list[str]
    storage_duplicates: list[str]
    batch_duplicates_count: int
    duration: float


@dataclass
class Phase2Result:
    '''Result of Phase 2: Smart Replacement Detection.

    Attributes:
        replacement_map: Mapping from memory to list of ReplacementInfo objects.
        total_replacements: Total number of replacements detected.
        duration: Time taken for phase 2 execution.
    '''

    replacement_map: dict[str, list[ReplacementInfo]]
    total_replacements: int
    duration: float


@dataclass
class Phase3Result:
    '''Result of Phase 3: Sequential Storage.

    Attributes:
        stored_count: Number of memories stored.
        replaced_count: Number of memories replaced.
        replacements: List of ReplacementInfo objects.
        duration: Time taken for phase 3 execution.
    '''

    stored_count: int
    replaced_count: int
    replacements: list[ReplacementInfo]
    duration: float


class DuplicateDetectionPhase:
    '''Phase 1: Parallel duplicate detection.

    This phase:
    1. Deduplicates within the batch itself
    2. Checks against existing storage in parallel
    3. Returns unique memories and storage duplicates
    '''

    def __init__(
        self,
        semantic_engine: ISemanticSearchEngine,
        tantivy_engine: Any,  # TantivyEngine | None
        config: Config,
        logger: StructuredLogger,
    ):
        '''Initialize duplicate detection phase.

        Args:
            semantic_engine: USearchEngine for semantic search and direct database lookup.
            tantivy_engine: TantivyEngine for fast exact phrase matching.
            config: Application configuration.
            logger: Structured logger instance.
        '''
        self._semantic_engine = semantic_engine
        self._tantivy_engine = tantivy_engine
        self.config = config
        self.logger = logger
        self._project_id = config.project_id

    async def execute(self, memories: list[str]) -> Phase1Result:
        '''Execute Phase 1: Parallel duplicate detection.

        Args:
            memories: List of memories to check for duplicates.

        Returns:
            Phase1Result with unique memories and duplicate information.
        '''
        phase_start = time.perf_counter()

        # Step 1: Deduplicate within the batch itself (preserve order, keep first)
        seen: dict[str, int] = {}
        unique_memories: list[str] = []
        batch_duplicate_indices: list[int] = []

        for idx, memory in enumerate(memories):
            if memory in seen:
                batch_duplicate_indices.append(idx)
            else:
                seen[memory] = idx
                unique_memories.append(memory)

        batch_duplicates_count = len(batch_duplicate_indices)
        if batch_duplicates_count > 0:
            self.logger.info(
                f'Phase 1: Found {batch_duplicates_count} duplicates within batch',
                extra={
                    'batch_duplicates': batch_duplicates_count,
                    'unique_count': len(unique_memories),
                },
            )

        # Step 2: Parallel duplicate detection against existing storage
        duplicate_flags: dict[str, bool] = {}
        semaphore = anyio.Semaphore(self.config.add_max_concurrency)

        async def check_duplicate(memory: str) -> tuple[str, bool]:
            '''Check if memory is duplicate (with semaphore for concurrency control).'''
            async with semaphore:
                is_dup = await asyncify(self._has_exact_match)(memory)
                return (memory, is_dup)

        # Run all duplicate checks in parallel
        results: list[tuple[str, bool]] = []
        async with create_task_group() as tg:

            async def collect_result(memory: str) -> None:
                res = await check_duplicate(memory)
                results.append(res)

            for memory in unique_memories:
                _ = tg.soonify(collect_result)(memory)

        # Process results
        for memory, is_dup in results:
            duplicate_flags[memory] = is_dup

        # Separate duplicates from non-duplicates
        storage_duplicates: list[str] = []
        non_duplicate_memories: list[str] = []

        for memory in unique_memories:
            if duplicate_flags.get(memory, False):
                storage_duplicates.append(memory)
            else:
                non_duplicate_memories.append(memory)

        duration = time.perf_counter() - phase_start
        self.logger.info(
            f'Phase 1 complete: {len(non_duplicate_memories)} unique, '
            f'{len(storage_duplicates)} storage duplicates ({duration:.3f}s)',
            extra={
                'phase': 1,
                'duration_ms': int(duration * 1000),
                'unique_count': len(non_duplicate_memories),
                'storage_duplicates': len(storage_duplicates),
                'batch_duplicates': batch_duplicates_count,
            },
        )

        return Phase1Result(
            unique_memories=non_duplicate_memories,
            storage_duplicates=storage_duplicates,
            batch_duplicates_count=batch_duplicates_count,
            duration=duration,
        )

    def _has_exact_match(self, content: str) -> bool:
        '''Check whether the exact memory already exists in storage.

        Uses Tantivy for fast exact phrase matching when hybrid search is enabled,
        falling back to direct database lookup otherwise. Both paths are O(log n)
        avoiding the ~100-500ms embedding API call overhead.

        Sprint 2.1 Optimization: Fallback now uses get_id_by_content() for direct
        indexed database lookup instead of semantic search with embedding API call.
        '''
        return has_exact_match(
            semantic_engine=self._semantic_engine,
            tantivy_engine=self._tantivy_engine,
            project_id=self._project_id,
            content=content,
            logger=self.logger,
        )


class SmartReplacementPhase:
    '''Phase 2: Parallel smart replacement detection.

    This phase:
    1. Finds similar existing memories via semantic search
    2. Filters by similarity threshold
    3. Checks candidates in parallel with LLM
    4. Returns replacement map
    '''

    def __init__(
        self,
        semantic_engine: ISemanticSearchEngine,
        config: Config,
        logger: StructuredLogger,
        memory_manager: Any,  # MemoryManager for lazy SmartReplacer fetching
    ):
        '''Initialize smart replacement phase.

        Args:
            semantic_engine: USearchEngine for semantic search.
            config: Application configuration.
            logger: Structured logger instance.
            memory_manager: MemoryManager instance for lazy SmartReplacer fetching.
        '''
        self._semantic_engine = semantic_engine
        self.config = config
        self.logger = logger
        self._project_id = config.project_id
        self._memory_manager = memory_manager

    def _get_smart_replacer(self) -> Any:
        '''Get SmartReplacer (lazy loading via memory_manager).

        Returns:
            SmartReplacer instance if configured, None otherwise.
        '''
        if self._memory_manager is None:
            return None
        return self._memory_manager.smart_replacer

    @property
    def smart_replace_enabled(self) -> bool:
        '''Check if smart replacement is configured and available.

        Returns:
            True if SmartReplacer is configured, False otherwise.
        '''
        return self._get_smart_replacer() is not None

    async def execute(self, memories: list[str]) -> Phase2Result:
        '''Execute Phase 2: Parallel smart replacement detection.

        Args:
            memories: List of non-duplicate memories to check for replacements.

        Returns:
            Phase2Result with replacement information.
        '''
        phase_start = time.perf_counter()

        # Map: memory -> list of replacement infos
        replacement_map: dict[str, list[ReplacementInfo]] = {}

        # Get SmartReplacer (lazy loaded via memory_manager if available)
        smart_replacer = self._get_smart_replacer()

        if smart_replacer is None or not memories:
            return Phase2Result(
                replacement_map=replacement_map,
                total_replacements=0,
                duration=time.perf_counter() - phase_start,
            )

        # Capture in local variable for type narrowing in nested function
        smart_replacer = smart_replacer

        # Limit concurrent replacement checks to avoid overload
        semaphore = anyio.Semaphore(self.config.add_max_concurrency)

        replacement_results: list[tuple[str, list[ReplacementInfo]]] = []
        async with create_task_group() as tg:

            async def check_replacement_for_memory(memory: str) -> None:
                '''Check replacement for a single memory.'''
                async with semaphore:
                    infos = await self._check_for_replacement(memory, smart_replacer)
                replacement_results.append((memory, infos))

            for memory in memories:
                _ = tg.soonify(check_replacement_for_memory)(memory)

        # Collect results
        for memory, infos in replacement_results:
            replacement_map[memory] = infos

        duration = time.perf_counter() - phase_start
        total_replacements = sum(len(infos) for infos in replacement_map.values())

        self.logger.info(
            f'Phase 2 complete: {total_replacements} replacements detected ({duration:.3f}s)',
            extra={
                'phase': 2,
                'duration_ms': int(duration * 1000),
                'memories_checked': len(memories),
                'total_replacements': total_replacements,
            },
        )

        return Phase2Result(
            replacement_map=replacement_map,
            total_replacements=total_replacements,
            duration=duration,
        )

    async def _check_for_replacement(
        self, new_memory: str, smart_replacer: Any
    ) -> list[ReplacementInfo]:
        '''Check if new memory should replace existing memories.

        Uses semantic search to find the most similar existing memories,
        applies similarity pre-filter, then uses LLM to determine which
        should be replaced.

        Args:
            new_memory: The new memory being added.
            smart_replacer: SmartReplacer instance.

        Returns:
            List of ReplacementInfo objects for memories that should be replaced.
            Empty list if no replacements needed.
        '''

        try:
            # Step 1: Find top N most similar memories via semantic search
            candidate_limit = self.config.smart_replace_candidate_limit
            similar_results = await asyncify(self._semantic_engine.search)(
                query=new_memory,
                project_id=self._project_id,
                limit=candidate_limit,
            )

            if not similar_results:
                self.logger.debug(
                    'No similar memories found for replacement check',
                    extra={
                        'project_id': self._project_id,
                        'new_memory_preview': new_memory[:100],
                    },
                )
                return []

            # Step 2: Filter candidates by similarity threshold (pre-filter)
            min_similarity = self.config.smart_replace_min_similarity
            filtered_candidates = [
                (mem, score)
                for mem, score, _ in similar_results
                if score >= min_similarity and mem != new_memory
            ]

            if not filtered_candidates:
                self.logger.debug(
                    f'All candidates below similarity threshold ({min_similarity})',
                    extra={
                        'project_id': self._project_id,
                        'candidate_count': len(similar_results),
                        'min_similarity': min_similarity,
                    },
                )
                return []

            self.logger.debug(
                f'Found {len(filtered_candidates)} candidates above similarity threshold',
                extra={
                    'project_id': self._project_id,
                    'candidate_count': len(filtered_candidates),
                    'min_similarity': min_similarity,
                    'top_similarity': filtered_candidates[0][1]
                    if filtered_candidates
                    else 0,
                },
            )

            # Step 3: Check candidates in parallel with LLM (rate-limited)
            # Reuse rerank_max_concurrency for API rate limiting
            semaphore = anyio.Semaphore(self.config.rerank_max_concurrency)

            async def check_single_candidate(
                existing_memory: str, similarity_score: float
            ) -> ReplacementInfo | None:
                '''Check a single candidate for replacement (with semaphore).'''
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
                                f'Smart replacement triggered (confidence={confidence:.2f})',
                                extra={
                                    'project_id': self._project_id,
                                    'should_replace': True,
                                    'confidence': confidence,
                                    'similarity_score': similarity_score,
                                    'reason': reason,
                                    'old_memory_preview': truncate_memory(
                                        existing_memory, max_length=60
                                    ),
                                    'new_memory_preview': truncate_memory(
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
                                f'No replacement needed (confidence={confidence:.2f}): {reason}',
                                extra={
                                    'project_id': self._project_id,
                                    'should_replace': False,
                                    'confidence': confidence,
                                    'reason': reason,
                                },
                            )
                            return None
                    except Exception as candidate_error:
                        # Graceful degradation: log warning and skip this candidate
                        self.logger.warning(
                            f'LLM check failed for candidate: {candidate_error}',
                            extra={
                                'project_id': self._project_id,
                                'error': str(candidate_error),
                                'existing_preview': existing_memory[:100],
                            },
                        )
                        return None

            # Run all candidate checks in parallel
            results: list[ReplacementInfo | None] = []

            async def collect_result(
                existing_memory: str, similarity_score: float
            ) -> None:
                '''Run check and collect result.'''
                result = await check_single_candidate(existing_memory, similarity_score)
                results.append(result)

            async with anyio.create_task_group() as tg:
                for existing_memory, similarity_score in filtered_candidates:
                    tg.start_soon(collect_result, existing_memory, similarity_score)

            # Filter out None results (failed checks or no replacement needed)
            return [r for r in results if r is not None]

        except Exception as e:
            # Graceful degradation: log warning and proceed without replacement
            self.logger.warning(
                f'Smart replacement check failed: {e}',
                extra={
                    'project_id': self._project_id,
                    'error': str(e),
                    'new_memory_preview': new_memory[:100],
                },
            )
            return []


class StoragePhase:
    '''Phase 3: Sequential storage with replacement handling.

    This phase:
    1. Processes replacements (archives and deletes old memories)
    2. Adds new memories to storage
    3. Commits changes to both engines
    '''

    def __init__(
        self,
        semantic_engine: ISemanticSearchEngine,
        tantivy_engine: Any,  # TantivyEngine | None
        config: Config,
        logger: StructuredLogger,
        write_lock: threading.Lock | None = None,
    ):
        '''Initialize storage phase.

        Args:
            semantic_engine: USearchEngine for semantic storage.
            tantivy_engine: TantivyEngine for full-text storage.
            config: Application configuration.
            logger: Structured logger instance.
        '''
        self._semantic_engine = semantic_engine
        self._tantivy_engine = tantivy_engine
        self.config = config
        self.logger = logger
        self._project_id = config.project_id
        self._write_lock = write_lock

    async def execute(
        self,
        memories: list[str],
        replacement_map: dict[str, list[ReplacementInfo]],
        dry_run: bool = False,
    ) -> Phase3Result:
        '''Execute Phase 3: Sequential storage.

        Args:
            memories: List of non-duplicate memories to store.
            replacement_map: Mapping from memory to replacement info.
            dry_run: If True, only check without making changes.

        Returns:
            Phase3Result with storage information.
        '''
        phase_start = time.perf_counter()

        stored_count = 0
        replaced_count = 0
        replacements: list[ReplacementInfo] = []
        memories_to_add: list[str] = []

        # Process each memory sequentially
        for idx, memory in enumerate(memories):
            replacement_infos = replacement_map.get(memory, [])

            # Process replacements (delete old memories)
            for replacement_info in replacement_infos:
                self.logger.info(
                    'Replacing old memory with new one',
                    extra={
                        'memory_index': idx + 1,
                        'action': 'replace',
                        'old_preview': truncate_memory(
                            replacement_info.old_memory, max_length=60
                        ),
                        'new_preview': truncate_memory(memory, max_length=60),
                        'confidence': replacement_info.confidence,
                        'similarity': replacement_info.similarity_score,
                        'dry_run': dry_run,
                    },
                )

                if not dry_run:
                    try:
                        # Archive the old memory before deletion (for recovery)
                        archived = await asyncify(self._archive_for_replacement)(
                            old_memory=replacement_info.old_memory,
                            new_memory=memory,
                            confidence=replacement_info.confidence,
                            reason=replacement_info.reason,
                        )
                        if archived:
                            self.logger.debug(
                                'Old memory archived for recovery',
                                extra={
                                    'memory_index': idx + 1,
                                    'action': 'archive_success',
                                },
                            )

                        # Delete the old memory
                        msg_id = self._semantic_engine.get_id_by_content(
                            self._project_id, replacement_info.old_memory
                        )
                        if msg_id is None:
                            self.logger.warning(
                                'Old memory not found for replacement delete',
                                extra={
                                    'memory_index': idx + 1,
                                    'action': 'delete_missing',
                                },
                            )
                            continue

                        self._delete_memory(
                            memory_id=str(msg_id),
                            content=replacement_info.old_memory,
                        )

                        replaced_count += 1
                        replacements.append(replacement_info)
                        self.logger.debug(
                            'Old memory removed successfully',
                            extra={
                                'memory_index': idx + 1,
                                'action': 'delete_success',
                                'archived': archived,
                            },
                        )
                    except Exception as delete_error:
                        # Graceful degradation: log warning and continue
                        self.logger.warning(
                            f'Failed to delete old memory: {delete_error}',
                            extra={
                                'memory_index': idx + 1,
                                'action': 'delete_failed',
                                'error': str(delete_error),
                            },
                        )
                else:
                    # In dry_run mode, just record what would be replaced
                    replaced_count += 1
                    replacements.append(replacement_info)

            # Add the new memory (unless dry_run)
            if not dry_run:
                memories_to_add.append(memory)
            else:
                # In dry_run, assume it would be stored
                stored_count += 1

        if not dry_run and memories_to_add:
            stored_memories = self._add_memories_batch(memories_to_add)
            stored_count = len(stored_memories)
            if stored_count != len(memories_to_add):
                self.logger.warning(
                    'Batch add stored fewer memories than expected',
                    extra={
                        'expected_count': len(memories_to_add),
                        'stored_count': stored_count,
                    },
                )

        # Commit changes (only in live mode)
        if not dry_run:
            if self._tantivy_engine is not None:
                self._tantivy_engine.commit()
            self._semantic_engine.commit()

        duration = time.perf_counter() - phase_start

        self.logger.info(
            f'Phase 3 complete: {stored_count} stored ({duration:.3f}s)',
            extra={
                'phase': 3,
                'duration_ms': int(duration * 1000),
                'stored_count': stored_count,
                'replaced_count': replaced_count,
            },
        )

        return Phase3Result(
            stored_count=stored_count,
            replaced_count=replaced_count,
            replacements=replacements,
            duration=duration,
        )

    def _add_memories_batch(self, memories: list[str]) -> list[str]:
        '''Add multiple memories to both semantic and full-text engines.

        Args:
            memories: List of memories to store.

        Returns:
            List of memories actually stored (duplicates skipped).
        '''
        if not memories:
            return []

        try:
            if self._write_lock is None:
                inserted_memories = self._semantic_engine.add_batch(
                    project_id=self._project_id,
                    contents=memories,
                    infer=self.config.enable_llm_infer,
                )

                if self._tantivy_engine is not None:
                    for memory in inserted_memories:
                        self._tantivy_engine.add(self._project_id, memory)
            else:
                with self._write_lock:
                    inserted_memories = self._semantic_engine.add_batch(
                        project_id=self._project_id,
                        contents=memories,
                        infer=self.config.enable_llm_infer,
                    )

                    if self._tantivy_engine is not None:
                        for memory in inserted_memories:
                            self._tantivy_engine.add(self._project_id, memory)

            self.logger.debug(
                'Batch added memories to hybrid storage',
                extra={
                    'project_id': self._project_id,
                    'memory_count': len(inserted_memories),
                    'engines': ['semantic', 'tantivy'],
                },
            )
            return inserted_memories

        except Exception as e:
            raise StorageError(f'Failed to add memory batch: {e}') from e

    def _delete_memory(self, memory_id: str, content: str) -> None:
        '''Delete a memory from semantic and full-text engines with write locking.'''
        if self._write_lock is None:
            self._semantic_engine.delete(memory_id=memory_id)
            if self._tantivy_engine is not None:
                self._tantivy_engine.delete(self._project_id, content)
            return

        with self._write_lock:
            self._semantic_engine.delete(memory_id=memory_id)
            if self._tantivy_engine is not None:
                self._tantivy_engine.delete(self._project_id, content)

    def _add_memory(self, content: str) -> bool:
        '''Add a single memory to BOTH USearch semantic and Tantivy full-text engines.

        Args:
            content: The memory to store.

        Returns:
            True if the memory was stored, False if it was skipped as a duplicate.

        Raises:
            RuntimeError: If storage operation fails.
        '''
        if self.config.deduplicate_memories and self._has_exact_match(content):
            self.logger.info(
                'Duplicate memory detected, skipping storage',
                extra={
                    'project_id': self._project_id,
                    'memory_preview': content[:200],
                },
            )
            return False

        try:
            self._semantic_engine.add(
                project_id=self._project_id,
                content=content,
                infer=self.config.enable_llm_infer,
            )

            # 2. Add to Tantivy full-text search engine
            if self._tantivy_engine is not None:
                self._tantivy_engine.add(self._project_id, content)

            self.logger.debug(
                'Memory added to hybrid storage',
                extra={
                    'project_id': self._project_id,
                    'memory_length': len(content),
                    'engines': ['semantic', 'tantivy'],
                },
            )
            return True

        except Exception as e:
            raise StorageError(f'Failed to add memory to hybrid storage: {e}') from e

    def _has_exact_match(self, content: str) -> bool:
        '''Check whether the exact memory already exists in storage.

        Uses Tantivy for fast exact phrase matching when hybrid search is enabled,
        falling back to direct database lookup otherwise. Both paths are O(log n)
        avoiding the ~100-500ms embedding API call overhead.
        '''
        return has_exact_match(
            semantic_engine=self._semantic_engine,
            tantivy_engine=self._tantivy_engine,
            project_id=self._project_id,
            content=content,
            logger=self.logger,
        )

    def _archive_for_replacement(
        self, old_memory: str, new_memory: str, confidence: float, reason: str
    ) -> bool:
        """Archive a memory before replacement (for recovery).

        This method archives a memory to the archived_messages table before
        it gets deleted during smart replacement. This enables recovery if
        the LLM made a mistake in determining the replacement.

        Args:
            old_memory: The memory being replaced and archived.
            new_memory: The new memory that replaces it.
            confidence: LLM confidence score for the replacement decision.
            reason: LLM's explanation for why replacement was appropriate.

        Returns:
            True if archiving succeeded, False otherwise.
        """
        try:
            msg_id = self._semantic_engine.get_id_by_content(
                self._project_id, old_memory
            )

            if msg_id is None:
                self.logger.warning(
                    'Cannot archive - memory not found',
                    extra={
                        'project_id': self._project_id,
                        'old_memory_preview': old_memory[:50],
                    },
                )
                return False

            memory_store = self._semantic_engine.memory_store
            archive_id = memory_store.archive(
                memory_id=msg_id,
                project_id=self._project_id,
                content=old_memory,
                replaced_by=new_memory,
                reason=reason,
                confidence=confidence,
            )

            if archive_id is not None:
                self.logger.info(
                    'Memory archived before replacement',
                    extra={
                        'project_id': self._project_id,
                        'archive_id': archive_id,
                        'original_id': msg_id,
                        'confidence': confidence,
                    },
                )
                return True

            return False

        except Exception as e:
            # Graceful degradation - log warning but don't block the replacement
            self.logger.warning(
                f'Failed to archive memory for replacement: {e}',
                extra={
                    'project_id': self._project_id,
                    'error': str(e),
                },
            )
            return False


class AddPipeline:
    '''Orchestrates the 3-phase add pipeline.

    This class coordinates the execution of the three phases:
    1. DuplicateDetectionPhase
    2. SmartReplacementPhase
    3. StoragePhase

    It combines the results from each phase into the final AddResult.
    '''

    def __init__(
        self,
        duplicate_detection_phase: DuplicateDetectionPhase,
        smart_replacement_phase: SmartReplacementPhase,
        storage_phase: StoragePhase,
        config: Config,
        logger: StructuredLogger,
    ):
        '''Initialize add pipeline.

        Args:
            duplicate_detection_phase: Phase 1 instance.
            smart_replacement_phase: Phase 2 instance.
            storage_phase: Phase 3 instance.
            config: Application configuration.
            logger: Structured logger instance.
        '''
        self._phase1 = duplicate_detection_phase
        self._phase2 = smart_replacement_phase
        self._phase3 = storage_phase
        self.config = config
        self.logger = logger

    async def execute(self, memories: list[str], dry_run: bool = False) -> AddResult:
        '''Execute the 3-phase add pipeline.

        Args:
            memories: List of memories to add.
            dry_run: If True, only check for replacements without making changes.

        Returns:
            AddResult with detailed information about the operation.

        Raises:
            StorageError: If storage operation fails (not raised in dry_run mode).
        '''
        result = AddResult()

        if not memories:
            return result

        mode_str = 'DRY RUN' if dry_run else 'LIVE'
        self.logger.info(
            f'Starting phased parallel memory addition [{mode_str}]: {len(memories)} memories',
            extra={
                'total_memories': len(memories),
                'smart_replace_enabled': self._phase2.smart_replace_enabled,
                'dry_run': dry_run,
                'optimization': 'phased_parallel',
            },
        )

        try:
            # Phase 1: Parallel duplicate detection
            phase1_result = await self._phase1.execute(memories)
            result.skipped_count = (
                len(phase1_result.storage_duplicates)
                + phase1_result.batch_duplicates_count
            )

            # Phase 2: Parallel smart replacement detection
            phase2_result = await self._phase2.execute(phase1_result.unique_memories)

            # Phase 3: Sequential database writes
            phase3_result = await self._phase3.execute(
                phase1_result.unique_memories,
                phase2_result.replacement_map,
                dry_run,
            )

            result.stored_count = phase3_result.stored_count
            result.replaced_count = phase3_result.replaced_count
            result.replacements = phase3_result.replacements

        except Exception as e:
            mode_str = 'DRY_RUN' if dry_run else 'LIVE'
            self.logger.error(
                f'Phased parallel memory addition failed [{mode_str}]: {e}',
                extra={
                    'mode': mode_str,
                    'dry_run': dry_run,
                    'total_memories': len(memories),
                    'stored_count': result.stored_count,
                    'replaced_count': result.replaced_count,
                    'error': str(e),
                    'error_type': type(e).__name__,
                },
            )
            if not dry_run:
                raise StorageError(f'Failed to add memories: {e}') from e

        self.logger.info(
            f'Phased parallel addition completed [{mode_str}]: '
            f'{result.stored_count}/{len(memories)} stored, '
            f'{result.replaced_count} replaced, {result.skipped_count} skipped',
            extra={
                'total_memories': len(memories),
                'stored_count': result.stored_count,
                'replaced_count': result.replaced_count,
                'skipped_count': result.skipped_count,
                'replacement_details': [
                    {
                        'old': truncate_memory(r.old_memory, 50),
                        'confidence': r.confidence,
                    }
                    for r in result.replacements
                ],
                'dry_run': dry_run,
                'optimization': 'phased_parallel',
            },
        )

        return result
