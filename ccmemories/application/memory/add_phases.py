"""Add operation phases for the 3-phase parallel add pipeline.

This module extracts the add operation logic from MemoryManager into
separate phase classes. The 3 phases are:
1. Duplicate Detection (parallel)
2. Smart Replacement Detection (parallel)
3. Sequential Storage

Each phase is implemented as a separate class that takes inputs and
produces outputs for the next phase.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import anyio
from asyncer import asyncify, create_task_group

from ..config import Config
from ..exceptions import StorageError
from ..types import ISemanticSearchEngine
from ..utils import StructuredLogger, truncate_message


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


@dataclass
class Phase1Result:
    """Result of Phase 1: Duplicate Detection.

    Attributes:
        unique_messages: Messages that are unique within the batch.
        storage_duplicates: Messages that already exist in storage.
        batch_duplicates_count: Count of duplicates within the batch.
        duration: Time taken for phase 1 execution.
    """

    unique_messages: List[str]
    storage_duplicates: List[str]
    batch_duplicates_count: int
    duration: float


@dataclass
class Phase2Result:
    """Result of Phase 2: Smart Replacement Detection.

    Attributes:
        replacement_map: Mapping from message to list of ReplacementInfo objects.
        total_replacements: Total number of replacements detected.
        duration: Time taken for phase 2 execution.
    """

    replacement_map: Dict[str, List[ReplacementInfo]]
    total_replacements: int
    duration: float


@dataclass
class Phase3Result:
    """Result of Phase 3: Sequential Storage.

    Attributes:
        stored_count: Number of messages stored.
        replaced_count: Number of memories replaced.
        replacements: List of ReplacementInfo objects.
        duration: Time taken for phase 3 execution.
    """

    stored_count: int
    replaced_count: int
    replacements: List[ReplacementInfo]
    duration: float


class DuplicateDetectionPhase:
    """Phase 1: Parallel duplicate detection.

    This phase:
    1. Deduplicates within the batch itself
    2. Checks against existing storage in parallel
    3. Returns unique messages and storage duplicates
    """

    def __init__(
        self,
        semantic_engine: ISemanticSearchEngine,
        tantivy_engine: Any,  # TantivyEngine | None
        config: Config,
        logger: StructuredLogger,
    ):
        """Initialize duplicate detection phase.

        Args:
            semantic_engine: USearchEngine for semantic search and direct database lookup.
            tantivy_engine: TantivyEngine for fast exact phrase matching.
            config: Application configuration.
            logger: Structured logger instance.
        """
        self._semantic_engine = semantic_engine
        self._tantivy_engine = tantivy_engine
        self.config = config
        self.logger = logger
        self._project_id = config.project_id

    async def execute(self, messages: List[str]) -> Phase1Result:
        """Execute Phase 1: Parallel duplicate detection.

        Args:
            messages: List of messages to check for duplicates.

        Returns:
            Phase1Result with unique messages and duplicate information.
        """
        phase_start = time.perf_counter()

        # Step 1: Deduplicate within the batch itself (preserve order, keep first)
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

        # Step 2: Parallel duplicate detection against existing storage
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

        duration = time.perf_counter() - phase_start
        self.logger.info(
            f"Phase 1 complete: {len(non_duplicate_messages)} unique, "
            f"{len(storage_duplicates)} storage duplicates ({duration:.3f}s)",
            extra={
                "phase": 1,
                "duration_ms": int(duration * 1000),
                "unique_count": len(non_duplicate_messages),
                "storage_duplicates": len(storage_duplicates),
                "batch_duplicates": batch_duplicates_count,
            },
        )

        return Phase1Result(
            unique_messages=non_duplicate_messages,
            storage_duplicates=storage_duplicates,
            batch_duplicates_count=batch_duplicates_count,
            duration=duration,
        )

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
                    self._project_id,
                    limit=5,  # Small limit since we only need to check existence
                )
                # Check for exact string match in results
                has_match = any(msg == message for msg, _ in results)
                if has_match:
                    self.logger.debug(
                        "Tantivy found exact duplicate",
                        extra={"project_id": self._project_id},
                    )
                return has_match
            except Exception as e:
                self.logger.warning(
                    "Tantivy duplicate check failed; falling back to database lookup",
                    extra={"project_id": self._project_id, "error": str(e)},
                )

        # Optimized fallback: Direct database lookup (O(log n), no embedding API call)
        # This avoids the 100-500ms embedding API overhead of semantic search
        try:
            msg_id = self._semantic_engine.get_id_by_message(self._project_id, message)
            if msg_id is not None:
                self.logger.debug(
                    "Database lookup found exact duplicate",
                    extra={"project_id": self._project_id, "msg_id": msg_id},
                )
                return True
            return False
        except Exception as e:
            self.logger.warning(
                "Duplicate detection failed; proceeding without deduplication",
                extra={
                    "project_id": self._project_id,
                    "error": str(e),
                },
            )
            return False

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


class SmartReplacementPhase:
    """Phase 2: Parallel smart replacement detection.

    This phase:
    1. Finds similar existing memories via semantic search
    2. Filters by similarity threshold
    3. Checks candidates in parallel with LLM
    4. Returns replacement map
    """

    def __init__(
        self,
        semantic_engine: ISemanticSearchEngine,
        smart_replacer: Any,  # SmartReplacer | None
        config: Config,
        logger: StructuredLogger,
    ):
        """Initialize smart replacement phase.

        Args:
            semantic_engine: USearchEngine for semantic search.
            smart_replacer: SmartReplacer for LLM replacement detection.
            config: Application configuration.
            logger: Structured logger instance.
        """
        self._semantic_engine = semantic_engine
        self._smart_replacer = smart_replacer
        self.config = config
        self.logger = logger
        self._project_id = config.project_id

    async def execute(self, messages: List[str]) -> Phase2Result:
        """Execute Phase 2: Parallel smart replacement detection.

        Args:
            messages: List of non-duplicate messages to check for replacements.

        Returns:
            Phase2Result with replacement information.
        """
        phase_start = time.perf_counter()

        # Map: message -> list of replacement infos
        replacement_map: Dict[str, List[ReplacementInfo]] = {}

        if self._smart_replacer is None or not messages:
            return Phase2Result(
                replacement_map=replacement_map,
                total_replacements=0,
                duration=time.perf_counter() - phase_start,
            )

        # Capture in local variable for type narrowing in nested function
        smart_replacer = self._smart_replacer

        async with create_task_group() as tg:
            replacement_results: List[Tuple[str, List[ReplacementInfo]]] = []

            async def check_replacement_for_msg(msg: str) -> None:
                """Check replacement for a single message."""
                infos = await self._check_for_replacement(msg, smart_replacer)
                replacement_results.append((msg, infos))

            for msg in messages:
                tg.soonify(check_replacement_for_msg)(msg)

        # Collect results
        for msg, infos in replacement_results:
            replacement_map[msg] = infos

        duration = time.perf_counter() - phase_start
        total_replacements = sum(len(infos) for infos in replacement_map.values())

        self.logger.info(
            f"Phase 2 complete: {total_replacements} replacements detected ({duration:.3f}s)",
            extra={
                "phase": 2,
                "duration_ms": int(duration * 1000),
                "messages_checked": len(messages),
                "total_replacements": total_replacements,
            },
        )

        return Phase2Result(
            replacement_map=replacement_map,
            total_replacements=total_replacements,
            duration=duration,
        )

    async def _check_for_replacement(
        self, new_memory: str, smart_replacer: Any
    ) -> List[ReplacementInfo]:
        """Check if new memory should replace existing memories.

        Uses semantic search to find the most similar existing memories,
        applies similarity pre-filter, then uses LLM to determine which
        should be replaced.

        Args:
            new_memory: The new memory being added.
            smart_replacer: SmartReplacer instance.

        Returns:
            List of ReplacementInfo objects for memories that should be replaced.
            Empty list if no replacements needed.
        """
        replacements: List[ReplacementInfo] = []

        try:
            # Step 1: Find top N most similar memories via semantic search
            candidate_limit = self.config.smart_replace_candidate_limit
            similar_results = self._semantic_engine.search(
                query=new_memory,
                project_id=self._project_id,
                limit=candidate_limit,
            )

            if not similar_results:
                self.logger.debug(
                    "No similar memories found for replacement check",
                    extra={
                        "project_id": self._project_id,
                        "new_memory_preview": new_memory[:100],
                    },
                )
                return []

            # Step 2: Filter candidates by similarity threshold (pre-filter)
            # Note: similar_results are 3-tuples (message, score, created_at)
            min_similarity = self.config.smart_replace_min_similarity
            filtered_candidates = [
                (mem, score)
                for mem, score, _ in similar_results
                if score >= min_similarity and mem != new_memory
            ]

            if not filtered_candidates:
                self.logger.debug(
                    f"All candidates below similarity threshold ({min_similarity})",
                    extra={
                        "project_id": self._project_id,
                        "candidate_count": len(similar_results),
                        "min_similarity": min_similarity,
                    },
                )
                return []

            self.logger.debug(
                f"Found {len(filtered_candidates)} candidates above similarity threshold",
                extra={
                    "project_id": self._project_id,
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
                                    "project_id": self._project_id,
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
                                    "project_id": self._project_id,
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
                                "project_id": self._project_id,
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
                    "project_id": self._project_id,
                    "error": str(e),
                    "new_memory_preview": new_memory[:100],
                },
            )
            return []


class StoragePhase:
    """Phase 3: Sequential storage with replacement handling.

    This phase:
    1. Processes replacements (archives and deletes old memories)
    2. Adds new messages to storage
    3. Commits changes to both engines
    """

    def __init__(
        self,
        semantic_engine: ISemanticSearchEngine,
        tantivy_engine: Any,  # TantivyEngine | None
        config: Config,
        logger: StructuredLogger,
    ):
        """Initialize storage phase.

        Args:
            semantic_engine: USearchEngine for semantic storage.
            tantivy_engine: TantivyEngine for full-text storage.
            config: Application configuration.
            logger: Structured logger instance.
        """
        self._semantic_engine = semantic_engine
        self._tantivy_engine = tantivy_engine
        self.config = config
        self.logger = logger
        self._project_id = config.project_id

    async def execute(
        self,
        messages: List[str],
        replacement_map: Dict[str, List[ReplacementInfo]],
        dry_run: bool = False,
    ) -> Phase3Result:
        """Execute Phase 3: Sequential storage.

        Args:
            messages: List of non-duplicate messages to store.
            replacement_map: Mapping from message to replacement info.
            dry_run: If True, only check without making changes.

        Returns:
            Phase3Result with storage information.
        """
        phase_start = time.perf_counter()

        stored_count = 0
        replaced_count = 0
        replacements: List[ReplacementInfo] = []

        # Process each message sequentially
        for idx, message in enumerate(messages):
            replacement_infos = replacement_map.get(message, [])

            # Process replacements (delete old memories)
            for replacement_info in replacement_infos:
                self.logger.info(
                    "Replacing old memory with new one",
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
                        archived = await asyncify(self._archive_for_replacement)(
                            old_memory=replacement_info.old_memory,
                            new_memory=message,
                            confidence=replacement_info.confidence,
                            reason=replacement_info.reason,
                        )
                        if archived:
                            self.logger.debug(
                                "Old memory archived for recovery",
                                extra={
                                    "message_index": idx + 1,
                                    "action": "archive_success",
                                },
                            )

                        # Delete the old memory
                        deleted = await asyncify(self._semantic_engine.delete)(
                            memory_id=str(
                                self._semantic_engine.get_id_by_message(
                                    self._project_id, replacement_info.old_memory
                                )
                                or ""
                            )
                        )
                        if deleted is None:
                            # Delete from Tantivy too
                            if self._tantivy_engine is not None:
                                self._tantivy_engine.delete(
                                    self._project_id, replacement_info.old_memory
                                )

                            replaced_count += 1
                            replacements.append(replacement_info)
                            self.logger.debug(
                                "Old memory removed successfully",
                                extra={
                                    "message_index": idx + 1,
                                    "action": "delete_success",
                                    "archived": archived,
                                },
                            )
                    except Exception as delete_error:
                        # Graceful degradation: log warning and continue
                        self.logger.warning(
                            f"Failed to delete old memory: {delete_error}",
                            extra={
                                "message_index": idx + 1,
                                "action": "delete_failed",
                                "error": str(delete_error),
                            },
                        )
                else:
                    # In dry_run mode, just record what would be replaced
                    replaced_count += 1
                    replacements.append(replacement_info)

            # Add the new message (unless dry_run)
            if not dry_run:
                stored = await asyncify(self._add_message)(message)
                if stored:
                    stored_count += 1
                else:
                    # Shouldn't happen since we already checked, but handle gracefully
                    self.logger.warning(
                        "Message was marked as unique but failed to store",
                        extra={"message_index": idx + 1, "reason": "storage_failed"},
                    )
            else:
                # In dry_run, assume it would be stored
                stored_count += 1

        # Commit changes (only in live mode)
        if not dry_run:
            if self._tantivy_engine is not None:
                self._tantivy_engine.commit()
            self._semantic_engine.commit()

        duration = time.perf_counter() - phase_start

        self.logger.info(
            f"Phase 3 complete: {stored_count} stored ({duration:.3f}s)",
            extra={
                "phase": 3,
                "duration_ms": int(duration * 1000),
                "stored_count": stored_count,
                "replaced_count": replaced_count,
            },
        )

        return Phase3Result(
            stored_count=stored_count,
            replaced_count=replaced_count,
            replacements=replacements,
            duration=duration,
        )

    def _add_message(self, message: str) -> bool:
        """Add a single message to BOTH USearch semantic and Tantivy full-text engines.

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
                    "project_id": self._project_id,
                    "message_preview": message[:200],
                },
            )
            return False

        try:
            # 1. Add to USearch semantic engine
            self._semantic_engine.add(
                project_id=self._project_id,
                message=message,
                infer=self.config.enable_llm_infer,
            )

            # 2. Add to Tantivy full-text search engine
            if self._tantivy_engine is not None:
                self._tantivy_engine.add(self._project_id, message)

            self.logger.debug(
                "Message added to hybrid storage",
                extra={
                    "project_id": self._project_id,
                    "message_length": len(message),
                    "engines": ["semantic", "tantivy"],
                },
            )
            return True

        except Exception as e:
            raise StorageError(f"Failed to add message to hybrid storage: {e}") from e

    def _has_exact_match(self, message: str) -> bool:
        """Check whether the exact message already exists in storage.

        Uses Tantivy for fast exact phrase matching when hybrid search is enabled,
        falling back to direct database lookup otherwise. Both paths are O(log n)
        avoiding the ~100-500ms embedding API call overhead.
        """
        # Fast path: Use Tantivy for exact match if hybrid search is enabled
        if self._tantivy_engine is not None:
            try:
                # Use quoted exact phrase search with escaped query
                escaped_query = self._escape_tantivy_query(message)
                results = self._tantivy_engine.search(
                    f'"{escaped_query}"',
                    self._project_id,
                    limit=5,  # Small limit since we only need to check existence
                )
                # Check for exact string match in results
                has_match = any(msg == message for msg, _ in results)
                if has_match:
                    self.logger.debug(
                        "Tantivy found exact duplicate",
                        extra={"project_id": self._project_id},
                    )
                return has_match
            except Exception as e:
                self.logger.warning(
                    "Tantivy duplicate check failed; falling back to database lookup",
                    extra={"project_id": self._project_id, "error": str(e)},
                )

        # Optimized fallback: Direct database lookup (O(log n), no embedding API call)
        try:
            msg_id = self._semantic_engine.get_id_by_message(self._project_id, message)
            if msg_id is not None:
                self.logger.debug(
                    "Database lookup found exact duplicate",
                    extra={"project_id": self._project_id, "msg_id": msg_id},
                )
                return True
            return False
        except Exception as e:
            self.logger.warning(
                "Duplicate detection failed; proceeding without deduplication",
                extra={
                    "project_id": self._project_id,
                    "error": str(e),
                },
            )
            return False

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
                self._project_id, old_memory
            )

            if msg_id is None:
                self.logger.warning(
                    "Cannot archive - message not found",
                    extra={
                        "project_id": self._project_id,
                        "old_memory_preview": old_memory[:50],
                    },
                )
                return False

            # Access the message store through the semantic engine
            message_store = self._semantic_engine.message_store
            archive_id = message_store.archive(
                message_id=msg_id,
                project_id=self._project_id,
                message=old_memory,
                replaced_by=new_memory,
                reason=reason,
                confidence=confidence,
            )

            if archive_id is not None:
                self.logger.info(
                    "Message archived before replacement",
                    extra={
                        "project_id": self._project_id,
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
                    "project_id": self._project_id,
                    "error": str(e),
                },
            )
            return False


class AddPipeline:
    """Orchestrates the 3-phase add pipeline.

    This class coordinates the execution of the three phases:
    1. DuplicateDetectionPhase
    2. SmartReplacementPhase
    3. StoragePhase

    It combines the results from each phase into the final AddResult.
    """

    def __init__(
        self,
        duplicate_detection_phase: DuplicateDetectionPhase,
        smart_replacement_phase: SmartReplacementPhase,
        storage_phase: StoragePhase,
        config: Config,
        logger: StructuredLogger,
    ):
        """Initialize add pipeline.

        Args:
            duplicate_detection_phase: Phase 1 instance.
            smart_replacement_phase: Phase 2 instance.
            storage_phase: Phase 3 instance.
            config: Application configuration.
            logger: Structured logger instance.
        """
        self._phase1 = duplicate_detection_phase
        self._phase2 = smart_replacement_phase
        self._phase3 = storage_phase
        self.config = config
        self.logger = logger

    async def execute(self, messages: List[str], dry_run: bool = False) -> AddResult:
        """Execute the 3-phase add pipeline.

        Args:
            messages: List of messages to add.
            dry_run: If True, only check for replacements without making changes.

        Returns:
            AddResult with detailed information about the operation.

        Raises:
            StorageError: If storage operation fails (not raised in dry_run mode).
        """
        result = AddResult()

        if not messages:
            return result

        mode_str = "DRY RUN" if dry_run else "LIVE"
        self.logger.info(
            f"Starting phased parallel message addition [{mode_str}]: {len(messages)} messages",
            extra={
                "total_messages": len(messages),
                "smart_replace_enabled": self._phase2._smart_replacer is not None,
                "dry_run": dry_run,
                "optimization": "phased_parallel",
            },
        )

        try:
            # Phase 1: Parallel duplicate detection
            phase1_result = await self._phase1.execute(messages)
            result.skipped_count = (
                len(phase1_result.storage_duplicates)
                + phase1_result.batch_duplicates_count
            )

            # Phase 2: Parallel smart replacement detection
            phase2_result = await self._phase2.execute(phase1_result.unique_messages)

            # Phase 3: Sequential database writes
            phase3_result = await self._phase3.execute(
                phase1_result.unique_messages,
                phase2_result.replacement_map,
                dry_run,
            )

            result.stored_count = phase3_result.stored_count
            result.replaced_count = phase3_result.replaced_count
            result.replacements = phase3_result.replacements

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
