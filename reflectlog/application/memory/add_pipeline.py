"""Add pipeline with pluggable phases.

This module provides the AddPipeline class that orchestrates the
3-phase add process with pluggable phases:
1. Duplicate Detection
2. Smart Replacement Detection
3. Sequential Storage
"""

from dataclasses import dataclass, field
import logging
import threading
from typing import Protocol

from reflectlog.application.config import Config
from reflectlog.application.utils import StructuredLogger
from reflectlog.core.memory import IMemoryBackend
from reflectlog.core.reranking import IReranker

logger = logging.getLogger(__name__)


def _empty_replacements() -> list[ReplacementInfo]:
    return []


@dataclass
class ReplacementInfo:
    """Information about a memory replacement."""

    old_memory: str
    new_memory: str
    confidence: float
    reason: str
    similarity_score: float = 0.0


@dataclass
class AddResult:
    """Result of adding memories to memory storage."""

    stored_count: int = 0
    skipped_count: int = 0
    replaced_count: int = 0
    replacements: list[ReplacementInfo] = field(default_factory=_empty_replacements)


class IDuplicateDetectionPhase(Protocol):
    """Protocol for duplicate detection phases."""

    async def detect(
        self,
        memories: list[str],
        project_id: str,
    ) -> tuple[list[str], list[str], int]:
        """Detect duplicates in memories.

        Args:
            memories: Memories to check.
            project_id: Project identifier.

        Returns:
            Tuple of (unique_memories, duplicates, duplicate_count).
        """
        ...


class IReplacementDetectionPhase(Protocol):
    """Protocol for smart replacement detection phases."""

    async def detect(
        self,
        memories: list[str],
        project_id: str,
    ) -> tuple[list[str], list[ReplacementInfo]]:
        """Detect potential replacements.

        Args:
            memories: Memories to check.
            project_id: Project identifier.

        Returns:
            Tuple of (memories_to_store, replacement_info).
        """
        ...


class IStoragePhase(Protocol):
    """Protocol for storage phases."""

    async def store(
        self,
        memories: list[str],
        project_id: str,
    ) -> int:
        """Store memories.

        Args:
            memories: Memories to store.
            project_id: Project identifier.

        Returns:
            Number of memories stored.
        """
        ...


class DefaultDuplicateDetectionPhase:
    """Default duplicate detection phase."""

    def __init__(
        self,
        semantic_backend: IMemoryBackend,
        fulltext_backend: IMemoryBackend | None,
        deduplicate_enabled: bool,
    ):
        super().__init__()
        self._semantic = semantic_backend
        self._fulltext = fulltext_backend
        self._deduplicate = deduplicate_enabled

    async def detect(
        self,
        memories: list[str],
        project_id: str,
    ) -> tuple[list[str], list[str], int]:
        """Detect duplicates using storage check and batch check."""
        if not self._deduplicate:
            return memories, [], 0

        # Check for batch duplicates
        seen: set[str] = set()
        batch_dupes: set[str] = set()
        unique: list[str] = []
        for msg in memories:
            if msg in seen:
                batch_dupes.add(msg)
            else:
                seen.add(msg)
                unique.append(msg)

        # Check for storage duplicates
        storage_dupes: list[str] = []
        unique_without_dupes: list[str] = []
        for msg in unique:
            exists = await self._semantic.exists(project_id, msg)
            if exists:
                storage_dupes.append(msg)
            else:
                unique_without_dupes.append(msg)

        return (
            unique_without_dupes,
            list(storage_dupes) + list(batch_dupes),
            len(storage_dupes) + len(batch_dupes),
        )


class NoopReplacementDetectionPhase:
    """No-op replacement detection (skip smart replacement)."""

    async def detect(
        self,
        memories: list[str],
        project_id: str,
    ) -> tuple[list[str], list[ReplacementInfo]]:
        """Return memories unchanged with no replacements."""
        return memories, []


class DefaultStoragePhase:
    """Default storage phase using backend."""

    def __init__(
        self,
        semantic_backend: IMemoryBackend,
        fulltext_backend: IMemoryBackend | None,
        write_lock: threading.Lock,
    ):
        super().__init__()
        self._semantic = semantic_backend
        self._fulltext = fulltext_backend
        self._write_lock = write_lock

    async def store(
        self,
        memories: list[str],
        project_id: str,
    ) -> int:
        """Store memories with write lock protection."""
        with self._write_lock:
            if not memories:
                return 0

            stored = await self._semantic.add_batch(project_id, memories)

            if self._fulltext is not None:
                for msg in stored:
                    _ = await self._fulltext.add(project_id, msg)

            return len(stored)


@dataclass
class AddPipelineConfig:
    """Configuration for add pipeline phases."""

    deduplicate_memories: bool
    enable_smart_replace: bool
    smart_replace_threshold: float


class AddPipeline:
    """Modular add pipeline with pluggable phases.

    This pipeline orchestrates the add process with configurable phases:
    1. Duplicate Detection
    2. Smart Replacement Detection
    3. Sequential Storage

    Example:
        pipeline = AddPipeline(
            dedup_phase=DefaultDuplicateDetectionPhase(semantic, fulltext),
            replace_phase=NoopReplacementDetectionPhase(),
            storage_phase=DefaultStoragePhase(semantic, fulltext),
            config=pipeline_config,
        )
        result = await pipeline.execute(memories, project_id)
    """

    def __init__(
        self,
        dedup_phase: IDuplicateDetectionPhase,
        replace_phase: IReplacementDetectionPhase,
        storage_phase: IStoragePhase,
        config: AddPipelineConfig,
        logger: StructuredLogger,
    ):
        super().__init__()
        """Initialize add pipeline.

        Args:
            dedup_phase: Phase for detecting duplicates.
            replace_phase: Phase for detecting replacements.
            storage_phase: Phase for storing memories.
            config: Pipeline configuration.
            logger: Structured logger.
        """
        self._dedup_phase = dedup_phase
        self._replace_phase = replace_phase
        self._storage_phase = storage_phase
        self._config = config
        self._logger = logger

    async def execute(
        self,
        memories: list[str],
        project_id: str,
        dry_run: bool = False,
    ) -> AddResult:
        """Execute the full add pipeline.

        Args:
            memories: Memories to add.
            project_id: Project identifier.
            dry_run: If True, only check without making changes.

        Returns:
            AddResult with details about stored/skipped/replaced memories.
        """
        # Phase 1: Duplicate Detection
        unique_memories, duplicates, _ = await self._dedup_phase.detect(
            memories, project_id
        )

        if dry_run:
            return AddResult(
                stored_count=len(unique_memories),
                skipped_count=len(duplicates),
            )

        # Phase 2: Smart Replacement Detection
        if self._config.enable_smart_replace:
            memories_to_store, replacements = await self._replace_phase.detect(
                unique_memories, project_id
            )
        else:
            memories_to_store, replacements = unique_memories, []

        # Phase 3: Sequential Storage
        stored_count = await self._storage_phase.store(memories_to_store, project_id)

        return AddResult(
            stored_count=stored_count,
            skipped_count=len(duplicates),
            replaced_count=len(replacements),
            replacements=replacements,
        )


def create_default_pipeline(
    semantic_backend: IMemoryBackend,
    fulltext_backend: IMemoryBackend | None,
    replace_detector: IReranker | None,
    config: Config,
    logger: StructuredLogger,
    write_lock: threading.Lock,
) -> AddPipeline:
    """Create an add pipeline with default configuration.

    Args:
        semantic_backend: Semantic search backend (USearch).
        fulltext_backend: Full-text search backend (Tantivy) or None.
        replace_detector: Replacement detector (SmartReplacer) or None.
        config: Application configuration.
        logger: Structured logger.
        write_lock: Lock for serializing writes.

    Returns:
        Configured AddPipeline instance.
    """
    # Duplicate detection phase
    dedup_phase = DefaultDuplicateDetectionPhase(
        semantic_backend,
        fulltext_backend,
        config.deduplicate_memories,
    )

    # Replacement detection phase
    if config.enable_smart_replace and replace_detector is not None:
        replace_phase = DefaultReplacementDetectionPhase(replace_detector)
    else:
        replace_phase = NoopReplacementDetectionPhase()

    # Storage phase
    storage_phase = DefaultStoragePhase(
        semantic_backend,
        fulltext_backend,
        write_lock,
    )

    # Pipeline config
    pipeline_config = AddPipelineConfig(
        deduplicate_memories=config.deduplicate_memories,
        enable_smart_replace=config.enable_smart_replace,
        smart_replace_threshold=config.smart_replace_threshold,
    )

    return AddPipeline(
        dedup_phase=dedup_phase,
        replace_phase=replace_phase,
        storage_phase=storage_phase,
        config=pipeline_config,
        logger=logger,
    )


class DefaultReplacementDetectionPhase:
    """Default smart replacement detection phase using a reranker."""

    def __init__(self, replacer: IReranker):
        super().__init__()
        self._replacer = replacer

    async def detect(
        self,
        memories: list[str],
        project_id: str,
    ) -> tuple[list[str], list[ReplacementInfo]]:
        """Detect potential replacements using the replacer."""
        # TODO: Implement smart replacement detection
        # This would use the replacer to check for similar memories
        return memories, []
