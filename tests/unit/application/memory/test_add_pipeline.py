"""Unit tests for the add pipeline module.

Tests cover:
- ReplacementInfo and AddResult dataclasses
- AddPipelineConfig dataclass
- DefaultDuplicateDetectionPhase: dedup enabled/disabled, batch + storage dedup
- NoopReplacementDetectionPhase: passthrough behavior
- DefaultReplacementDetectionPhase: stub behavior
- DefaultStoragePhase: write-lock, dual-engine storage
- AddPipeline: full pipeline execution with all phases
- create_default_pipeline factory function
"""

import threading
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.memory.add_pipeline import (
    AddPipeline,
    AddPipelineConfig,
    AddResult,
    DefaultDuplicateDetectionPhase,
    DefaultReplacementDetectionPhase,
    DefaultStoragePhase,
    NoopReplacementDetectionPhase,
    ReplacementInfo,
    create_default_pipeline,
)
from reflectlog.application.utils import StructuredLogger
from reflectlog.core.logging import IStructuredLogger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_semantic_backend() -> AsyncMock:
    """Mock semantic backend (IMemoryBackend)."""
    backend = AsyncMock()
    backend.exists = AsyncMock(return_value=False)
    backend.add_batch = AsyncMock(side_effect=lambda pid, msgs: msgs)
    backend.add = AsyncMock(return_value="id-1")
    return backend


@pytest.fixture
def mock_fulltext_backend() -> AsyncMock:
    """Mock full-text backend (IMemoryBackend)."""
    backend = AsyncMock()
    backend.add = AsyncMock(return_value="id-2")
    return backend


@pytest.fixture
def mock_logger() -> IStructuredLogger:
    """Mock structured logger."""
    return cast(IStructuredLogger, Mock(spec=StructuredLogger))


@pytest.fixture
def write_lock() -> threading.Lock:
    """Thread lock for storage phase."""
    return threading.Lock()


@pytest.fixture
def pipeline_config() -> AddPipelineConfig:
    """Standard pipeline configuration."""
    return AddPipelineConfig(
        deduplicate_memories=True,
        enable_smart_replace=False,
        smart_replace_threshold=0.7,
    )


@pytest.fixture
def pipeline_config_with_replace() -> AddPipelineConfig:
    """Pipeline configuration with smart replacement enabled."""
    return AddPipelineConfig(
        deduplicate_memories=True,
        enable_smart_replace=True,
        smart_replace_threshold=0.7,
    )


# ---------------------------------------------------------------------------
# ReplacementInfo dataclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReplacementInfo:
    """Tests for ReplacementInfo dataclass."""

    def test_basic_creation(self) -> None:
        """Create ReplacementInfo with required fields."""
        info = ReplacementInfo(
            old_memory="old text",
            new_memory="new text",
            confidence=0.85,
            reason="updated info",
        )
        assert info.old_memory == "old text"
        assert info.new_memory == "new text"
        assert info.confidence == 0.85
        assert info.reason == "updated info"
        assert info.similarity_score == 0.0

    def test_custom_similarity_score(self) -> None:
        """Create ReplacementInfo with custom similarity_score."""
        info = ReplacementInfo(
            old_memory="a",
            new_memory="b",
            confidence=0.9,
            reason="reason",
            similarity_score=0.75,
        )
        assert info.similarity_score == 0.75


# ---------------------------------------------------------------------------
# AddResult dataclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAddResult:
    """Tests for AddResult dataclass."""

    def test_default_values(self) -> None:
        """Default AddResult has all zeros and empty replacements."""
        result = AddResult()
        assert result.stored_count == 0
        assert result.skipped_count == 0
        assert result.replaced_count == 0
        assert result.replacements == []

    def test_custom_values(self) -> None:
        """Create AddResult with custom values."""
        info = ReplacementInfo(
            old_memory="old",
            new_memory="new",
            confidence=0.8,
            reason="update",
        )
        result = AddResult(
            stored_count=3,
            skipped_count=1,
            replaced_count=1,
            replacements=[info],
        )
        assert result.stored_count == 3
        assert result.skipped_count == 1
        assert result.replaced_count == 1
        assert len(result.replacements) == 1
        assert result.replacements[0].confidence == 0.8

    def test_replacements_list_independence(self) -> None:
        """Each AddResult should have an independent replacements list."""
        r1 = AddResult()
        r2 = AddResult()
        r1.replacements.append(
            ReplacementInfo(
                old_memory="a",
                new_memory="b",
                confidence=0.5,
                reason="r",
            )
        )
        assert len(r2.replacements) == 0


# ---------------------------------------------------------------------------
# AddPipelineConfig dataclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAddPipelineConfig:
    """Tests for AddPipelineConfig dataclass."""

    def test_creation(self) -> None:
        """Create config with all fields."""
        cfg = AddPipelineConfig(
            deduplicate_memories=True,
            enable_smart_replace=False,
            smart_replace_threshold=0.7,
        )
        assert cfg.deduplicate_memories is True
        assert cfg.enable_smart_replace is False
        assert cfg.smart_replace_threshold == 0.7


# ---------------------------------------------------------------------------
# DefaultDuplicateDetectionPhase
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDefaultDuplicateDetectionPhase:
    """Tests for DefaultDuplicateDetectionPhase."""

    async def test_dedup_disabled_returns_all_memories(
        self, mock_semantic_backend: AsyncMock
    ) -> None:
        """When dedup is disabled, all memories pass through unchanged."""
        phase = DefaultDuplicateDetectionPhase(
            semantic_backend=mock_semantic_backend,
            fulltext_backend=None,
            deduplicate_enabled=False,
        )
        memories = ["msg1", "msg2", "msg1"]
        unique, dupes, count = await phase.detect(memories, "proj")

        assert unique == ["msg1", "msg2", "msg1"]
        assert dupes == []
        assert count == 0
        mock_semantic_backend.exists.assert_not_called()

    async def test_batch_duplicate_removal(
        self, mock_semantic_backend: AsyncMock
    ) -> None:
        """Duplicate memories within the same batch are detected."""
        phase = DefaultDuplicateDetectionPhase(
            semantic_backend=mock_semantic_backend,
            fulltext_backend=None,
            deduplicate_enabled=True,
        )
        memories = ["alpha", "beta", "alpha", "gamma", "beta"]
        unique, dupes, count = await phase.detect(memories, "proj")

        assert unique == ["alpha", "beta", "gamma"]
        assert count == 2
        # batch dupes appear in duplicates list
        assert "alpha" in dupes
        assert "beta" in dupes

    async def test_storage_duplicate_removal(
        self, mock_semantic_backend: AsyncMock
    ) -> None:
        """Memories already in storage are detected as duplicates."""
        mock_semantic_backend.exists = AsyncMock(
            side_effect=lambda pid, msg: msg == "existing"
        )
        phase = DefaultDuplicateDetectionPhase(
            semantic_backend=mock_semantic_backend,
            fulltext_backend=None,
            deduplicate_enabled=True,
        )
        memories = ["existing", "new_msg"]
        unique, dupes, count = await phase.detect(memories, "proj")

        assert unique == ["new_msg"]
        assert "existing" in dupes
        assert count == 1

    async def test_combined_batch_and_storage_duplicates(
        self, mock_semantic_backend: AsyncMock
    ) -> None:
        """Both batch and storage duplicates are detected together."""
        mock_semantic_backend.exists = AsyncMock(
            side_effect=lambda pid, msg: msg == "in_storage"
        )
        phase = DefaultDuplicateDetectionPhase(
            semantic_backend=mock_semantic_backend,
            fulltext_backend=None,
            deduplicate_enabled=True,
        )
        memories = ["in_storage", "new", "new"]
        unique, dupes, count = await phase.detect(memories, "proj")

        assert unique == ["new"]
        assert count == 2
        assert "in_storage" in dupes
        assert "new" in dupes

    async def test_empty_memories(self, mock_semantic_backend: AsyncMock) -> None:
        """Empty memory list returns empty results."""
        phase = DefaultDuplicateDetectionPhase(
            semantic_backend=mock_semantic_backend,
            fulltext_backend=None,
            deduplicate_enabled=True,
        )
        unique, dupes, count = await phase.detect([], "proj")

        assert unique == []
        assert dupes == []
        assert count == 0

    async def test_no_duplicates(self, mock_semantic_backend: AsyncMock) -> None:
        """When all memories are unique, none are skipped."""
        phase = DefaultDuplicateDetectionPhase(
            semantic_backend=mock_semantic_backend,
            fulltext_backend=None,
            deduplicate_enabled=True,
        )
        memories = ["a", "b", "c"]
        unique, dupes, count = await phase.detect(memories, "proj")

        assert unique == ["a", "b", "c"]
        assert dupes == []
        assert count == 0

    async def test_fulltext_backend_not_consulted(
        self,
        mock_semantic_backend: AsyncMock,
        mock_fulltext_backend: AsyncMock,
    ) -> None:
        """Fulltext backend is stored but not used for dedup checks."""
        phase = DefaultDuplicateDetectionPhase(
            semantic_backend=mock_semantic_backend,
            fulltext_backend=mock_fulltext_backend,
            deduplicate_enabled=True,
        )
        memories = ["msg"]
        await phase.detect(memories, "proj")

        mock_fulltext_backend.exists.assert_not_called()


# ---------------------------------------------------------------------------
# NoopReplacementDetectionPhase
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNoopReplacementDetectionPhase:
    """Tests for NoopReplacementDetectionPhase."""

    async def test_returns_memories_unchanged(self) -> None:
        """Noop phase returns memories as-is with no replacements."""
        phase = NoopReplacementDetectionPhase()
        memories = ["msg1", "msg2"]
        result_msgs, replacements = await phase.detect(memories, "proj")

        assert result_msgs == ["msg1", "msg2"]
        assert replacements == []

    async def test_empty_memories(self) -> None:
        """Noop phase handles empty memory list."""
        phase = NoopReplacementDetectionPhase()
        result_msgs, replacements = await phase.detect([], "proj")

        assert result_msgs == []
        assert replacements == []


# ---------------------------------------------------------------------------
# DefaultReplacementDetectionPhase
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDefaultReplacementDetectionPhase:
    """Tests for DefaultReplacementDetectionPhase."""

    async def test_returns_memories_unchanged(self) -> None:
        """Current stub returns memories as-is with no replacements."""
        mock_reranker = AsyncMock()
        mock_reranker.name = "test_reranker"
        phase = DefaultReplacementDetectionPhase(mock_reranker)
        memories = ["msg1", "msg2"]
        result_msgs, replacements = await phase.detect(memories, "proj")

        assert result_msgs == ["msg1", "msg2"]
        assert replacements == []

    async def test_empty_memories(self) -> None:
        """Handles empty memory list."""
        mock_reranker = AsyncMock()
        mock_reranker.name = "test_reranker"
        phase = DefaultReplacementDetectionPhase(mock_reranker)
        result_msgs, replacements = await phase.detect([], "proj")

        assert result_msgs == []
        assert replacements == []


# ---------------------------------------------------------------------------
# DefaultStoragePhase
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDefaultStoragePhase:
    """Tests for DefaultStoragePhase."""

    async def test_store_empty_memories_returns_zero(
        self,
        mock_semantic_backend: AsyncMock,
        write_lock: threading.Lock,
    ) -> None:
        """Storing empty memory list returns 0 without calling backend."""
        phase = DefaultStoragePhase(
            semantic_backend=mock_semantic_backend,
            fulltext_backend=None,
            write_lock=write_lock,
        )
        count = await phase.store([], "proj")

        assert count == 0
        mock_semantic_backend.add_batch.assert_not_called()

    async def test_store_to_semantic_only(
        self,
        mock_semantic_backend: AsyncMock,
        write_lock: threading.Lock,
    ) -> None:
        """Store memories to semantic backend only (no fulltext)."""
        phase = DefaultStoragePhase(
            semantic_backend=mock_semantic_backend,
            fulltext_backend=None,
            write_lock=write_lock,
        )
        memories = ["msg1", "msg2"]
        count = await phase.store(memories, "proj")

        assert count == 2
        mock_semantic_backend.add_batch.assert_called_once_with("proj", memories)

    async def test_store_to_both_backends(
        self,
        mock_semantic_backend: AsyncMock,
        mock_fulltext_backend: AsyncMock,
        write_lock: threading.Lock,
    ) -> None:
        """Store memories to both semantic and fulltext backends."""
        phase = DefaultStoragePhase(
            semantic_backend=mock_semantic_backend,
            fulltext_backend=mock_fulltext_backend,
            write_lock=write_lock,
        )
        memories = ["msg1", "msg2"]
        count = await phase.store(memories, "proj")

        assert count == 2
        mock_semantic_backend.add_batch.assert_called_once_with("proj", memories)
        assert mock_fulltext_backend.add.call_count == 2
        mock_fulltext_backend.add.assert_any_call("proj", "msg1")
        mock_fulltext_backend.add.assert_any_call("proj", "msg2")

    async def test_store_returns_count_from_semantic_backend(
        self,
        mock_fulltext_backend: AsyncMock,
        write_lock: threading.Lock,
    ) -> None:
        """Stored count comes from semantic backend result length."""
        semantic = AsyncMock()
        # Backend returns fewer memories than input (e.g. dedup)
        semantic.add_batch = AsyncMock(return_value=["msg1"])

        phase = DefaultStoragePhase(
            semantic_backend=semantic,
            fulltext_backend=mock_fulltext_backend,
            write_lock=write_lock,
        )
        count = await phase.store(["msg1", "msg2"], "proj")

        assert count == 1
        # Only the stored memory is synced to fulltext
        mock_fulltext_backend.add.assert_called_once_with("proj", "msg1")

    async def test_store_acquires_write_lock(
        self,
        mock_semantic_backend: AsyncMock,
        write_lock: threading.Lock,
    ) -> None:
        """Storage phase acquires the write lock during operation."""
        lock_acquired = False
        original_add_batch = mock_semantic_backend.add_batch

        async def check_lock(pid: str, msgs: list[str]) -> list[str]:
            nonlocal lock_acquired
            lock_acquired = write_lock.locked()
            return await original_add_batch(pid, msgs)

        mock_semantic_backend.add_batch = check_lock

        phase = DefaultStoragePhase(
            semantic_backend=mock_semantic_backend,
            fulltext_backend=None,
            write_lock=write_lock,
        )
        await phase.store(["msg"], "proj")

        assert lock_acquired is True


# ---------------------------------------------------------------------------
# AddPipeline
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAddPipeline:
    """Tests for AddPipeline orchestration."""

    def _make_pipeline(
        self,
        *,
        dedup_result: tuple[list[str], list[str], int] | None = None,
        replace_result: tuple[list[str], list[ReplacementInfo]] | None = None,
        store_result: int = 0,
        config: AddPipelineConfig | None = None,
    ) -> tuple[AddPipeline, AsyncMock, AsyncMock, AsyncMock]:
        """Helper to build a pipeline with mock phases."""
        dedup = AsyncMock()
        dedup.detect = AsyncMock(return_value=dedup_result or (["msg1", "msg2"], [], 0))

        replace = AsyncMock()
        replace.detect = AsyncMock(
            return_value=replace_result or (["msg1", "msg2"], [])
        )

        storage = AsyncMock()
        storage.store = AsyncMock(return_value=store_result)

        cfg = config or AddPipelineConfig(
            deduplicate_memories=True,
            enable_smart_replace=False,
            smart_replace_threshold=0.7,
        )
        mock_logger: IStructuredLogger = cast(
            IStructuredLogger, Mock(spec=StructuredLogger)
        )

        pipeline = AddPipeline(
            dedup_phase=dedup,
            replace_phase=replace,
            storage_phase=storage,
            config=cfg,
            logger=mock_logger,
        )
        return pipeline, dedup, replace, storage

    async def test_full_pipeline_no_duplicates_no_replace(self) -> None:
        """Full pipeline with no duplicates and no replacement."""
        pipeline, dedup, replace, storage = self._make_pipeline(
            dedup_result=(["msg1", "msg2"], [], 0),
            store_result=2,
        )
        result = await pipeline.execute(["msg1", "msg2"], "proj")

        assert result.stored_count == 2
        assert result.skipped_count == 0
        assert result.replaced_count == 0
        assert result.replacements == []
        dedup.detect.assert_called_once()
        storage.store.assert_called_once()

    async def test_pipeline_with_duplicates(self) -> None:
        """Pipeline correctly reports skipped duplicates."""
        pipeline, _, _, storage = self._make_pipeline(
            dedup_result=(["msg2"], ["msg1"], 1),
            store_result=1,
        )
        result = await pipeline.execute(["msg1", "msg2"], "proj")

        assert result.stored_count == 1
        assert result.skipped_count == 1

    async def test_dry_run_skips_storage(self) -> None:
        """Dry run returns counts without storing anything."""
        pipeline, dedup, replace, storage = self._make_pipeline(
            dedup_result=(["msg1", "msg2"], ["msg3"], 1),
        )
        result = await pipeline.execute(["msg1", "msg2", "msg3"], "proj", dry_run=True)

        assert result.stored_count == 2  # unique count
        assert result.skipped_count == 1  # duplicate count
        assert result.replaced_count == 0
        replace.detect.assert_not_called()
        storage.store.assert_not_called()

    async def test_smart_replace_enabled(self) -> None:
        """When smart replace is enabled, replacement phase runs."""
        replacement = ReplacementInfo(
            old_memory="old",
            new_memory="msg1",
            confidence=0.9,
            reason="updated",
        )
        pipeline, _, replace, storage = self._make_pipeline(
            dedup_result=(["msg1"], [], 0),
            replace_result=(["msg1"], [replacement]),
            store_result=1,
            config=AddPipelineConfig(
                deduplicate_memories=True,
                enable_smart_replace=True,
                smart_replace_threshold=0.7,
            ),
        )
        result = await pipeline.execute(["msg1"], "proj")

        assert result.stored_count == 1
        assert result.replaced_count == 1
        assert len(result.replacements) == 1
        assert result.replacements[0].confidence == 0.9
        replace.detect.assert_called_once_with(["msg1"], "proj")

    async def test_smart_replace_disabled_skips_phase(self) -> None:
        """When smart replace is disabled, replacement phase is skipped."""
        pipeline, _, replace, storage = self._make_pipeline(
            dedup_result=(["msg1"], [], 0),
            store_result=1,
            config=AddPipelineConfig(
                deduplicate_memories=True,
                enable_smart_replace=False,
                smart_replace_threshold=0.7,
            ),
        )
        result = await pipeline.execute(["msg1"], "proj")

        assert result.stored_count == 1
        assert result.replaced_count == 0
        assert result.replacements == []
        replace.detect.assert_not_called()

    async def test_pipeline_passes_project_id(self) -> None:
        """Pipeline forwards project_id to all phases."""
        pipeline, dedup, replace, storage = self._make_pipeline(
            dedup_result=(["msg"], [], 0),
            replace_result=(["msg"], []),
            store_result=1,
            config=AddPipelineConfig(
                deduplicate_memories=True,
                enable_smart_replace=True,
                smart_replace_threshold=0.7,
            ),
        )
        await pipeline.execute(["msg"], "test-project")

        dedup.detect.assert_called_once_with(["msg"], "test-project")
        replace.detect.assert_called_once_with(["msg"], "test-project")
        storage.store.assert_called_once_with(["msg"], "test-project")

    async def test_pipeline_empty_after_dedup(self) -> None:
        """When all memories are duplicates, storage receives empty list."""
        pipeline, _, _, storage = self._make_pipeline(
            dedup_result=([], ["msg1", "msg2"], 2),
            store_result=0,
        )
        result = await pipeline.execute(["msg1", "msg2"], "proj")

        assert result.stored_count == 0
        assert result.skipped_count == 2
        storage.store.assert_called_once_with([], "proj")

    async def test_pipeline_replace_reduces_memories(self) -> None:
        """Replacement phase can filter out memories."""
        pipeline, _, replace, storage = self._make_pipeline(
            dedup_result=(["msg1", "msg2"], [], 0),
            replace_result=(
                ["msg2"],
                [
                    ReplacementInfo(
                        old_memory="old",
                        new_memory="msg1",
                        confidence=0.95,
                        reason="replaced",
                    ),
                ],
            ),
            store_result=1,
            config=AddPipelineConfig(
                deduplicate_memories=True,
                enable_smart_replace=True,
                smart_replace_threshold=0.7,
            ),
        )
        result = await pipeline.execute(["msg1", "msg2"], "proj")

        assert result.stored_count == 1
        assert result.replaced_count == 1
        # Storage receives filtered memories from replace phase
        storage.store.assert_called_once_with(["msg2"], "proj")


# ---------------------------------------------------------------------------
# create_default_pipeline factory
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateDefaultPipeline:
    """Tests for create_default_pipeline factory function."""

    def _make_config(
        self,
        *,
        deduplicate: bool = True,
        smart_replace: bool = False,
        smart_replace_threshold: float = 0.7,
    ) -> Mock:
        """Create a minimal mock Config with required fields."""
        config = Mock(spec=Config)
        config.deduplicate_memories = deduplicate
        config.enable_smart_replace = smart_replace
        config.smart_replace_threshold = smart_replace_threshold
        return config

    def test_creates_pipeline_without_replace(
        self,
        mock_semantic_backend: AsyncMock,
        mock_fulltext_backend: AsyncMock,
        mock_logger: Mock,
        write_lock: threading.Lock,
    ) -> None:
        """Factory creates pipeline with noop replacement when disabled."""
        config = self._make_config(smart_replace=False)

        pipeline = create_default_pipeline(
            semantic_backend=mock_semantic_backend,
            fulltext_backend=mock_fulltext_backend,
            replace_detector=None,
            config=config,
            logger=mock_logger,
            write_lock=write_lock,
        )

        assert isinstance(pipeline, AddPipeline)
        assert isinstance(pipeline._replace_phase, NoopReplacementDetectionPhase)

    def test_creates_pipeline_with_replace(
        self,
        mock_semantic_backend: AsyncMock,
        mock_fulltext_backend: AsyncMock,
        mock_logger: Mock,
        write_lock: threading.Lock,
    ) -> None:
        """Factory creates pipeline with replacement when enabled."""
        config = self._make_config(smart_replace=True)
        mock_reranker = AsyncMock()
        mock_reranker.name = "test_reranker"

        pipeline = create_default_pipeline(
            semantic_backend=mock_semantic_backend,
            fulltext_backend=mock_fulltext_backend,
            replace_detector=mock_reranker,
            config=config,
            logger=mock_logger,
            write_lock=write_lock,
        )

        assert isinstance(pipeline, AddPipeline)
        assert isinstance(pipeline._replace_phase, DefaultReplacementDetectionPhase)

    def test_noop_replace_when_enabled_but_no_detector(
        self,
        mock_semantic_backend: AsyncMock,
        mock_fulltext_backend: AsyncMock,
        mock_logger: Mock,
        write_lock: threading.Lock,
    ) -> None:
        """Noop replacement when smart replace enabled but no detector."""
        config = self._make_config(smart_replace=True)

        pipeline = create_default_pipeline(
            semantic_backend=mock_semantic_backend,
            fulltext_backend=mock_fulltext_backend,
            replace_detector=None,
            config=config,
            logger=mock_logger,
            write_lock=write_lock,
        )

        assert isinstance(pipeline._replace_phase, NoopReplacementDetectionPhase)

    def test_creates_pipeline_without_fulltext(
        self,
        mock_semantic_backend: AsyncMock,
        mock_logger: Mock,
        write_lock: threading.Lock,
    ) -> None:
        """Factory creates pipeline when fulltext backend is None."""
        config = self._make_config()

        pipeline = create_default_pipeline(
            semantic_backend=mock_semantic_backend,
            fulltext_backend=None,
            replace_detector=None,
            config=config,
            logger=mock_logger,
            write_lock=write_lock,
        )

        assert isinstance(pipeline, AddPipeline)
        assert isinstance(pipeline._dedup_phase, DefaultDuplicateDetectionPhase)
        assert isinstance(pipeline._storage_phase, DefaultStoragePhase)

    def test_pipeline_config_mirrors_app_config(
        self,
        mock_semantic_backend: AsyncMock,
        mock_logger: Mock,
        write_lock: threading.Lock,
    ) -> None:
        """Pipeline config fields match source Config values."""
        config = self._make_config(
            deduplicate=False,
            smart_replace=True,
            smart_replace_threshold=0.85,
        )
        mock_reranker = AsyncMock()
        mock_reranker.name = "test"

        pipeline = create_default_pipeline(
            semantic_backend=mock_semantic_backend,
            fulltext_backend=None,
            replace_detector=mock_reranker,
            config=config,
            logger=mock_logger,
            write_lock=write_lock,
        )

        assert pipeline._config.deduplicate_memories is False
        assert pipeline._config.enable_smart_replace is True
        assert pipeline._config.smart_replace_threshold == 0.85

    def test_dedup_phase_receives_backends(
        self,
        mock_semantic_backend: AsyncMock,
        mock_fulltext_backend: AsyncMock,
        mock_logger: Mock,
        write_lock: threading.Lock,
    ) -> None:
        """Dedup phase is configured with correct backends."""
        config = self._make_config()

        pipeline = create_default_pipeline(
            semantic_backend=mock_semantic_backend,
            fulltext_backend=mock_fulltext_backend,
            replace_detector=None,
            config=config,
            logger=mock_logger,
            write_lock=write_lock,
        )

        dedup = pipeline._dedup_phase
        assert isinstance(dedup, DefaultDuplicateDetectionPhase)
        assert dedup._semantic is mock_semantic_backend
        assert dedup._fulltext is mock_fulltext_backend

    def test_storage_phase_receives_lock_and_backends(
        self,
        mock_semantic_backend: AsyncMock,
        mock_fulltext_backend: AsyncMock,
        mock_logger: Mock,
        write_lock: threading.Lock,
    ) -> None:
        """Storage phase is configured with correct backends and lock."""
        config = self._make_config()

        pipeline = create_default_pipeline(
            semantic_backend=mock_semantic_backend,
            fulltext_backend=mock_fulltext_backend,
            replace_detector=None,
            config=config,
            logger=mock_logger,
            write_lock=write_lock,
        )

        storage = pipeline._storage_phase
        assert isinstance(storage, DefaultStoragePhase)
        assert storage._semantic is mock_semantic_backend
        assert storage._fulltext is mock_fulltext_backend
        assert storage._write_lock is write_lock
