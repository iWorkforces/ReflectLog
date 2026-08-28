#!/usr/bin/env python3
"""Unit tests for add_phases.py - 3-phase parallel add pipeline.

Covers uncovered lines: 291, 417-524, 588-663, 670, 676, 719,
723-731, 759-768, 782-815, 824, 850-900.
"""

import threading
from typing import cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.core.exceptions import StorageError
from reflectlog.core.types import ReplacementTransition
from reflectlog.application.memory.add_phases import (
    AddPipeline,
    AddResult,
    DuplicateDetectionPhase,
    Phase1Result,
    Phase2Result,
    Phase3Result,
    ReplacementInfo,
    SmartReplacementPhase,
    StoragePhase,
)
from reflectlog.application.utils.logging import StructuredLogger
from reflectlog.core.logging import IStructuredLogger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config() -> Config:
    """Minimal mock Config for add_phases tests."""
    config = Mock(spec=Config)
    config.workspace_id = "test_project"
    config.add_max_concurrency = 4
    config.rerank_max_concurrency = 5
    config.smart_replace_candidate_limit = 3
    config.smart_replace_min_similarity = 0.5
    config.enable_llm_infer = False
    config.deduplicate_memories = True
    return config


@pytest.fixture
def mock_logger():
    """Mock structured logger."""
    return cast(IStructuredLogger, Mock(spec=StructuredLogger))


@pytest.fixture
def mock_semantic_engine():
    """Mock ISemanticSearchEngine."""
    engine = MagicMock()
    engine.add = MagicMock(return_value=None)
    engine.add_batch = MagicMock(
        side_effect=lambda workspace_id, contents, infer: contents
    )
    engine.search = MagicMock(return_value=[])
    engine.delete = MagicMock(return_value=None)
    engine.commit = MagicMock(return_value=None)
    engine.get_id_by_content = MagicMock(return_value=None)
    engine.memory_store = MagicMock()
    return engine


@pytest.fixture
def mock_tantivy_engine():
    """Mock TantivyEngine."""
    engine = MagicMock()
    engine.add = MagicMock(return_value=None)
    engine.delete = MagicMock(return_value=None)
    engine.commit = MagicMock(return_value=None)
    engine.search = MagicMock(return_value=[])
    return engine


# ---------------------------------------------------------------------------
# Dataclass Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDataclasses:
    """Tests for dataclass constructors and defaults."""

    def test_replacement_info_defaults(self):
        """ReplacementInfo should have default similarity_score of 0.0."""
        info = ReplacementInfo(
            old_memory="old",
            new_memory="new",
            confidence=0.9,
            reason="updated",
        )
        assert info.similarity_score == 0.0
        assert info.old_memory == "old"
        assert info.confidence == 0.9

    def test_add_result_defaults(self):
        """AddResult defaults should be zero counts and empty list."""
        result = AddResult()
        assert result.stored_count == 0
        assert result.skipped_count == 0
        assert result.replaced_count == 0
        assert result.replacements == []

    def test_phase1_result_fields(self):
        """Phase1Result should store all fields."""
        result = Phase1Result(
            unique_memories=["a", "b"],
            storage_duplicates=["c"],
            batch_duplicates_count=1,
            duration=0.5,
        )
        assert len(result.unique_memories) == 2
        assert result.batch_duplicates_count == 1

    def test_phase2_result_fields(self):
        """Phase2Result should store all fields."""
        result = Phase2Result(
            replacement_map={},
            total_replacements=0,
            duration=0.1,
        )
        assert result.total_replacements == 0

    def test_phase3_result_fields(self):
        """Phase3Result should store all fields."""
        result = Phase3Result(
            stored_count=2,
            replaced_count=1,
            replacements=[],
            duration=0.3,
        )
        assert result.stored_count == 2
        assert result.replaced_count == 1


# ---------------------------------------------------------------------------
# DuplicateDetectionPhase Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDuplicateDetectionPhase:
    """Tests for Phase 1: Duplicate Detection."""

    async def test_execute_no_duplicates(
        self, mock_semantic_engine, mock_tantivy_engine, mock_config, mock_logger
    ):
        """All unique memories should pass through."""
        phase = DuplicateDetectionPhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )

        # No exact match found
        mock_semantic_engine.get_id_by_content.return_value = None
        mock_tantivy_engine.search.return_value = []

        result = await phase.execute(["msg1", "msg2"])

        assert len(result.unique_memories) == 2
        assert result.batch_duplicates_count == 0
        assert len(result.storage_duplicates) == 0

    async def test_execute_with_batch_duplicates(
        self, mock_semantic_engine, mock_tantivy_engine, mock_config, mock_logger
    ):
        """Duplicate memories within batch should be deduplicated."""
        phase = DuplicateDetectionPhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        mock_tantivy_engine.search.return_value = []

        result = await phase.execute(["msg1", "msg2", "msg1", "msg2"])

        assert len(result.unique_memories) == 2
        assert result.batch_duplicates_count == 2
        mock_logger.info.assert_called()

    async def test_execute_with_storage_duplicates(
        self, mock_semantic_engine, mock_tantivy_engine, mock_config, mock_logger
    ):
        """Memories already in storage should be marked as duplicates."""
        phase = DuplicateDetectionPhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )

        # Tantivy finds exact match for "existing"
        def tantivy_search(query, workspace_id, limit):
            if "existing" in query:
                return [("existing", 1.0)]
            return []

        mock_tantivy_engine.search.side_effect = tantivy_search

        result = await phase.execute(["existing", "new_msg"])

        assert "new_msg" in result.unique_memories
        assert "existing" in result.storage_duplicates


# ---------------------------------------------------------------------------
# SmartReplacementPhase Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSmartReplacementPhase:
    """Tests for Phase 2: Smart Replacement Detection."""

    def test_get_smart_replacer_when_memory_manager_is_none(
        self, mock_semantic_engine, mock_config, mock_logger
    ):
        """_get_smart_replacer returns None when memory_manager is None (line 291)."""
        phase = SmartReplacementPhase(
            semantic_engine=mock_semantic_engine,
            config=mock_config,
            logger=mock_logger,
            memory_manager=None,
        )
        assert phase._get_smart_replacer() is None

    def test_get_smart_replacer_returns_replacer(
        self, mock_semantic_engine, mock_config, mock_logger
    ):
        """_get_smart_replacer should delegate to memory_manager.smart_replacer."""
        mock_mm = MagicMock()
        mock_mm.smart_replacer = MagicMock()
        phase = SmartReplacementPhase(
            semantic_engine=mock_semantic_engine,
            config=mock_config,
            logger=mock_logger,
            memory_manager=mock_mm,
        )
        assert phase._get_smart_replacer() is mock_mm.smart_replacer

    async def test_execute_no_smart_replacer(
        self, mock_semantic_engine, mock_config, mock_logger
    ):
        """Phase 2 should return empty results when no smart replacer (line 311-316)."""
        phase = SmartReplacementPhase(
            semantic_engine=mock_semantic_engine,
            config=mock_config,
            logger=mock_logger,
            memory_manager=None,
        )
        result = await phase.execute(["msg1"])

        assert result.total_replacements == 0
        assert result.replacement_map == {}

    async def test_execute_empty_memories(
        self, mock_semantic_engine, mock_config, mock_logger
    ):
        """Phase 2 should return early on empty memories list."""
        mock_mm = MagicMock()
        mock_mm.smart_replacer = MagicMock()
        phase = SmartReplacementPhase(
            semantic_engine=mock_semantic_engine,
            config=mock_config,
            logger=mock_logger,
            memory_manager=mock_mm,
        )
        result = await phase.execute([])

        assert result.total_replacements == 0

    async def test_check_for_replacement_no_similar_results(
        self, mock_semantic_engine, mock_config, mock_logger
    ):
        """No similar results from semantic search returns empty list."""
        mock_mm = MagicMock()
        mock_replacer = MagicMock()
        mock_mm.smart_replacer = mock_replacer
        phase = SmartReplacementPhase(
            semantic_engine=mock_semantic_engine,
            config=mock_config,
            logger=mock_logger,
            memory_manager=mock_mm,
        )
        mock_semantic_engine.search.return_value = []

        result = await phase._check_for_replacement("new memory", mock_replacer)

        assert result == []

    async def test_check_for_replacement_below_threshold(
        self, mock_semantic_engine, mock_config, mock_logger
    ):
        """Candidates below similarity threshold should be filtered out."""
        mock_mm = MagicMock()
        mock_replacer = MagicMock()
        mock_mm.smart_replacer = mock_replacer
        phase = SmartReplacementPhase(
            semantic_engine=mock_semantic_engine,
            config=mock_config,
            logger=mock_logger,
            memory_manager=mock_mm,
        )
        # Score 0.3 is below threshold of 0.5
        mock_semantic_engine.search.return_value = [
            ("old memory", 0.3, "2024-01-01T00:00:00"),
        ]

        result = await phase._check_for_replacement("new memory", mock_replacer)

        assert result == []

    async def test_check_for_replacement_same_message_filtered(
        self, mock_semantic_engine, mock_config, mock_logger
    ):
        """Candidates that are identical to new_memory should be filtered out."""
        mock_mm = MagicMock()
        mock_replacer = MagicMock()
        mock_mm.smart_replacer = mock_replacer
        phase = SmartReplacementPhase(
            semantic_engine=mock_semantic_engine,
            config=mock_config,
            logger=mock_logger,
            memory_manager=mock_mm,
        )
        # Same memory with high score - should be filtered (mem != new_memory)
        mock_semantic_engine.search.return_value = [
            ("new memory", 1.0, "2024-01-01T00:00:00"),
        ]

        result = await phase._check_for_replacement("new memory", mock_replacer)

        assert result == []

    async def test_check_for_replacement_should_replace(
        self, mock_semantic_engine, mock_config, mock_logger
    ):
        """LLM confirms replacement - should return ReplacementInfo (lines 417-471)."""
        mock_mm = MagicMock()
        mock_replacer = MagicMock()
        mock_mm.smart_replacer = mock_replacer
        phase = SmartReplacementPhase(
            semantic_engine=mock_semantic_engine,
            config=mock_config,
            logger=mock_logger,
            memory_manager=mock_mm,
        )
        mock_semantic_engine.search.return_value = [
            ("old memory about python", 0.85, "2024-01-01T00:00:00"),
        ]
        # LLM says replace
        mock_replacer.check_replacement = AsyncMock(
            return_value=(True, 0.95, "Updated information")
        )

        result = await phase._check_for_replacement(
            "new memory about python", mock_replacer
        )

        assert len(result) == 1
        assert result[0].old_memory == "old memory about python"
        assert result[0].new_memory == "new memory about python"
        assert result[0].confidence == 0.95
        assert result[0].reason == "Updated information"
        assert result[0].similarity_score == 0.85

    async def test_check_for_replacement_should_not_replace(
        self, mock_semantic_engine, mock_config, mock_logger
    ):
        """LLM says no replacement - returns empty list (lines 472-482)."""
        mock_mm = MagicMock()
        mock_replacer = MagicMock()
        mock_mm.smart_replacer = mock_replacer
        phase = SmartReplacementPhase(
            semantic_engine=mock_semantic_engine,
            config=mock_config,
            logger=mock_logger,
            memory_manager=mock_mm,
        )
        mock_semantic_engine.search.return_value = [
            ("old memory", 0.85, "2024-01-01T00:00:00"),
        ]
        mock_replacer.check_replacement = AsyncMock(
            return_value=(False, 0.3, "Different topic")
        )

        result = await phase._check_for_replacement("new memory", mock_replacer)

        assert result == []

    async def test_check_for_replacement_llm_error_graceful(
        self, mock_semantic_engine, mock_config, mock_logger
    ):
        """LLM call failure should be handled gracefully (lines 483-493)."""
        mock_mm = MagicMock()
        mock_replacer = MagicMock()
        mock_mm.smart_replacer = mock_replacer
        phase = SmartReplacementPhase(
            semantic_engine=mock_semantic_engine,
            config=mock_config,
            logger=mock_logger,
            memory_manager=mock_mm,
        )
        mock_semantic_engine.search.return_value = [
            ("old memory", 0.85, "2024-01-01T00:00:00"),
        ]
        mock_replacer.check_replacement = AsyncMock(
            side_effect=RuntimeError("LLM API timeout")
        )

        result = await phase._check_for_replacement("new memory", mock_replacer)

        assert result == []
        mock_logger.warning.assert_called()

    async def test_check_for_replacement_outer_exception(
        self, mock_semantic_engine, mock_config, mock_logger
    ):
        """Outer exception in _check_for_replacement is caught (lines 514-524)."""
        mock_mm = MagicMock()
        mock_replacer = MagicMock()
        mock_mm.smart_replacer = mock_replacer
        phase = SmartReplacementPhase(
            semantic_engine=mock_semantic_engine,
            config=mock_config,
            logger=mock_logger,
            memory_manager=mock_mm,
        )
        # Semantic search itself throws
        mock_semantic_engine.search.side_effect = RuntimeError("Search engine failed")

        result = await phase._check_for_replacement("new memory", mock_replacer)

        assert result == []
        mock_logger.warning.assert_called()

    async def test_execute_with_replacements(
        self, mock_semantic_engine, mock_config, mock_logger
    ):
        """Full execute with replacement detection (lines 417-510)."""
        mock_mm = MagicMock()
        mock_replacer = MagicMock()
        mock_mm.smart_replacer = mock_replacer
        phase = SmartReplacementPhase(
            semantic_engine=mock_semantic_engine,
            config=mock_config,
            logger=mock_logger,
            memory_manager=mock_mm,
        )
        mock_semantic_engine.search.return_value = [
            ("old memory", 0.85, "2024-01-01T00:00:00"),
        ]
        mock_replacer.check_replacement = AsyncMock(
            return_value=(True, 0.9, "Updated info")
        )

        result = await phase.execute(["new memory"])

        assert result.total_replacements == 1
        assert "new memory" in result.replacement_map
        assert len(result.replacement_map["new memory"]) == 1

    async def test_execute_multiple_candidates(
        self, mock_semantic_engine, mock_config, mock_logger
    ):
        """Multiple candidates above threshold are all checked (lines 505-510)."""
        mock_mm = MagicMock()
        mock_replacer = MagicMock()
        mock_mm.smart_replacer = mock_replacer
        phase = SmartReplacementPhase(
            semantic_engine=mock_semantic_engine,
            config=mock_config,
            logger=mock_logger,
            memory_manager=mock_mm,
        )
        mock_semantic_engine.search.return_value = [
            ("candidate 1", 0.9, "2024-01-01T00:00:00"),
            ("candidate 2", 0.85, "2024-01-02T00:00:00"),
        ]
        # First candidate should be replaced, second not
        call_count = 0

        def mock_check(new_memory, existing_memory):
            nonlocal call_count
            call_count += 1
            if existing_memory == "candidate 1":
                return (True, 0.95, "Replace this")
            return (False, 0.2, "Keep this")

        mock_replacer.check_replacement = AsyncMock(side_effect=mock_check)

        result = await phase.execute(["new memory"])

        assert result.total_replacements == 1


# ---------------------------------------------------------------------------
# StoragePhase Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStoragePhase:
    """Tests for Phase 3: Sequential Storage."""

    async def test_execute_no_replacements(
        self, mock_semantic_engine, mock_tantivy_engine, mock_config, mock_logger
    ):
        """Simple storage without replacements."""
        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        result = await phase.execute(["msg1", "msg2"], replacement_map={})

        assert result.stored_count == 2
        assert result.replaced_count == 0
        mock_semantic_engine.add_batch.assert_called_once()
        mock_semantic_engine.commit.assert_called_once()
        mock_tantivy_engine.commit.assert_called_once()

    async def test_execute_dry_run_no_storage(
        self, mock_semantic_engine, mock_tantivy_engine, mock_config, mock_logger
    ):
        """Dry run should not write to storage (line 670)."""
        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        result = await phase.execute(["msg1"], replacement_map={}, dry_run=True)

        assert result.stored_count == 1
        mock_semantic_engine.add_batch.assert_not_called()
        mock_semantic_engine.commit.assert_not_called()
        mock_tantivy_engine.commit.assert_not_called()

    async def test_execute_dry_run_with_replacements(
        self, mock_semantic_engine, mock_tantivy_engine, mock_config, mock_logger
    ):
        """Dry run should record replacements without deleting (lines 660-663)."""
        replacement_info = ReplacementInfo(
            old_memory="old msg",
            new_memory="new msg",
            confidence=0.9,
            reason="updated",
            similarity_score=0.85,
        )
        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        result = await phase.execute(
            ["new msg"],
            replacement_map={"new msg": [replacement_info]},
            dry_run=True,
        )

        assert result.replaced_count == 1
        assert len(result.replacements) == 1
        assert result.stored_count == 1
        # No actual deletes or durable records in dry_run
        mock_semantic_engine.delete.assert_not_called()
        mock_semantic_engine.memory_store.begin_replacement_transition.assert_not_called()
        mock_semantic_engine.memory_store.archive.assert_not_called()

    async def test_execute_with_replacement_success(
        self, mock_semantic_engine, mock_tantivy_engine, mock_config, mock_logger
    ):
        """Replacement flow: archive + delete + add (lines 588-649)."""
        replacement_info = ReplacementInfo(
            old_memory="old msg",
            new_memory="new msg",
            confidence=0.9,
            reason="updated",
            similarity_score=0.85,
        )
        mock_semantic_engine.get_id_by_content.side_effect = (
            lambda _pid, content: 42 if content == "old msg" else 99
        )
        mock_semantic_engine.index = {99}
        mock_tantivy_engine.find_by_exact_match.side_effect = (
            lambda _pid, content: [content] if content == "new msg" else []
        )
        mock_semantic_engine.memory_store.begin_replacement_transition.return_value = (
            ReplacementTransition(
                id=9,
                workspace_id="test_project",
                old_memory_id=42,
                old_content="old msg",
                new_content="new msg",
                archive_id=100,
                reason="updated",
                confidence=0.9,
                status="pending",
            )
        )

        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        result = await phase.execute(
            ["new msg"],
            replacement_map={"new msg": [replacement_info]},
        )

        assert result.replaced_count == 1
        assert result.stored_count == 1
        assert len(result.replacements) == 1
        mock_semantic_engine.memory_store.begin_replacement_transition.assert_called_once()
        mock_semantic_engine.delete.assert_called_once_with(memory_id="42")
        mock_tantivy_engine.delete.assert_called_once_with("test_project", "old msg")
        mock_semantic_engine.memory_store.complete_replacement_transition.assert_called_once_with(
            9
        )

    async def test_execute_replacement_old_memory_not_found(
        self, mock_semantic_engine, mock_tantivy_engine, mock_config, mock_logger
    ):
        """When old memory ID not found, skip replacement (lines 625-633)."""
        replacement_info = ReplacementInfo(
            old_memory="old msg",
            new_memory="new msg",
            confidence=0.9,
            reason="updated",
        )
        mock_semantic_engine.get_id_by_content.return_value = None
        mock_semantic_engine.memory_store.archive.return_value = 100

        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        result = await phase.execute(
            ["new msg"],
            replacement_map={"new msg": [replacement_info]},
        )

        # Replacement skipped because msg_id was None
        assert result.replaced_count == 0
        mock_semantic_engine.delete.assert_not_called()
        mock_semantic_engine.memory_store.begin_replacement_transition.assert_not_called()
        mock_logger.warning.assert_called()

    async def test_execute_replacement_delete_error(
        self, mock_semantic_engine, mock_tantivy_engine, mock_config, mock_logger
    ):
        """Delete error during replacement is caught gracefully (lines 650-659)."""
        replacement_info = ReplacementInfo(
            old_memory="old msg",
            new_memory="new msg",
            confidence=0.9,
            reason="updated",
        )
        mock_semantic_engine.get_id_by_content.return_value = 42
        mock_semantic_engine.index = set()
        mock_semantic_engine.memory_store.begin_replacement_transition.return_value = (
            ReplacementTransition(
                id=9,
                workspace_id="test_project",
                old_memory_id=42,
                old_content="old msg",
                new_content="new msg",
                archive_id=100,
                reason="updated",
                confidence=0.9,
                status="pending",
            )
        )
        mock_semantic_engine.delete.side_effect = RuntimeError("Delete failed")

        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        with pytest.raises(StorageError, match="Failed to delete old memory"):
            await phase.execute(
                ["new msg"],
                replacement_map={"new msg": [replacement_info]},
            )

        mock_semantic_engine.memory_store.complete_replacement_transition.assert_not_called()
        mock_logger.warning.assert_called()

    async def test_execute_batch_stored_fewer_than_expected(
        self, mock_semantic_engine, mock_tantivy_engine, mock_config, mock_logger
    ):
        """Warning when batch add returns fewer memories (line 676)."""
        # Clear side_effect from fixture so return_value is used
        mock_semantic_engine.add_batch.side_effect = None
        mock_semantic_engine.add_batch.return_value = ["msg1"]

        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        result = await phase.execute(["msg1", "msg2"], replacement_map={})

        assert result.stored_count == 1
        mock_logger.warning.assert_called()

    async def test_execute_no_tantivy(
        self, mock_semantic_engine, mock_config, mock_logger
    ):
        """Storage without tantivy engine should still work."""
        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=None,
            config=mock_config,
            logger=mock_logger,
        )
        result = await phase.execute(["msg1"], replacement_map={})

        assert result.stored_count == 1
        mock_semantic_engine.commit.assert_called_once()

    # -----------------------------------------------------------------------
    # _add_memories_batch Tests (lines 709-755)
    # -----------------------------------------------------------------------

    def test_add_memories_batch_empty(
        self, mock_semantic_engine, mock_tantivy_engine, mock_config, mock_logger
    ):
        """Empty memories list returns empty (line 719)."""
        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        result = phase._add_memories_batch([])
        assert result == []

    def test_add_memories_batch_no_write_lock(
        self, mock_semantic_engine, mock_tantivy_engine, mock_config, mock_logger
    ):
        """Batch add without write_lock (lines 723-731)."""
        mock_semantic_engine.add_batch.return_value = ["msg1", "msg2"]

        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
            write_lock=None,
        )
        result = phase._add_memories_batch(["msg1", "msg2"])

        assert result == ["msg1", "msg2"]
        mock_semantic_engine.add_batch.assert_called_once_with(
            workspace_id="test_project",
            contents=["msg1", "msg2"],
            infer=False,
        )
        assert mock_tantivy_engine.add.call_count == 2

    def test_add_memories_batch_no_write_lock_no_tantivy(
        self, mock_semantic_engine, mock_config, mock_logger
    ):
        """Batch add without write_lock and no tantivy (line 729-731 skip)."""
        mock_semantic_engine.add_batch.return_value = ["msg1"]

        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=None,
            config=mock_config,
            logger=mock_logger,
            write_lock=None,
        )
        result = phase._add_memories_batch(["msg1"])

        assert result == ["msg1"]

    def test_add_memories_batch_with_write_lock(
        self, mock_semantic_engine, mock_tantivy_engine, mock_config, mock_logger
    ):
        """Batch add with write_lock (lines 733-742)."""
        mock_semantic_engine.add_batch.return_value = ["msg1"]
        lock = threading.Lock()

        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
            write_lock=lock,
        )
        result = phase._add_memories_batch(["msg1"])

        assert result == ["msg1"]
        mock_tantivy_engine.add.assert_called_once()

    def test_add_memories_batch_with_write_lock_no_tantivy(
        self, mock_semantic_engine, mock_config, mock_logger
    ):
        """Batch add with write_lock but no tantivy (line 740-742 skip)."""
        mock_semantic_engine.add_batch.return_value = ["msg1"]
        lock = threading.Lock()

        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=None,
            config=mock_config,
            logger=mock_logger,
            write_lock=lock,
        )
        result = phase._add_memories_batch(["msg1"])

        assert result == ["msg1"]

    def test_add_memories_batch_exception_raises_storage_error(
        self, mock_semantic_engine, mock_tantivy_engine, mock_config, mock_logger
    ):
        """Exception in batch add raises StorageError (line 754-755)."""
        mock_semantic_engine.add_batch.side_effect = RuntimeError("DB error")

        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        with pytest.raises(StorageError, match="Failed to add memory batch"):
            phase._add_memories_batch(["msg1"])

    # -----------------------------------------------------------------------
    # _delete_memory Tests (lines 757-768)
    # -----------------------------------------------------------------------

    def test_delete_memory_no_write_lock(
        self, mock_semantic_engine, mock_tantivy_engine, mock_config, mock_logger
    ):
        """Delete without write_lock (lines 759-763)."""
        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
            write_lock=None,
        )
        phase._delete_memory(memory_id="42", content="old msg")

        mock_semantic_engine.delete.assert_called_once_with(memory_id="42")
        mock_tantivy_engine.delete.assert_called_once_with("test_project", "old msg")

    def test_delete_memory_no_write_lock_no_tantivy(
        self, mock_semantic_engine, mock_config, mock_logger
    ):
        """Delete without write_lock and no tantivy (line 761-762 skip)."""
        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=None,
            config=mock_config,
            logger=mock_logger,
            write_lock=None,
        )
        phase._delete_memory(memory_id="42", content="old msg")

        mock_semantic_engine.delete.assert_called_once_with(memory_id="42")

    def test_delete_memory_with_write_lock(
        self, mock_semantic_engine, mock_tantivy_engine, mock_config, mock_logger
    ):
        """Delete with write_lock (lines 765-768)."""
        lock = threading.Lock()
        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
            write_lock=lock,
        )
        phase._delete_memory(memory_id="42", content="old msg")

        mock_semantic_engine.delete.assert_called_once_with(memory_id="42")
        mock_tantivy_engine.delete.assert_called_once_with("test_project", "old msg")

    def test_delete_memory_with_write_lock_no_tantivy(
        self, mock_semantic_engine, mock_config, mock_logger
    ):
        """Delete with write_lock but no tantivy (line 767-768 skip)."""
        lock = threading.Lock()
        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=None,
            config=mock_config,
            logger=mock_logger,
            write_lock=lock,
        )
        phase._delete_memory(memory_id="42", content="old msg")

        mock_semantic_engine.delete.assert_called_once_with(memory_id="42")

    # -----------------------------------------------------------------------
    # _add_memory Tests (lines 770-815)
    # -----------------------------------------------------------------------

    def test_add_memory_success(
        self, mock_semantic_engine, mock_tantivy_engine, mock_config, mock_logger
    ):
        """Single memory add succeeds (lines 792-812)."""
        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        result = phase._add_memory("new msg")

        assert result is True
        mock_semantic_engine.add.assert_called_once_with(
            workspace_id="test_project",
            content="new msg",
            infer=False,
        )
        mock_tantivy_engine.add.assert_called_once_with("test_project", "new msg")

    def test_add_memory_success_no_tantivy(
        self, mock_semantic_engine, mock_config, mock_logger
    ):
        """Single memory add without tantivy (line 801 skip)."""
        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=None,
            config=mock_config,
            logger=mock_logger,
        )
        result = phase._add_memory("new msg")

        assert result is True
        mock_semantic_engine.add.assert_called_once()

    def test_add_memory_duplicate_skipped(
        self, mock_semantic_engine, mock_tantivy_engine, mock_config, mock_logger
    ):
        """Duplicate memory returns False (lines 782-790)."""
        mock_config.deduplicate_memories = True
        # Tantivy finds exact match
        mock_tantivy_engine.search.return_value = [("dup msg", 1.0)]

        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        result = phase._add_memory("dup msg")

        assert result is False
        mock_semantic_engine.add.assert_not_called()

    def test_add_memory_dedup_disabled(
        self, mock_semantic_engine, mock_tantivy_engine, mock_config, mock_logger
    ):
        """When dedup is disabled, memory is always added."""
        mock_config.deduplicate_memories = False

        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        result = phase._add_memory("new msg")

        assert result is True

    def test_add_memory_exception_raises_storage_error(
        self, mock_semantic_engine, mock_tantivy_engine, mock_config, mock_logger
    ):
        """Exception in _add_memory raises StorageError (lines 814-815)."""
        mock_semantic_engine.add.side_effect = RuntimeError("Insert failed")

        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        with pytest.raises(
            StorageError, match="Failed to add memory to hybrid storage"
        ):
            phase._add_memory("new msg")

    # -----------------------------------------------------------------------
    # _has_exact_match Tests (line 824)
    # -----------------------------------------------------------------------

    def test_has_exact_match_delegates(
        self, mock_semantic_engine, mock_tantivy_engine, mock_config, mock_logger
    ):
        """_has_exact_match delegates to match_utils.has_exact_match (line 824)."""
        mock_tantivy_engine.search.return_value = [("some msg", 1.0)]

        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        result = phase._has_exact_match("some msg")

        assert result is True

    def test_has_exact_match_no_match(
        self, mock_semantic_engine, mock_tantivy_engine, mock_config, mock_logger
    ):
        """_has_exact_match returns False when no match found."""
        mock_tantivy_engine.search.return_value = []
        mock_semantic_engine.get_id_by_content.return_value = None

        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        result = phase._has_exact_match("nonexistent msg")

        assert result is False

    # -----------------------------------------------------------------------
    # _record_replacement_transition Tests
    # -----------------------------------------------------------------------

    def test_record_replacement_transition_success(
        self, mock_semantic_engine, mock_config, mock_logger
    ):
        """Successful transition record returns the stored row."""
        transition = ReplacementTransition(
            id=9,
            workspace_id="test_project",
            old_memory_id=42,
            old_content="old msg",
            new_content="new msg",
            archive_id=100,
            reason="updated",
            confidence=0.9,
            status="pending",
        )
        mock_semantic_engine.get_id_by_content.return_value = 42
        mock_semantic_engine.memory_store.begin_replacement_transition.return_value = (
            transition
        )

        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=None,
            config=mock_config,
            logger=mock_logger,
        )
        info = ReplacementInfo(
            old_memory="old msg",
            new_memory="new msg",
            confidence=0.9,
            reason="updated",
        )
        result = phase._record_replacement_transition(info, "new msg")

        assert result is transition
        mock_semantic_engine.memory_store.begin_replacement_transition.assert_called_once_with(
            old_memory_id=42,
            workspace_id="test_project",
            old_content="old msg",
            new_content="new msg",
            reason="updated",
            confidence=0.9,
        )
        mock_logger.info.assert_called()

    def test_record_replacement_transition_mem_not_found(
        self, mock_semantic_engine, mock_config, mock_logger
    ):
        """Recording returns None when the old memory is missing."""
        mock_semantic_engine.get_id_by_content.return_value = None

        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=None,
            config=mock_config,
            logger=mock_logger,
        )
        info = ReplacementInfo(
            old_memory="missing msg",
            new_memory="new msg",
            confidence=0.9,
            reason="updated",
        )
        result = phase._record_replacement_transition(info, "new msg")

        assert result is None
        mock_semantic_engine.memory_store.begin_replacement_transition.assert_not_called()
        mock_logger.warning.assert_called()

    def test_record_replacement_transition_exception(
        self, mock_semantic_engine, mock_config, mock_logger
    ):
        """Store failure raises StorageError and does not get swallowed."""
        mock_semantic_engine.get_id_by_content.return_value = 42
        mock_semantic_engine.memory_store.begin_replacement_transition.side_effect = (
            StorageError("archive failed")
        )

        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=None,
            config=mock_config,
            logger=mock_logger,
        )
        info = ReplacementInfo(
            old_memory="old msg",
            new_memory="new msg",
            confidence=0.9,
            reason="updated",
        )
        with pytest.raises(StorageError, match="Failed to record replacement"):
            phase._record_replacement_transition(info, "new msg")

    async def test_records_all_transitions_before_any_delete(
        self, mock_semantic_engine, mock_tantivy_engine, mock_config, mock_logger
    ):
        """Every intended old is recorded before the first index delete."""
        order: list[str] = []
        mock_semantic_engine.get_id_by_content.side_effect = [11, 12, 99, 99]
        mock_semantic_engine.index = {99}
        mock_tantivy_engine.find_by_exact_match.side_effect = (
            lambda _pid, content: [content] if content == "new msg" else []
        )

        def record(*_args: object, **_kwargs: object) -> ReplacementTransition:
            order.append("record")
            return ReplacementTransition(
                id=len(order),
                workspace_id="test_project",
                old_memory_id=10 + len(order),
                old_content="old",
                new_content="new msg",
                archive_id=100,
                reason="updated",
                confidence=0.9,
                status="pending",
            )

        def delete(*, memory_id: str) -> None:
            order.append(f"delete:{memory_id}")

        mock_semantic_engine.memory_store.begin_replacement_transition.side_effect = (
            record
        )
        mock_semantic_engine.delete.side_effect = delete

        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        infos = [
            ReplacementInfo(old_memory="old-a", new_memory="new msg", confidence=0.9, reason="a"),
            ReplacementInfo(old_memory="old-b", new_memory="new msg", confidence=0.9, reason="b"),
        ]
        result = await phase.execute(
            ["new msg"], replacement_map={"new msg": infos}
        )
        assert result.replaced_count == 2
        assert order[:2] == ["record", "record"]
        assert order[2].startswith("delete:")

    async def test_archive_failure_prevents_delete(
        self, mock_semantic_engine, mock_tantivy_engine, mock_config, mock_logger
    ):
        """An archive/transition failure must not delete the old memory."""
        replacement_info = ReplacementInfo(
            old_memory="old msg",
            new_memory="new msg",
            confidence=0.9,
            reason="updated",
        )
        mock_semantic_engine.get_id_by_content.return_value = 42
        mock_semantic_engine.memory_store.begin_replacement_transition.side_effect = (
            StorageError("archive failed")
        )

        phase = StoragePhase(
            semantic_engine=mock_semantic_engine,
            tantivy_engine=mock_tantivy_engine,
            config=mock_config,
            logger=mock_logger,
        )
        with pytest.raises(StorageError, match="Failed to record replacement"):
            await phase.execute(
                ["new msg"],
                replacement_map={"new msg": [replacement_info]},
            )

        mock_semantic_engine.delete.assert_not_called()
        mock_tantivy_engine.delete.assert_not_called()


# ---------------------------------------------------------------------------
# AddPipeline Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAddPipeline:
    """Tests for the AddPipeline orchestrator."""

    async def test_execute_empty_memories(self, mock_config, mock_logger):
        """Empty memories list returns default AddResult."""
        phase1 = MagicMock()
        phase2 = MagicMock()
        phase2._get_smart_replacer.return_value = None
        phase3 = MagicMock()

        pipeline = AddPipeline(
            duplicate_detection_phase=phase1,
            smart_replacement_phase=phase2,
            storage_phase=phase3,
            config=mock_config,
            logger=mock_logger,
        )
        result = await pipeline.execute([])

        assert result.stored_count == 0
        assert result.skipped_count == 0
        phase1.execute.assert_not_called()

    async def test_execute_full_pipeline(self, mock_config, mock_logger):
        """Full pipeline: Phase 1 → Phase 2 → Phase 3."""
        phase1 = MagicMock()
        phase1.execute = AsyncMock(
            return_value=Phase1Result(
                unique_memories=["msg1"],
                storage_duplicates=["dup"],
                batch_duplicates_count=1,
                duration=0.1,
            )
        )

        phase2 = MagicMock()
        phase2._get_smart_replacer.return_value = None
        phase2.execute = AsyncMock(
            return_value=Phase2Result(
                replacement_map={},
                total_replacements=0,
                duration=0.1,
            )
        )

        phase3 = MagicMock()
        phase3.execute = AsyncMock(
            return_value=Phase3Result(
                stored_count=1,
                replaced_count=0,
                replacements=[],
                duration=0.1,
            )
        )

        pipeline = AddPipeline(
            duplicate_detection_phase=phase1,
            smart_replacement_phase=phase2,
            storage_phase=phase3,
            config=mock_config,
            logger=mock_logger,
        )
        result = await pipeline.execute(["msg1", "dup", "msg1"])

        assert result.stored_count == 1
        # skipped = storage_duplicates(1) + batch_duplicates(1) = 2
        assert result.skipped_count == 2

    async def test_execute_pipeline_error_live_mode(self, mock_config, mock_logger):
        """Pipeline error in live mode raises StorageError (lines 988-1003)."""
        phase1 = MagicMock()
        phase1.execute = MagicMock(side_effect=RuntimeError("Phase 1 failed"))

        phase2 = MagicMock()
        phase2._get_smart_replacer.return_value = None
        phase3 = MagicMock()

        pipeline = AddPipeline(
            duplicate_detection_phase=phase1,
            smart_replacement_phase=phase2,
            storage_phase=phase3,
            config=mock_config,
            logger=mock_logger,
        )

        with pytest.raises(StorageError, match="Failed to add memories"):
            await pipeline.execute(["msg1"], dry_run=False)

        mock_logger.error.assert_called()

    async def test_execute_pipeline_error_dry_run_mode(self, mock_config, mock_logger):
        """Pipeline error in dry_run mode does not raise (lines 1002-1003)."""
        phase1 = MagicMock()
        phase1.execute = MagicMock(side_effect=RuntimeError("Phase 1 failed"))

        phase2 = MagicMock()
        phase2._get_smart_replacer.return_value = None
        phase3 = MagicMock()

        pipeline = AddPipeline(
            duplicate_detection_phase=phase1,
            smart_replacement_phase=phase2,
            storage_phase=phase3,
            config=mock_config,
            logger=mock_logger,
        )
        # Should NOT raise in dry_run mode
        result = await pipeline.execute(["msg1"], dry_run=True)

        assert result.stored_count == 0
        mock_logger.error.assert_called()

    async def test_execute_dry_run_full_pipeline(self, mock_config, mock_logger):
        """Full pipeline in dry_run mode."""
        phase1 = MagicMock()
        phase1.execute = AsyncMock(
            return_value=Phase1Result(
                unique_memories=["msg1"],
                storage_duplicates=[],
                batch_duplicates_count=0,
                duration=0.1,
            )
        )

        phase2 = MagicMock()
        phase2._get_smart_replacer.return_value = MagicMock()
        phase2.execute = AsyncMock(
            return_value=Phase2Result(
                replacement_map={},
                total_replacements=0,
                duration=0.1,
            )
        )

        phase3 = MagicMock()
        phase3.execute = AsyncMock(
            return_value=Phase3Result(
                stored_count=1,
                replaced_count=0,
                replacements=[],
                duration=0.1,
            )
        )

        pipeline = AddPipeline(
            duplicate_detection_phase=phase1,
            smart_replacement_phase=phase2,
            storage_phase=phase3,
            config=mock_config,
            logger=mock_logger,
        )
        result = await pipeline.execute(["msg1"], dry_run=True)

        assert result.stored_count == 1
        phase3.execute.assert_called_once()

    async def test_execute_with_replacements(self, mock_config, mock_logger):
        """Pipeline with replacements propagates results."""
        replacement = ReplacementInfo(
            old_memory="old",
            new_memory="new",
            confidence=0.9,
            reason="updated",
            similarity_score=0.85,
        )

        phase1 = MagicMock()
        phase1.execute = AsyncMock(
            return_value=Phase1Result(
                unique_memories=["new"],
                storage_duplicates=[],
                batch_duplicates_count=0,
                duration=0.1,
            )
        )

        phase2 = MagicMock()
        phase2._get_smart_replacer.return_value = MagicMock()
        phase2.execute = AsyncMock(
            return_value=Phase2Result(
                replacement_map={"new": [replacement]},
                total_replacements=1,
                duration=0.1,
            )
        )

        phase3 = MagicMock()
        phase3.execute = AsyncMock(
            return_value=Phase3Result(
                stored_count=1,
                replaced_count=1,
                replacements=[replacement],
                duration=0.1,
            )
        )

        pipeline = AddPipeline(
            duplicate_detection_phase=phase1,
            smart_replacement_phase=phase2,
            storage_phase=phase3,
            config=mock_config,
            logger=mock_logger,
        )
        result = await pipeline.execute(["new"])

        assert result.stored_count == 1
        assert result.replaced_count == 1
        assert len(result.replacements) == 1
        assert result.replacements[0].old_memory == "old"
