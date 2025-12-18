"""Unit tests for TantivyEngine."""

import os
import tempfile
from unittest.mock import MagicMock

import pytest

from ccmemories.infrastructure.tantivy_engine import TantivyConfig, TantivyEngine


@pytest.mark.unit
class TestTantivyConfig:
    """Test TantivyConfig dataclass."""

    def test_config_creation(self) -> None:
        """Test creating TantivyConfig with required fields."""
        config = TantivyConfig(project_id="test-project", index_path="/tmp/test-index")
        assert config.project_id == "test-project"
        assert config.index_path == "/tmp/test-index"

    def test_config_immutability(self) -> None:
        """Test that TantivyConfig is frozen/immutable."""
        config = TantivyConfig(project_id="test", index_path="/tmp/test")
        with pytest.raises(AttributeError):
            config.project_id = "new-project"  # type: ignore[misc]


@pytest.mark.unit
class TestTantivyEngineInitialization:
    """Test TantivyEngine initialization."""

    def test_init_with_config_object(self) -> None:
        """Test initialization with TantivyConfig object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(project_id="test-project", index_path=tmpdir)
            engine = TantivyEngine(config)
            assert engine.config.project_id == "test-project"
            assert engine.name == "tantivy"

    def test_init_with_dict_config(self) -> None:
        """Test initialization with dict config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dict = {"project_id": "test-project", "index_path": tmpdir}
            engine = TantivyEngine(config_dict)
            assert engine.config.project_id == "test-project"

    def test_init_creates_index_directory(self) -> None:
        """Test that initialization creates the index directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "new_index")
            config = TantivyConfig(project_id="test", index_path=index_path)
            TantivyEngine(config)
            assert os.path.isdir(index_path)

    def test_init_with_logger(self) -> None:
        """Test initialization with logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(project_id="test", index_path=tmpdir)
            engine = TantivyEngine(config, logger=mock_logger)
            assert engine.logger is mock_logger
            # Logger should have been called during initialization
            assert mock_logger.info.called


@pytest.mark.unit
class TestTantivyEngineOperations:
    """Test TantivyEngine CRUD operations."""

    @pytest.fixture
    def engine(self) -> TantivyEngine:
        """Create a TantivyEngine with temporary directory."""
        tmpdir = tempfile.mkdtemp()
        config = TantivyConfig(project_id="test-project", index_path=tmpdir)
        return TantivyEngine(config)

    def test_add_document(self, engine: TantivyEngine) -> None:
        """Test adding a document to the index."""
        # Should not raise
        engine.add("user-1", "Test message content")

    def test_commit(self, engine: TantivyEngine) -> None:
        """Test committing changes."""
        engine.add("user-1", "Test message")
        # Should not raise
        engine.commit()

    def test_search_empty_index(self, engine: TantivyEngine) -> None:
        """Test searching an empty index."""
        results = engine.search("test query", "user-1", limit=5)
        assert results == []

    def test_search_after_add_and_commit(self, engine: TantivyEngine) -> None:
        """Test searching after adding documents."""
        engine.add("user-1", "Python programming language")
        engine.add("user-1", "JavaScript is for web development")
        engine.commit()

        results = engine.search("Python", "user-1", limit=5)
        assert len(results) >= 1
        # First result should contain "Python"
        assert "Python" in results[0][0]

    def test_search_filters_by_project_id(self, engine: TantivyEngine) -> None:
        """Test that search filters by project_id."""
        engine.add("user-1", "Message for user 1")
        engine.add("user-2", "Message for user 2")
        engine.commit()

        # Search for user-1 should only return user-1's messages
        results = engine.search("Message", "user-1", limit=10)
        for msg, score in results:
            assert "user 1" in msg or results == []

    def test_search_respects_limit(self, engine: TantivyEngine) -> None:
        """Test that search respects the limit parameter."""
        for i in range(10):
            engine.add("user-1", f"Document number {i}")
        engine.commit()

        results = engine.search("Document", "user-1", limit=3)
        assert len(results) <= 3


@pytest.mark.unit
class TestTantivyEngineLazyInitialization:
    """Test lazy initialization of writer and searcher."""

    def test_writer_lazy_init(self) -> None:
        """Test that writer is lazily initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(project_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            # Writer should not be initialized yet
            assert engine._writer is None

            # Access writer property
            writer = engine.writer
            assert writer is not None
            assert engine._writer is not None

    def test_searcher_lazy_init(self) -> None:
        """Test that searcher is lazily initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(project_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            # Searcher should not be initialized yet
            assert engine._searcher is None

            # Access searcher property
            searcher = engine.searcher
            assert searcher is not None
            assert engine._searcher is not None


@pytest.mark.unit
class TestTantivyEngineErrorHandling:
    """Test TantivyEngine error handling."""

    def test_search_handles_errors_gracefully(self) -> None:
        """Test that search errors are handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(project_id="test", index_path=tmpdir)
            engine = TantivyEngine(config, logger=mock_logger)

            # Force an error by making index None
            engine._index = None

            results = engine.search("test", "user-1", limit=5)
            assert results == []


@pytest.mark.unit
class TestTantivyEnginePersistence:
    """Test TantivyEngine index persistence."""

    def test_index_persists_across_instances(self) -> None:
        """Test that data persists when creating new engine instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(project_id="test", index_path=tmpdir)

            # First instance: add data
            engine1 = TantivyEngine(config)
            engine1.add("user-1", "Persistent test message")
            engine1.commit()

            # Second instance: should see the data
            engine2 = TantivyEngine(config)
            results = engine2.search("Persistent", "user-1", limit=5)

            assert len(results) >= 1
            assert "Persistent" in results[0][0]


@pytest.mark.unit
class TestTantivyEngineDelete:
    """Test TantivyEngine delete operations."""

    @pytest.fixture
    def engine(self) -> TantivyEngine:
        """Create a TantivyEngine with temporary directory."""
        tmpdir = tempfile.mkdtemp()
        config = TantivyConfig(project_id="test-project", index_path=tmpdir)
        return TantivyEngine(config)

    def test_delete_existing_document(self, engine: TantivyEngine) -> None:
        """Test deleting an existing document."""
        # Add and commit a document
        engine.add("test-project", "Hello world test message")
        engine.commit()

        # Verify it exists
        results = engine.search("Hello", "test-project", limit=10)
        assert len(results) == 1

        # Delete it
        result = engine.delete("test-project", "Hello world test message")
        assert result is True

        # Verify it's gone
        results = engine.search("Hello", "test-project", limit=10)
        assert len(results) == 0

    def test_delete_nonexistent_document(self, engine: TantivyEngine) -> None:
        """Test deleting a document that doesn't exist."""
        engine.add("test-project", "Existing message")
        engine.commit()

        result = engine.delete("test-project", "Nonexistent message")
        assert result is False

        # Original document should still exist
        results = engine.search("Existing", "test-project", limit=10)
        assert len(results) == 1

    def test_delete_after_multiple_commits(self, engine: TantivyEngine) -> None:
        """Test delete works after multiple commits (the original bug)."""
        # Add and commit multiple times
        engine.add("test-project", "Message 1")
        engine.commit()

        engine.add("test-project", "Message 2")
        engine.commit()

        engine.add("test-project", "Message 3")
        engine.commit()

        # Delete should work (this was the original bug)
        result = engine.delete("test-project", "Message 2")
        assert result is True

        # Verify Message 2 is gone but others remain
        results = engine.search("Message", "test-project", limit=10)
        messages = [r[0] for r in results]
        assert "Message 2" not in messages
        assert any("Message 1" in m for m in messages)
        assert any("Message 3" in m for m in messages)

    def test_delete_preserves_other_documents(self, engine: TantivyEngine) -> None:
        """Test that delete only removes the target document."""
        engine.add("test-project", "Keep this document")
        engine.add("test-project", "Delete this document")
        engine.add("test-project", "Also keep this one")
        engine.commit()

        result = engine.delete("test-project", "Delete this document")
        assert result is True

        # Check remaining documents using _get_all_docs (not search)
        # since search might not match all docs depending on query
        messages = engine._get_all_docs("test-project")

        assert len(messages) == 2
        assert any("Keep this" in m for m in messages)
        assert any("Also keep" in m for m in messages)
        assert not any("Delete this" in m for m in messages)

    def test_delete_removes_all_duplicates(self, engine: TantivyEngine) -> None:
        """Test that delete removes all occurrences of the same message."""
        # Add duplicate messages
        engine.add("test-project", "Duplicate message")
        engine.add("test-project", "Duplicate message")
        engine.add("test-project", "Unique message")
        engine.commit()

        result = engine.delete("test-project", "Duplicate message")
        assert result is True

        # All duplicates should be gone
        results = engine.search("Duplicate", "test-project", limit=10)
        assert len(results) == 0

        # Unique message should remain
        results = engine.search("Unique", "test-project", limit=10)
        assert len(results) == 1

    def test_delete_respects_project_id(self, engine: TantivyEngine) -> None:
        """Test that delete only affects the specified project."""
        engine.add("project-a", "Same message")
        engine.add("project-b", "Same message")
        engine.commit()

        # Delete from project-a only
        result = engine.delete("project-a", "Same message")
        assert result is True

        # project-b's message should still exist
        results = engine.search("Same", "project-b", limit=10)
        assert len(results) == 1

    def test_delete_with_empty_index(self, engine: TantivyEngine) -> None:
        """Test delete on empty index returns False."""
        result = engine.delete("test-project", "Any message")
        assert result is False

    def test_delete_with_logger(self) -> None:
        """Test that delete logs appropriately."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(project_id="test", index_path=tmpdir)
            engine = TantivyEngine(config, logger=mock_logger)

            engine.add("test", "Test message")
            engine.commit()

            engine.delete("test", "Test message")

            # Should have logged the delete operation
            assert mock_logger.debug.called


@pytest.mark.unit
class TestTantivyEngineHelperMethods:
    """Test TantivyEngine helper methods."""

    @pytest.fixture
    def engine(self) -> TantivyEngine:
        """Create a TantivyEngine with temporary directory."""
        tmpdir = tempfile.mkdtemp()
        config = TantivyConfig(project_id="test-project", index_path=tmpdir)
        return TantivyEngine(config)

    def test_get_all_docs_empty_index(self, engine: TantivyEngine) -> None:
        """Test _get_all_docs on empty index."""
        docs = engine._get_all_docs("test-project")
        assert docs == []

    def test_get_all_docs_with_documents(self, engine: TantivyEngine) -> None:
        """Test _get_all_docs returns all documents for project."""
        engine.add("test-project", "Doc 1")
        engine.add("test-project", "Doc 2")
        engine.add("test-project", "Doc 3")
        engine.commit()

        docs = engine._get_all_docs("test-project")
        assert len(docs) == 3
        assert "Doc 1" in docs
        assert "Doc 2" in docs
        assert "Doc 3" in docs

    def test_get_all_docs_filters_by_project(self, engine: TantivyEngine) -> None:
        """Test _get_all_docs only returns docs for specified project."""
        engine.add("project-a", "A's doc")
        engine.add("project-b", "B's doc")
        engine.commit()

        docs_a = engine._get_all_docs("project-a")
        assert len(docs_a) == 1
        assert "A's doc" in docs_a

        docs_b = engine._get_all_docs("project-b")
        assert len(docs_b) == 1
        assert "B's doc" in docs_b

    def test_recreate_writer_if_needed(self, engine: TantivyEngine) -> None:
        """Test _recreate_writer_if_needed creates a new writer."""
        # Start with no writer
        assert engine._writer is None

        # Recreate should create a new writer
        with engine._writer_lock:
            engine._recreate_writer_if_needed()

        # Should have a writer now
        assert engine._writer is not None

        # After commit, writer should still be valid (optimization: reuse writer)
        engine.add("test", "test message")
        engine.commit()
        assert engine._writer is not None  # Writer reused, not invalidated

        # After flush, writer should be None (flush invalidates writer)
        engine.flush()
        assert engine._writer is None

        # Recreate again should work
        with engine._writer_lock:
            engine._recreate_writer_if_needed()
        assert engine._writer is not None


@pytest.mark.unit
class TestTantivyEngineWriterReuse:
    """Tests for writer reuse optimization."""

    def test_writer_reused_after_commit(self) -> None:
        """Test that writer is reused after commit (not recreated)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(project_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            # First add - creates writer
            engine.add("test", "Message 1")
            writer_after_add = engine._writer
            assert writer_after_add is not None

            # Commit - writer should stay valid
            engine.commit()
            writer_after_commit = engine._writer
            assert writer_after_commit is writer_after_add  # Same object!

            # Second add - should reuse same writer
            engine.add("test", "Message 2")
            writer_after_second_add = engine._writer
            assert writer_after_second_add is writer_after_add  # Still same!

            engine.commit()
            engine.close()

    def test_multiple_commit_cycles_reuse_writer(self) -> None:
        """Test writer reused across multiple add-commit cycles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(project_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            engine.add("test", "Message 1")
            first_writer = engine._writer

            for i in range(5):
                engine.commit()
                engine.add("test", f"Message {i + 2}")
                assert engine._writer is first_writer  # Always same writer

            engine.close()

    def test_flush_invalidates_writer(self) -> None:
        """Test that flush() invalidates the writer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(project_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            engine.add("test", "Message 1")
            assert engine._writer is not None

            engine.flush()
            assert engine._writer is None  # Writer invalidated

            # Next add creates new writer
            engine.add("test", "Message 2")
            assert engine._writer is not None

            engine.close()

    def test_flush_commits_and_refreshes_searcher(self) -> None:
        """Test that flush() commits data and refreshes searcher."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(project_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            engine.add("test", "Flushable message")
            engine.flush()

            # Data should be searchable after flush
            results = engine.search("Flushable", "test", limit=5)
            assert len(results) == 1
            assert "Flushable" in results[0][0]

            engine.close()

    def test_close_waits_for_merging_threads(self) -> None:
        """Test that close() properly waits for merging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(project_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            # Add many documents to trigger potential merging
            for i in range(100):
                engine.add("test", f"Document {i}")
            engine.commit()

            # Close should complete without error
            engine.close()
            assert engine._writer is None
            assert engine._searcher is None

    def test_data_integrity_across_commit_cycles(self) -> None:
        """Test that data is correctly persisted across multiple commit cycles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(project_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            # Add and commit in multiple cycles
            engine.add("test", "First batch message")
            engine.commit()

            engine.add("test", "Second batch message")
            engine.commit()

            engine.add("test", "Third batch message")
            engine.commit()

            # All messages should be searchable
            results = engine.search("batch", "test", limit=10)
            assert len(results) == 3

            # Verify specific messages
            messages = [r[0] for r in results]
            assert any("First" in m for m in messages)
            assert any("Second" in m for m in messages)
            assert any("Third" in m for m in messages)

            engine.close()

    def test_flush_then_add_creates_new_writer(self) -> None:
        """Test that adding after flush creates a new writer instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(project_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            engine.add("test", "Message 1")
            first_writer = engine._writer
            assert first_writer is not None

            engine.flush()
            assert engine._writer is None

            # New add should create new writer
            engine.add("test", "Message 2")
            second_writer = engine._writer
            assert second_writer is not None
            assert second_writer is not first_writer  # Different instance

            engine.close()

    def test_commit_with_logger(self) -> None:
        """Test that commit logs 'writer reusable' message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(project_id="test", index_path=tmpdir)
            engine = TantivyEngine(config, logger=mock_logger)

            engine.add("test", "Test message")
            engine.commit()

            # Should log with "writer reusable" message
            mock_logger.debug.assert_called()
            call_args = mock_logger.debug.call_args_list
            assert any("reusable" in str(call) for call in call_args)

            engine.close()

    def test_flush_with_logger(self) -> None:
        """Test that flush logs 'writer invalidated' message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(project_id="test", index_path=tmpdir)
            engine = TantivyEngine(config, logger=mock_logger)

            engine.add("test", "Test message")
            engine.flush()

            # Should log with "writer invalidated" message
            mock_logger.debug.assert_called()
            call_args = mock_logger.debug.call_args_list
            assert any("invalidated" in str(call) for call in call_args)

            engine.close()

    def test_flush_on_empty_writer(self) -> None:
        """Test that flush() handles no-writer case gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(project_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            # Writer not initialized yet
            assert engine._writer is None

            # Flush should not raise
            engine.flush()

            # Still no writer
            assert engine._writer is None

            engine.close()


@pytest.mark.unit
class TestTantivySoftDelete:
    """Tests for soft-delete (tombstone) functionality."""

    def test_soft_delete_creates_tombstone(self) -> None:
        """Test that soft_delete adds a tombstone document."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                project_id="test",
                index_path=tmpdir,
                soft_delete_enabled=True,
            )
            engine = TantivyEngine(config)

            # Add a document
            engine.add("test", "Message to delete")
            engine.commit()

            # Verify document exists
            assert engine.find_by_exact_match("test", "Message to delete")

            # Soft delete
            result = engine.soft_delete("test", "Message to delete")
            engine.commit()

            assert result is True

            # Document should now be filtered out from search
            results = engine.search("delete", "test", limit=10)
            assert len(results) == 0

            engine.close()

    def test_soft_delete_nonexistent_message_returns_false(self) -> None:
        """Test that soft_delete returns False for non-existent message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                project_id="test",
                index_path=tmpdir,
                soft_delete_enabled=True,
            )
            engine = TantivyEngine(config)

            result = engine.soft_delete("test", "Non-existent message")
            assert result is False

            engine.close()

    def test_soft_delete_with_disabled_config_still_works(self) -> None:
        """Test that soft_delete works even when soft_delete_enabled=False in config.

        Note: soft_delete_enabled config controls whether delete() uses soft_delete.
        The soft_delete() method itself only checks schema version support.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                project_id="test",
                index_path=tmpdir,
                soft_delete_enabled=False,  # This affects delete(), not soft_delete()
            )
            engine = TantivyEngine(config)

            engine.add("test", "Test message")
            engine.commit()

            # soft_delete() still works (only checks schema version)
            result = engine.soft_delete("test", "Test message")
            engine.commit()  # Must commit tombstone for search to see it
            assert result is True  # V2 schema supports soft-delete

            # Message should be filtered from search
            results = engine.search("Test", "test", limit=10)
            assert len(results) == 0

            engine.close()

    def test_search_filters_tombstoned_messages(self) -> None:
        """Test that search excludes tombstoned messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                project_id="test",
                index_path=tmpdir,
                soft_delete_enabled=True,
            )
            engine = TantivyEngine(config)

            # Add multiple messages
            engine.add("test", "Keep this message")
            engine.add("test", "Delete this message")
            engine.add("test", "Also keep this")
            engine.commit()

            # Verify all 3 are searchable
            results = engine.search("message", "test", limit=10)
            assert len(results) == 2  # "Keep this message" and "Delete this message"

            # Soft delete one
            engine.soft_delete("test", "Delete this message")
            engine.commit()

            # Search should only return non-deleted
            results = engine.search("message", "test", limit=10)
            messages = [r[0] for r in results]
            assert len(messages) == 1
            assert "Keep this message" in messages
            assert "Delete this message" not in messages

            engine.close()

    def test_get_all_docs_filters_tombstoned(self) -> None:
        """Test that _get_all_docs excludes tombstoned messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                project_id="test",
                index_path=tmpdir,
                soft_delete_enabled=True,
            )
            engine = TantivyEngine(config)

            engine.add("test", "Message A")
            engine.add("test", "Message B")
            engine.add("test", "Message C")
            engine.commit()

            # Soft delete Message B
            engine.soft_delete("test", "Message B")
            engine.commit()

            # _get_all_docs should exclude tombstoned
            docs = engine._get_all_docs("test")
            assert len(docs) == 2
            assert "Message A" in docs
            assert "Message C" in docs
            assert "Message B" not in docs

            engine.close()

    def test_delete_uses_soft_delete_when_enabled(self) -> None:
        """Test that delete() uses soft-delete when enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                project_id="test",
                index_path=tmpdir,
                soft_delete_enabled=True,
            )
            engine = TantivyEngine(config)

            engine.add("test", "Test message")
            engine.commit()

            # delete() should use soft-delete internally
            result = engine.delete("test", "Test message")
            assert result is True

            # Message should be filtered from search
            results = engine.search("Test", "test", limit=10)
            assert len(results) == 0

            engine.close()


@pytest.mark.unit
class TestTantivyCompaction:
    """Tests for compaction service functionality."""

    def test_get_tombstone_stats_empty_index(self) -> None:
        """Test tombstone stats on empty index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                project_id="test",
                index_path=tmpdir,
                soft_delete_enabled=True,
            )
            engine = TantivyEngine(config)

            stats = engine.get_tombstone_stats()
            assert stats["total_docs"] == 0
            assert stats["active_docs"] == 0
            assert stats["tombstones"] == 0

            engine.close()

    def test_get_tombstone_stats_with_tombstones(self) -> None:
        """Test tombstone stats after soft-deletes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                project_id="test",
                index_path=tmpdir,
                soft_delete_enabled=True,
            )
            engine = TantivyEngine(config)

            # Add 3 messages
            engine.add("test", "Message 1")
            engine.add("test", "Message 2")
            engine.add("test", "Message 3")
            engine.commit()

            # Soft delete 1
            engine.soft_delete("test", "Message 2")
            engine.commit()

            stats = engine.get_tombstone_stats()
            # 3 originals + 1 tombstone = 4 total docs
            assert stats["total_docs"] == 4
            assert stats["tombstones"] == 1
            # 2 unique active messages (Message 1, Message 3)
            assert stats["unique_active_messages"] == 2

            engine.close()

    def test_needs_compaction_below_threshold(self) -> None:
        """Test needs_compaction returns False when below thresholds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                project_id="test",
                index_path=tmpdir,
                soft_delete_enabled=True,
                compaction_threshold_ratio=0.5,
                compaction_max_tombstones=100,
            )
            engine = TantivyEngine(config)

            # Add messages
            engine.add("test", "Message 1")
            engine.add("test", "Message 2")
            engine.commit()

            # No tombstones - should not need compaction
            assert engine.needs_compaction() is False

            engine.close()

    def test_needs_compaction_ratio_exceeded(self) -> None:
        """Test needs_compaction returns True when ratio exceeded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                project_id="test",
                index_path=tmpdir,
                soft_delete_enabled=True,
                compaction_threshold_ratio=0.2,  # 20% threshold
                compaction_max_tombstones=1000,
            )
            engine = TantivyEngine(config)

            # Add 4 messages
            for i in range(4):
                engine.add("test", f"Message {i}")
            engine.commit()

            # Soft delete 2 (50% tombstone ratio > 20% threshold)
            engine.soft_delete("test", "Message 0")
            engine.soft_delete("test", "Message 1")
            engine.commit()

            assert engine.needs_compaction() is True

            engine.close()

    def test_needs_compaction_max_tombstones_exceeded(self) -> None:
        """Test needs_compaction returns True when max tombstones exceeded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                project_id="test",
                index_path=tmpdir,
                soft_delete_enabled=True,
                compaction_threshold_ratio=0.99,  # Very high ratio threshold
                compaction_max_tombstones=2,  # Low max tombstones
            )
            engine = TantivyEngine(config)

            # Add and delete enough to exceed max_tombstones
            for i in range(5):
                engine.add("test", f"Message {i}")
            engine.commit()

            for i in range(3):
                engine.soft_delete("test", f"Message {i}")
            engine.commit()

            # 3 tombstones > 2 max
            assert engine.needs_compaction() is True

            engine.close()

    def test_compact_removes_tombstones(self) -> None:
        """Test that compact() removes tombstoned messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                project_id="test",
                index_path=tmpdir,
                soft_delete_enabled=True,
                compaction_threshold_ratio=0.1,
            )
            engine = TantivyEngine(config)

            # Add messages
            engine.add("test", "Keep message")
            engine.add("test", "Delete message")
            engine.commit()

            # Soft delete one
            engine.soft_delete("test", "Delete message")
            engine.commit()

            # Verify tombstone exists before compaction
            stats_before = engine.get_tombstone_stats()
            assert stats_before["tombstones"] == 1

            # Force compaction
            result = engine.compact(force=True)

            assert result["compacted"] is True
            assert result["removed_tombstones"] == 1  # Correct key name

            # After compaction, no tombstones
            stats_after = engine.get_tombstone_stats()
            assert stats_after["tombstones"] == 0
            assert stats_after["unique_active_messages"] == 1

            # Kept message still searchable
            results = engine.search("Keep", "test", limit=10)
            assert len(results) == 1
            assert results[0][0] == "Keep message"

            engine.close()

    def test_compact_not_needed_returns_early(self) -> None:
        """Test that compact() returns early when not needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                project_id="test",
                index_path=tmpdir,
                soft_delete_enabled=True,
                compaction_threshold_ratio=0.5,
            )
            engine = TantivyEngine(config)

            engine.add("test", "Test message")
            engine.commit()

            # No tombstones - compact not needed
            result = engine.compact(force=False)
            assert result["compacted"] is False
            assert result["removed_tombstones"] == 0
            assert result["removed_originals"] == 0

            engine.close()

    def test_compact_force_ignores_thresholds(self) -> None:
        """Test that compact(force=True) ignores thresholds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                project_id="test",
                index_path=tmpdir,
                soft_delete_enabled=True,
                compaction_threshold_ratio=0.99,  # Very high
                compaction_max_tombstones=10000,  # Very high
            )
            engine = TantivyEngine(config)

            engine.add("test", "Message")
            engine.commit()
            engine.soft_delete("test", "Message")
            engine.commit()

            # With force=True, compaction should proceed
            result = engine.compact(force=True)
            assert result["compacted"] is True

            engine.close()


@pytest.mark.unit
class TestTantivyTombstoneHelpers:
    """Tests for tombstone helper methods."""

    def test_get_tombstoned_messages_empty(self) -> None:
        """Test _get_tombstoned_messages on empty index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                project_id="test",
                index_path=tmpdir,
                soft_delete_enabled=True,
            )
            engine = TantivyEngine(config)

            tombstoned = engine._get_tombstoned_messages("test")
            assert tombstoned == set()

            engine.close()

    def test_get_tombstoned_messages_returns_deleted(self) -> None:
        """Test _get_tombstoned_messages returns soft-deleted messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                project_id="test",
                index_path=tmpdir,
                soft_delete_enabled=True,
            )
            engine = TantivyEngine(config)

            engine.add("test", "Message A")
            engine.add("test", "Message B")
            engine.commit()

            engine.soft_delete("test", "Message A")
            engine.commit()

            tombstoned = engine._get_tombstoned_messages("test")
            assert tombstoned == {"Message A"}

            engine.close()

    def test_multiple_soft_deletes_same_message(self) -> None:
        """Test that multiple soft-deletes of same message work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                project_id="test",
                index_path=tmpdir,
                soft_delete_enabled=True,
            )
            engine = TantivyEngine(config)

            engine.add("test", "Test message")
            engine.commit()

            # Soft delete twice (second should return False)
            first_delete = engine.soft_delete("test", "Test message")
            engine.commit()
            assert first_delete is True

            # Second soft delete - message no longer found in search
            second_delete = engine.soft_delete("test", "Test message")
            assert second_delete is False

            engine.close()

    def test_find_by_exact_match_excludes_tombstoned(self) -> None:
        """Test that find_by_exact_match returns empty list for tombstoned messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                project_id="test",
                index_path=tmpdir,
                soft_delete_enabled=True,
            )
            engine = TantivyEngine(config)

            engine.add("test", "Test message")
            engine.commit()

            # Should find it before delete (returns list of matches)
            matches = engine.find_by_exact_match("test", "Test message")
            assert len(matches) == 1
            assert matches[0] == "Test message"

            # Soft delete
            engine.soft_delete("test", "Test message")
            engine.commit()

            # Should NOT find it after soft delete (empty list)
            matches = engine.find_by_exact_match("test", "Test message")
            assert len(matches) == 0

            engine.close()
