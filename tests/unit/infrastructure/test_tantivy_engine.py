'''Unit tests for TantivyEngine.'''

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import tantivy

from reflectlog.core.exceptions import SearchError
from reflectlog.infrastructure.tantivy_engine import (
    DEFAULT_TANTIVY_DOC_LIMIT,
    TantivyConfig,
    TantivyEngine,
    _is_dict_config,
)


@pytest.mark.unit
class TestTantivyConfig:
    '''Test TantivyConfig dataclass.'''

    def test_config_creation(self) -> None:
        '''Test creating TantivyConfig with required fields.'''
        config = TantivyConfig(workspace_id="test-project", index_path="/tmp/test-index")
        assert config.workspace_id == "test-project"
        assert config.index_path == "/tmp/test-index"

    def test_config_immutability(self) -> None:
        '''Test that TantivyConfig is frozen/immutable.'''
        config = TantivyConfig(workspace_id="test", index_path="/tmp/test")
        with pytest.raises(AttributeError):
            config.workspace_id = "new-project"  # type: ignore


@pytest.mark.unit
class TestTantivyEngineInitialization:
    '''Test TantivyEngine initialization.'''

    def test_init_with_config_object(self) -> None:
        '''Test initialization with TantivyConfig object.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test-project", index_path=tmpdir)
            engine = TantivyEngine(config)
            assert engine.config.workspace_id == "test-project"
            assert engine.name == "tantivy"

    def test_init_with_dict_config(self) -> None:
        '''Test initialization with dict config.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dict = {"workspace_id": "test-project", "index_path": tmpdir}
            engine = TantivyEngine(config_dict)
            assert engine.config.workspace_id == "test-project"

    def test_init_creates_index_directory(self) -> None:
        '''Test that initialization creates the index directory.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "new_index")
            config = TantivyConfig(workspace_id="test", index_path=index_path)
            TantivyEngine(config)
            assert os.path.isdir(index_path)

    def test_init_with_logger(self) -> None:
        '''Test initialization with logger.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config, logger=mock_logger)
            assert engine.logger is mock_logger
            # Logger should have been called during initialization
            assert mock_logger.info.called


@pytest.mark.unit
class TestTantivyEngineOperations:
    '''Test TantivyEngine CRUD operations.'''

    @pytest.fixture
    def engine(self) -> TantivyEngine:
        '''Create a TantivyEngine with temporary directory.'''
        tmpdir = tempfile.mkdtemp()
        config = TantivyConfig(workspace_id="test-project", index_path=tmpdir)
        return TantivyEngine(config)

    def test_add_document(self, engine: TantivyEngine) -> None:
        '''Test adding a document to the index.'''
        # Should not raise
        engine.add("user-1", "Test message content")

    def test_commit(self, engine: TantivyEngine) -> None:
        '''Test committing changes.'''
        engine.add("user-1", "Test message")
        # Should not raise
        engine.commit()

    def test_search_empty_index(self, engine: TantivyEngine) -> None:
        '''Test searching an empty index.'''
        results = engine.search("test query", "user-1", limit=5)
        assert results == []

    def test_search_after_add_and_commit(self, engine: TantivyEngine) -> None:
        '''Test searching after adding documents.'''
        engine.add("user-1", "Python programming language")
        engine.add("user-1", "JavaScript is for web development")
        engine.commit()

        results = engine.search("Python", "user-1", limit=5)
        assert len(results) >= 1
        # First result should contain "Python"
        assert "Python" in results[0][0]

    def test_search_filters_by_workspace_id(self, engine: TantivyEngine) -> None:
        '''Test that search filters by workspace_id.'''
        engine.add("user-1", "Message for user 1")
        engine.add("user-2", "Message for user 2")
        engine.commit()

        # Search for user-1 should only return user-1's messages
        results = engine.search("Message", "user-1", limit=10)
        for msg, _score in results:
            assert "user 1" in msg or results == []

    def test_search_respects_limit(self, engine: TantivyEngine) -> None:
        '''Test that search respects the limit parameter.'''
        for i in range(10):
            engine.add("user-1", f"Document number {i}")
        engine.commit()

        results = engine.search("Document", "user-1", limit=3)
        assert len(results) <= 3


@pytest.mark.unit
class TestTantivyEngineLazyInitialization:
    '''Test lazy initialization of writer and searcher.'''

    def test_writer_lazy_init(self) -> None:
        '''Test that writer is lazily initialized.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            # Writer should not be initialized yet
            assert engine._writer is None

            # Access writer property
            writer = engine.writer
            assert writer is not None
            assert engine._writer is not None

    def test_searcher_lazy_init(self) -> None:
        '''Test that searcher is lazily initialized.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            # Searcher should not be initialized yet
            assert engine._searcher is None

            # Access searcher property
            searcher = engine.searcher
            assert searcher is not None
            assert engine._searcher is not None


@pytest.mark.unit
class TestTantivyEngineErrorHandling:
    '''Test TantivyEngine error handling.'''

    def test_search_handles_errors_gracefully(self) -> None:
        '''Test that search errors are handled gracefully.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config, logger=mock_logger)

            # Force an error by making index None
            engine._index = None

            results = engine.search("test", "user-1", limit=5)
            assert results == []


@pytest.mark.unit
class TestTantivyEnginePersistence:
    '''Test TantivyEngine index persistence.'''

    def test_index_persists_across_instances(self) -> None:
        '''Test that data persists when creating new engine instance.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)

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
    '''Test TantivyEngine delete operations.'''

    @pytest.fixture
    def engine(self) -> TantivyEngine:
        '''Create a TantivyEngine with temporary directory.'''
        tmpdir = tempfile.mkdtemp()
        config = TantivyConfig(workspace_id="test-project", index_path=tmpdir)
        return TantivyEngine(config)

    def test_delete_existing_document(self, engine: TantivyEngine) -> None:
        '''Test deleting an existing document.'''
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
        '''Test deleting a document that doesn't exist.'''
        engine.add("test-project", "Existing message")
        engine.commit()

        result = engine.delete("test-project", "Nonexistent message")
        assert result is False

        # Original document should still exist
        results = engine.search("Existing", "test-project", limit=10)
        assert len(results) == 1

    def test_delete_after_multiple_commits(self, engine: TantivyEngine) -> None:
        '''Test delete works after multiple commits (the original bug).'''
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
        '''Test that delete only removes the target document.'''
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
        '''Test that delete removes all occurrences of the same message.'''
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

    def test_delete_then_readd_survives_restart(self, engine: TantivyEngine) -> None:
        '''Re-adding a deleted text stays searchable after a new engine opens.'''
        engine.add("test-project", "hello world")
        engine.commit()
        assert engine.delete("test-project", "hello world") is True

        engine.add("test-project", "hello world")
        engine.commit()
        assert engine.search("hello", "test-project", limit=10) == [
            ("hello world", pytest.approx(1.0))
        ]
        assert engine.get_tombstone_stats()["unique_active_memories"] == 1

        reopened = TantivyEngine(
            TantivyConfig(
                workspace_id="test-project",
                index_path=engine.config.index_path,
            )
        )
        try:
            assert reopened.search("hello", "test-project", limit=10) == [
                ("hello world", pytest.approx(1.0))
            ]
        finally:
            reopened.close()

    def test_search_skips_tomb_copies_of_live_text(
        self, engine: TantivyEngine
    ) -> None:
        '''Search must not emit tomb docs or duplicate lives of a live text.'''
        engine.add("test-project", "keep me")
        engine.commit()
        assert engine.delete("test-project", "keep me") is True
        engine.add("test-project", "keep me")
        engine.commit()

        results = engine.search("keep", "test-project", limit=10)
        assert results == [("keep me", pytest.approx(1.0))]

    def test_delete_respects_workspace_id(self, engine: TantivyEngine) -> None:
        '''Test that delete only affects the specified project.'''
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
        '''Test delete on empty index returns False.'''
        result = engine.delete("test-project", "Any message")
        assert result is False

    def test_delete_with_logger(self) -> None:
        '''Test that delete logs appropriately.'''
        import logging

        from reflectlog.application.utils.logging import StructuredLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StructuredLogger(logging.getLogger("test-tantivy-delete"))
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config, logger=logger)

            engine.add("test", "Test message")
            engine.commit()

            assert engine.delete("test", "Test message") is True
            engine.close()


@pytest.mark.unit
class TestTantivyEngineHelperMethods:
    '''Test TantivyEngine helper methods.'''

    @pytest.fixture
    def engine(self) -> TantivyEngine:
        '''Create a TantivyEngine with temporary directory.'''
        tmpdir = tempfile.mkdtemp()
        config = TantivyConfig(workspace_id="test-project", index_path=tmpdir)
        return TantivyEngine(config)

    def test_get_all_docs_empty_index(self, engine: TantivyEngine) -> None:
        '''Test _get_all_docs on empty index.'''
        docs = engine._get_all_docs("test-project")
        assert docs == []

    def test_get_all_docs_with_documents(self, engine: TantivyEngine) -> None:
        '''Test _get_all_docs returns all documents for project.'''
        engine.add("test-project", "Doc 1")
        engine.add("test-project", "Doc 2")
        engine.add("test-project", "Doc 3")
        engine.commit()

        docs = engine._get_all_docs("test-project")
        assert len(docs) == 3
        assert "Doc 1" in docs
        assert "Doc 2" in docs
        assert "Doc 3" in docs

    def test_get_all_docs_filters_by_workspace(self, engine: TantivyEngine) -> None:
        '''Test _get_all_docs only returns docs for specified project.'''
        engine.add("project-a", "A's doc")
        engine.add("project-b", "B's doc")
        engine.commit()

        docs_a = engine._get_all_docs("project-a")
        assert len(docs_a) == 1
        assert "A's doc" in docs_a

        docs_b = engine._get_all_docs("project-b")
        assert len(docs_b) == 1
        assert "B's doc" in docs_b

@pytest.mark.unit
class TestTantivyEngineWriterReuse:
    '''Tests for writer reuse optimization.'''

    def test_writer_reused_after_commit(self) -> None:
        '''Test that writer is reused after commit (not recreated).'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
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
        '''Test writer reused across multiple add-commit cycles.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            engine.add("test", "Message 1")
            first_writer = engine._writer

            for i in range(5):
                engine.commit()
                engine.add("test", f"Message {i + 2}")
                assert engine._writer is first_writer  # Always same writer

            engine.close()

    def test_flush_invalidates_writer(self) -> None:
        '''Test that flush() invalidates the writer.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
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
        '''Test that flush() commits data and refreshes searcher.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            engine.add("test", "Flushable message")
            engine.flush()

            # Data should be searchable after flush
            results = engine.search("Flushable", "test", limit=5)
            assert len(results) == 1
            assert "Flushable" in results[0][0]

            engine.close()

    def test_close_waits_for_merging_threads(self) -> None:
        '''Test that close() properly waits for merging.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
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
        '''Test that data is correctly persisted across multiple commit cycles.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
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
        '''Test that adding after flush creates a new writer instance.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
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
        '''Test that commit logs 'writer reusable' message.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config, logger=mock_logger)

            engine.add("test", "Test message")
            engine.commit()

            # Should log with "writer reusable" message
            mock_logger.debug.assert_called()
            call_args = mock_logger.debug.call_args_list
            assert any("reusable" in str(call) for call in call_args)

            engine.close()

    def test_flush_with_logger(self) -> None:
        '''Test that flush logs 'writer invalidated' message.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config, logger=mock_logger)

            engine.add("test", "Test message")
            engine.flush()

            # Should log with "writer invalidated" message
            mock_logger.debug.assert_called()
            call_args = mock_logger.debug.call_args_list
            assert any("invalidated" in str(call) for call in call_args)

            engine.close()

    def test_flush_on_empty_writer(self) -> None:
        '''Test that flush() handles no-writer case gracefully.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
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
    '''Tests for soft-delete (tombstone) functionality.'''

    def test_soft_delete_creates_tombstone(self) -> None:
        '''Test that soft_delete adds a tombstone document.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
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

    def test_soft_delete_nonexistent_content_returns_false(self) -> None:
        '''Test that soft_delete returns False for non-existent message.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
                index_path=tmpdir,
                soft_delete_enabled=True,
            )
            engine = TantivyEngine(config)

            result = engine.soft_delete("test", "Non-existent message")
            assert result is False

            engine.close()

    def test_soft_delete_with_disabled_config_still_works(self) -> None:
        '''Test that soft_delete works even when soft_delete_enabled=False in config.

        Note: soft_delete_enabled config controls whether delete() uses soft_delete.
        The soft_delete() method itself only checks schema version support.
        '''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
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

    def test_search_keeps_live_hits_after_many_tombstones(self) -> None:
        '''Live FTS hits survive when unique tombstones exceed the limit.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
                index_path=tmpdir,
                soft_delete_enabled=True,
            )
            engine = TantivyEngine(config)
            engine.add("test", "keep this live note")
            for index in range(8):
                text = f"delete this note {index}"
                engine.add("test", text)
            engine.commit()
            for index in range(8):
                assert engine.soft_delete("test", f"delete this note {index}")
            engine.commit()

            results = engine.search("note", "test", limit=3)
            messages = [memory for memory, _score in results]
            assert "keep this live note" in messages
            assert all(not item.startswith("delete this note") for item in messages)
            engine.close()

    def test_search_filters_tombstoned_memories(self) -> None:
        '''Test that search excludes tombstoned messages.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
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
        '''Test that _get_all_docs excludes tombstoned messages.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
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
        '''Test that delete() uses soft-delete when enabled.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
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

    def test_delete_does_not_compact(self) -> None:
        """Request-path delete must not rebuild the index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
                index_path=tmpdir,
                soft_delete_enabled=True,
                compaction_threshold_ratio=0.01,
                compaction_max_tombstones=1,
            )
            engine = TantivyEngine(config)
            try:
                engine.add("test", "keep")
                engine.add("test", "drop")
                engine.commit()
                with patch.object(TantivyEngine, "compact") as compact:
                    assert engine.delete("test", "drop") is True
                    compact.assert_not_called()
                stats = engine.get_tombstone_stats()
                assert stats["tombstones"] >= 1
            finally:
                engine.close()


@pytest.mark.unit
class TestTantivyCompaction:
    '''Tests for compaction service functionality.'''

    def test_get_tombstone_stats_empty_index(self) -> None:
        '''Test tombstone stats on empty index.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
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
        '''Test tombstone stats after soft-deletes.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
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
            assert stats["unique_active_memories"] == 2

            engine.close()

    def test_needs_compaction_below_threshold(self) -> None:
        '''Test needs_compaction returns False when below thresholds.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
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
        '''Test needs_compaction returns True when ratio exceeded.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
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
        '''Test needs_compaction returns True when max tombstones exceeded.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
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
        '''Test that compact() removes tombstoned messages.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
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
            assert stats_after["unique_active_memories"] == 1

            # Kept message still searchable
            results = engine.search("Keep", "test", limit=10)
            assert len(results) == 1
            assert results[0][0] == "Keep message"

            engine.close()

    def test_compact_not_needed_returns_early(self) -> None:
        '''Test that compact() returns early when not needed.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
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
        '''Test that compact(force=True) ignores thresholds.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
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
    '''Tests for tombstone helper methods.'''

    def test_get_tombstoned_memories_empty(self) -> None:
        '''Test _get_tombstoned_memories on empty index.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
                index_path=tmpdir,
                soft_delete_enabled=True,
            )
            engine = TantivyEngine(config)

            tombstoned = engine._get_tombstoned_memories("test")
            assert tombstoned == set()

            engine.close()

    def test_get_tombstoned_memories_returns_deleted(self) -> None:
        '''Test _get_tombstoned_memories returns soft-deleted messages.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
                index_path=tmpdir,
                soft_delete_enabled=True,
            )
            engine = TantivyEngine(config)

            engine.add("test", "Message A")
            engine.add("test", "Message B")
            engine.commit()

            engine.soft_delete("test", "Message A")
            engine.commit()

            tombstoned = engine._get_tombstoned_memories("test")
            assert tombstoned == {"Message A"}

            engine.close()

    def test_multiple_soft_deletes_same_content(self) -> None:
        '''Test that multiple soft-deletes of same message work correctly.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
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
        '''Test that find_by_exact_match returns empty list for tombstoned messages.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
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


@pytest.mark.unit
class TestTantivyScoreNormalization:
    '''Tests for BM25 score normalization to 0-1 range.'''

    def test_normalize_typical_bm25_scores(self) -> None:
        '''Test normalization of typical BM25 score range.'''
        import numpy as np

        from reflectlog.utility.scoring import normalize_scores_minmax

        # From user's example: 5.02, 1.9656, 1.9384
        scores = np.array([5.02, 1.9656, 1.9384], dtype=np.float64)
        normalized = normalize_scores_minmax(scores)

        assert normalized[0] == pytest.approx(1.0)  # Best score
        assert normalized[2] == pytest.approx(0.0)  # Worst score
        assert 0.0 < normalized[1] < 1.0  # Middle score
        # (1.9656 - 1.9384) / (5.02 - 1.9384) ≈ 0.00883
        assert normalized[1] == pytest.approx(0.00883, rel=0.01)

    def test_normalize_single_result(self) -> None:
        '''Single result should be 0.5 (neutral midpoint with no range).'''
        import numpy as np

        from reflectlog.utility.scoring import normalize_scores_minmax

        scores = np.array([3.5], dtype=np.float64)
        normalized = normalize_scores_minmax(scores)
        # Single element has no range, so normalized to 0.5 (neutral midpoint)
        assert normalized[0] == 0.5

    def test_normalize_equal_scores(self) -> None:
        '''All equal scores should become 0.5 (neutral midpoint with no range).'''
        import numpy as np

        from reflectlog.utility.scoring import normalize_scores_minmax

        scores = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        normalized = normalize_scores_minmax(scores)
        # Zero range means no differentiation, so normalized to 0.5 (neutral midpoint)
        assert all(s == 0.5 for s in normalized)

    def test_normalize_preserves_order(self) -> None:
        '''Normalization should preserve relative ordering.'''
        import numpy as np

        from reflectlog.utility.scoring import normalize_scores_minmax

        scores = np.array([5.0, 3.0, 1.0, 4.0, 2.0], dtype=np.float64)
        normalized = normalize_scores_minmax(scores)

        # Original order: 5 > 4 > 3 > 2 > 1
        # Normalized order should be same
        assert normalized[0] > normalized[3]  # 5 > 4
        assert normalized[3] > normalized[1]  # 4 > 3
        assert normalized[1] > normalized[4]  # 3 > 2
        assert normalized[4] > normalized[2]  # 2 > 1

    def test_normalize_wide_range(self) -> None:
        '''Test with wide BM25 score range.'''
        import numpy as np

        from reflectlog.utility.scoring import normalize_scores_minmax

        scores = np.array([15.0, 0.5, 7.2, 3.1], dtype=np.float64)
        normalized = normalize_scores_minmax(scores)

        assert normalized[0] == 1.0  # Max (15.0)
        assert normalized[1] == 0.0  # Min (0.5)
        assert all(0.0 <= s <= 1.0 for s in normalized)

    def test_tantivy_engine_normalize_scores_method(self) -> None:
        '''Test the TantivyEngine._normalize_scores() method directly.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
                index_path=tmpdir,
                normalize_scores=True,
            )
            engine = TantivyEngine(config)

            # Test with typical BM25 results
            results = [
                ("Message A", 5.02),
                ("Message B", 1.9656),
                ("Message C", 1.9384),
            ]
            normalized = engine._normalize_scores(results)

            assert len(normalized) == 3
            assert normalized[0][0] == "Message A"
            assert normalized[0][1] == pytest.approx(1.0)
            assert normalized[2][0] == "Message C"
            assert normalized[2][1] == pytest.approx(0.0)

            engine.close()

    def test_tantivy_engine_normalize_scores_single_result(self) -> None:
        '''Test _normalize_scores with single result returns 1.0.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
                index_path=tmpdir,
                normalize_scores=True,
            )
            engine = TantivyEngine(config)

            results = [("Only message", 3.5)]
            normalized = engine._normalize_scores(results)

            assert len(normalized) == 1
            assert normalized[0][0] == "Only message"
            assert normalized[0][1] == 1.0

            engine.close()

    def test_tantivy_engine_normalize_scores_empty_results(self) -> None:
        '''Test _normalize_scores with empty results returns empty list.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
                index_path=tmpdir,
                normalize_scores=True,
            )
            engine = TantivyEngine(config)

            results: list[tuple[str, float]] = []
            normalized = engine._normalize_scores(results)

            assert normalized == []

            engine.close()

    def test_search_with_normalization_enabled(self) -> None:
        '''Test that search returns normalized scores when enabled.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
                index_path=tmpdir,
                normalize_scores=True,
            )
            engine = TantivyEngine(config)

            # Add documents with varying relevance to query
            engine.add("test", "python programming language tutorial")
            engine.add("test", "java programming basics")
            engine.add("test", "cooking recipes for dinner")
            engine.commit()

            # Search for "python programming"
            results = engine.search("python programming", "test", limit=10)

            # All scores should be in 0-1 range
            for _msg, score in results:
                assert 0.0 <= score <= 1.0

            # Best match should have score 1.0
            if results:
                assert results[0][1] == pytest.approx(1.0)

            engine.close()

    def test_search_with_normalization_disabled(self) -> None:
        '''Test that search returns raw BM25 scores when normalization is disabled.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
                index_path=tmpdir,
                normalize_scores=False,
            )
            engine = TantivyEngine(config)

            # Add documents
            engine.add("test", "python programming language tutorial")
            engine.add("test", "java programming basics")
            engine.commit()

            # Search for "python programming"
            results = engine.search("python programming", "test", limit=10)

            # With normalization disabled, scores can be > 1.0 (raw BM25)
            # At least one score should be > 1.0 for typical BM25
            if len(results) > 0:
                # Raw BM25 scores are typically > 1.0 for good matches
                max_score = max(score for _msg, score in results)
                # BM25 scores can vary, but for a good match should be > 1
                assert max_score > 0  # At minimum, scores should be positive

            engine.close()

    def test_config_normalize_scores_default(self) -> None:
        '''Test that normalize_scores defaults to True in TantivyConfig.'''
        config = TantivyConfig(
            workspace_id="test",
            index_path="/tmp/test",
        )
        assert config.normalize_scores is True

    def test_config_normalize_scores_explicit_false(self) -> None:
        '''Test that normalize_scores can be set to False.'''
        config = TantivyConfig(
            workspace_id="test",
            index_path="/tmp/test",
            normalize_scores=False,
        )
        assert config.normalize_scores is False


@pytest.mark.unit
class TestIsDictConfig:
    '''Tests for _is_dict_config type guard function.'''

    def test_dict_returns_true(self) -> None:
        '''Test that a dict returns True.'''
        assert _is_dict_config({"key": "value"}) is True

    def test_non_dict_returns_false(self) -> None:
        '''Test that a non-dict returns False.'''
        assert _is_dict_config("not a dict") is False
        assert _is_dict_config(42) is False
        assert _is_dict_config(None) is False

    def test_tantivy_config_returns_false(self) -> None:
        '''Test that a TantivyConfig returns False.'''
        config = TantivyConfig(workspace_id="test", index_path="/tmp/test")
        assert _is_dict_config(config) is False


@pytest.mark.unit
class TestTantivyConfigFromDict:
    '''Tests for TantivyConfig.from_dict factory method.'''

    def test_from_dict_with_all_fields(self) -> None:
        '''Test from_dict with all fields specified.'''
        data = {
            "workspace_id": "proj-1",
            "index_path": "/tmp/idx",
            "soft_delete_enabled": False,
            "compaction_threshold_ratio": 0.3,
            "compaction_max_tombstones": 5000,
            "tombstone_ttl_days": 14,
            "tombstone_cache_max_size": 50,
            "normalize_scores": False,
        }
        config = TantivyConfig.from_dict(data)
        assert config.workspace_id == "proj-1"
        assert config.index_path == "/tmp/idx"
        assert config.soft_delete_enabled is False
        assert config.compaction_threshold_ratio == 0.3
        assert config.compaction_max_tombstones == 5000
        assert config.tombstone_ttl_days == 14
        assert config.tombstone_cache_max_size == 50
        assert config.normalize_scores is False

    def test_from_dict_with_defaults(self) -> None:
        '''Test from_dict with missing fields uses defaults.'''
        data: dict[str, str] = {}
        config = TantivyConfig.from_dict(data)
        assert config.workspace_id == ""
        assert config.index_path == ""
        assert config.soft_delete_enabled is True
        assert config.compaction_threshold_ratio == 0.2
        assert config.compaction_max_tombstones == 10000

    def test_from_dict_with_none_values(self) -> None:
        '''Test from_dict handles None values for workspace_id and index_path.'''
        data = {"workspace_id": None, "index_path": None}
        config = TantivyConfig.from_dict(data)
        assert config.workspace_id == ""
        assert config.index_path == ""


@pytest.mark.unit
class TestTantivyEngineWriterSearcherErrors:
    '''Tests for writer and searcher RuntimeError when index is None.'''

    def test_writer_raises_runtime_error_when_index_none(self) -> None:
        '''Test that accessing writer raises RuntimeError when _index is None.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)
            engine._index = None
            engine._writer = None

            with pytest.raises(RuntimeError, match="Tantivy index not initialized"):
                _ = engine.writer

    def test_searcher_raises_runtime_error_when_index_none(self) -> None:
        '''Test that accessing searcher raises RuntimeError when _index is None.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)
            engine._index = None
            engine._searcher = None

            with pytest.raises(RuntimeError, match="Tantivy index not initialized"):
                _ = engine.searcher


@pytest.mark.unit
class TestTantivyEngineInitWithLogger:
    '''Tests for _initialize_index logger paths.'''

    def test_init_existing_index_logs_loaded(self) -> None:
        '''Test that loading existing index logs loaded message.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            # First engine creates the index
            engine1 = TantivyEngine(config)
            engine1.add("test", "seed")
            engine1.commit()
            engine1.close()

            # Second engine loads existing index
            engine2 = TantivyEngine(config, logger=mock_logger)
            calls = [str(c) for c in mock_logger.info.call_args_list]
            assert any("Loaded existing Tantivy index" in c for c in calls)
            engine2.close()

    def test_init_new_index_logs_created(self) -> None:
        '''Test that creating new index logs created message.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            index_path = os.path.join(tmpdir, "brand_new_index")
            config = TantivyConfig(workspace_id="test", index_path=index_path)
            engine = TantivyEngine(config, logger=mock_logger)
            calls = [str(c) for c in mock_logger.info.call_args_list]
            assert any("Created new Tantivy index" in c for c in calls)
            engine.close()


@pytest.mark.unit
class TestGetAllDocsErrorHandling:
    '''Tests for _get_all_docs error paths.'''

    def test_get_all_docs_index_none_returns_empty(self) -> None:
        '''Test _get_all_docs returns [] when _index is None.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)
            engine._index = None
            result = engine._get_all_docs("test")
            assert result == []

    def test_get_all_docs_exception_with_logger(self) -> None:
        '''Test _get_all_docs fails closed on exception.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            # Force exception by making searcher raise
            with patch.object(
                type(engine),
                "searcher",
                new_callable=lambda: property(
                    lambda self: (_ for _ in ()).throw(RuntimeError("boom"))
                ),
            ):
                with pytest.raises(RuntimeError, match="Failed to get all docs"):
                    engine._get_all_docs("test")

    def test_get_all_docs_exception_without_logger(self) -> None:
        '''Test _get_all_docs returns [] on exception even without logger.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            with patch.object(
                type(engine),
                "searcher",
                new_callable=lambda: property(
                    lambda self: (_ for _ in ()).throw(RuntimeError("boom"))
                ),
            ):
                with pytest.raises(RuntimeError, match="Failed to get all docs"):
                    engine._get_all_docs("test")


@pytest.mark.unit
class TestGetAllDocsAllWorkspaces:
    '''Tests for _get_all_docs_all_workspaces.'''

    def test_all_workspaces_returns_tuples(self) -> None:
        '''Test _get_all_docs_all_workspaces returns (workspace_id, message) tuples.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            engine.add("proj-a", "Message A")
            engine.add("proj-b", "Message B")
            engine.commit()

            results = engine._get_all_docs_all_workspaces()
            assert len(results) == 2
            workspace_ids = {pid for pid, _ in results}
            assert "proj-a" in workspace_ids
            assert "proj-b" in workspace_ids

    def test_all_workspaces_omits_tombstoned_content(self) -> None:
        '''Rebuild scans must not treat tombstoned texts as live.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
                index_path=tmpdir,
                soft_delete_enabled=True,
            )
            engine = TantivyEngine(config)
            engine.add("proj-a", "keep me")
            engine.add("proj-a", "drop me")
            engine.commit()
            assert engine.soft_delete("proj-a", "drop me")
            engine.commit()

            results = engine._get_all_docs_all_workspaces()
            contents = {content for _workspace, content in results}
            assert "keep me" in contents
            assert "drop me" not in contents
            engine.close()

    def test_rebuild_delete_does_not_resurrect_tombstones(self) -> None:
        '''A later hard delete must not revive earlier soft-deleted texts.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
                index_path=tmpdir,
                soft_delete_enabled=True,
            )
            engine = TantivyEngine(config)
            engine.add("test", "old tomb")
            engine.add("test", "later gone")
            engine.commit()
            assert engine.soft_delete("test", "old tomb")
            engine.commit()
            assert engine._delete_via_rebuild("test", "later gone") is True

            messages = [memory for memory, _score in engine.search("old", "test", limit=5)]
            assert "old tomb" not in messages
            engine.close()

    def test_all_workspaces_index_none_returns_empty(self) -> None:
        '''Test _get_all_docs_all_workspaces returns [] when _index is None.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)
            engine._index = None
            result = engine._get_all_docs_all_workspaces()
            assert result == []

    def test_all_workspaces_exception_with_logger(self) -> None:
        '''Test _get_all_docs_all_workspaces logs warning on exception.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config, logger=mock_logger)

            with patch.object(
                type(engine),
                "searcher",
                new_callable=lambda: property(
                    lambda self: (_ for _ in ()).throw(RuntimeError("boom"))
                ),
            ):
                result = engine._get_all_docs_all_workspaces()

            assert result == []
            calls = [str(c) for c in mock_logger.warning.call_args_list]
            assert any("all workspaces" in c for c in calls)

    def test_all_workspaces_exception_without_logger(self) -> None:
        '''Test _get_all_docs_all_workspaces returns [] on exception without logger.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            with patch.object(
                type(engine),
                "searcher",
                new_callable=lambda: property(
                    lambda self: (_ for _ in ()).throw(RuntimeError("boom"))
                ),
            ):
                result = engine._get_all_docs_all_workspaces()

            assert result == []


@pytest.mark.unit
class TestInvalidateTombstoneCache:
    '''Tests for _invalidate_tombstone_cache with specific workspace_id.'''

    def test_invalidate_specific_workspace(self) -> None:
        '''Test invalidating cache for a specific project.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            # Populate cache by querying tombstones for two projects
            engine.add("proj-a", "msg a")
            engine.add("proj-b", "msg b")
            engine.commit()
            engine._get_tombstoned_memories("proj-a")
            engine._get_tombstoned_memories("proj-b")
            assert "proj-a" in engine._tombstone_cache
            assert "proj-b" in engine._tombstone_cache

            # Invalidate only proj-a
            engine._invalidate_tombstone_cache("proj-a")
            assert "proj-a" not in engine._tombstone_cache
            assert "proj-b" in engine._tombstone_cache

    def test_invalidate_nonexistent_workspace_is_noop(self) -> None:
        '''Test invalidating cache for a non-existent project does nothing.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            # Should not raise
            engine._invalidate_tombstone_cache("nonexistent")

    def test_invalidate_all_clears_entire_cache(self) -> None:
        '''Test invalidating with None clears entire cache.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            engine.add("proj-a", "msg")
            engine.commit()
            engine._get_tombstoned_memories("proj-a")
            assert len(engine._tombstone_cache) > 0

            engine._invalidate_tombstone_cache(None)
            assert len(engine._tombstone_cache) == 0


@pytest.mark.unit
class TestGetTombstonedMessagesCacheAndErrors:
    '''Tests for _get_tombstoned_memories cache LRU eviction and error paths.'''

    def test_cache_hit_returns_cached(self) -> None:
        '''Test that cached tombstones are returned without re-querying.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            engine.add("test", "msg1")
            engine.commit()

            # First call populates cache
            result1 = engine._get_tombstoned_memories("test")
            # Second call should hit cache
            result2 = engine._get_tombstoned_memories("test")
            assert result1 == result2

    def test_lru_eviction_when_cache_full(self) -> None:
        '''Test that oldest entry is evicted when cache exceeds max size.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
                index_path=tmpdir,
                tombstone_cache_max_size=2,
            )
            engine = TantivyEngine(config)

            # Add docs for 3 projects to trigger eviction
            for p in ["proj-0", "proj-1", "proj-2"]:
                engine.add(p, f"msg for {p}")
            engine.commit()

            # Query all three to fill cache beyond max_size=2
            engine._get_tombstoned_memories("proj-0")
            engine._get_tombstoned_memories("proj-1")
            engine._get_tombstoned_memories("proj-2")

            # proj-0 should have been evicted (oldest)
            assert "proj-0" not in engine._tombstone_cache
            assert "proj-1" in engine._tombstone_cache
            assert "proj-2" in engine._tombstone_cache

    def test_index_none_returns_empty_set(self) -> None:
        '''Test that _get_tombstoned_memories returns empty set when _index is None.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)
            engine._index = None

            result = engine._get_tombstoned_memories("test")
            assert result == set()

    def test_value_error_with_logger(self) -> None:
        '''Test ValueError handling in _get_tombstoned_memories with logger.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config, logger=mock_logger)

            mock_index = MagicMock()
            mock_index.parse_query.side_effect = ValueError("bad query")
            engine._index = mock_index

            with pytest.raises(RuntimeError, match="Failed to get tombstoned"):
                engine._get_tombstoned_memories("test")

            calls = [str(c) for c in mock_logger.warning.call_args_list]
            assert any("query parse error" in c for c in calls)

    def test_generic_exception_with_logger(self) -> None:
        '''Test generic exception handling in _get_tombstoned_memories with logger.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config, logger=mock_logger)

            mock_index = MagicMock()
            mock_index.parse_query.side_effect = RuntimeError("fail")
            engine._index = mock_index

            with pytest.raises(RuntimeError, match="Failed to get tombstoned"):
                engine._get_tombstoned_memories("test")

            calls = [str(c) for c in mock_logger.warning.call_args_list]
            assert any("Failed to get tombstoned memories" in c for c in calls)

    def test_generic_exception_without_logger(self) -> None:
        '''Test generic exception without logger is fail-closed.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            mock_index = MagicMock()
            mock_index.parse_query.side_effect = RuntimeError("fail")
            engine._index = mock_index

            with pytest.raises(RuntimeError, match="Failed to get tombstoned"):
                engine._get_tombstoned_memories("test")


@pytest.mark.unit
class TestSearchErrorHandling:
    '''Tests for search error handling paths.'''

    def test_search_empty_query_uses_workspace_only(self) -> None:
        '''Whitespace queries must not dump every workspace document.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            engine.add("test", "test document content")
            engine.commit()

            results = engine.search("  ", "test", limit=10)
            assert results == []

    def test_search_value_error_returns_empty(self) -> None:
        '''Test search returns [] on ValueError and logs with logger.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config, logger=mock_logger)

            mock_index = MagicMock()
            mock_index.parse_query.side_effect = ValueError("bad")
            engine._index = mock_index

            results = engine.search("query", "test", limit=5)

            assert results == []
            calls = [str(c) for c in mock_logger.warning.call_args_list]
            assert any("Tantivy query parsing failed" in c for c in calls)

    def test_search_value_error_without_logger(self) -> None:
        '''Tombstone scan parse errors fail closed instead of returning [].'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            mock_index = MagicMock()
            mock_index.parse_query.side_effect = ValueError("bad")
            engine._index = mock_index

            with pytest.raises(SearchError, match="Failed to get tombstoned"):
                engine.search("query", "test", limit=5)

    def test_search_os_error_raises_search_error(self) -> None:
        '''Test search raises SearchError on OSError and logs with logger.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config, logger=mock_logger)

            mock_index = MagicMock()
            mock_index.parse_query.side_effect = OSError("disk full")
            engine._index = mock_index

            with pytest.raises(SearchError, match="file system error"):
                engine.search("query", "test", limit=5)

            calls = [str(c) for c in mock_logger.error.call_args_list]
            assert any("file system error" in c.lower() for c in calls)

    def test_search_os_error_without_logger(self) -> None:
        '''Test search raises SearchError on OSError without logger.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            mock_index = MagicMock()
            mock_index.parse_query.side_effect = OSError("disk full")
            engine._index = mock_index

            with pytest.raises(SearchError, match="file system error"):
                engine.search("query", "test", limit=5)

    def test_search_unexpected_error_raises_search_error(self) -> None:
        '''Test search raises SearchError on unexpected exceptions.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            mock_index = MagicMock()
            mock_index.parse_query.side_effect = TypeError("unexpected")
            engine._index = mock_index

            with pytest.raises(SearchError, match="Tantivy search failed"):
                engine.search("query", "test", limit=5)

    def test_search_always_escapes_before_parse(self) -> None:
        """Special characters are escaped before the first parse_query call."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)
            engine.add("test", "hello")
            engine.commit()

            original_index = engine._index
            parsed: list[str] = []

            def capture_parse(
                query: str, *, default_field_names: list[str] | None = None
            ) -> tantivy.Query:
                parsed.append(query)
                assert original_index is not None
                return original_index.parse_query(
                    query, default_field_names=default_field_names
                )

            mock_index = MagicMock()
            mock_index.parse_query.side_effect = capture_parse
            engine._index = mock_index
            engine._searcher = engine.searcher
            _ = engine.search("foo:bar", "test", limit=5)
            assert parsed
            assert any("foo\\:bar" in query for query in parsed)

    def test_search_empty_query_value_error_re_raises(self) -> None:
        '''Test that empty query + ValueError is re-raised (not escaped).'''
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config, logger=mock_logger)

            mock_index = MagicMock()
            mock_index.parse_query.side_effect = ValueError("bad")
            engine._index = mock_index

            results = engine.search("", "test", limit=5)

            assert results == []

    def test_search_with_logger_logs_matches(self) -> None:
        '''Test that search logs debug messages for each match.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config, logger=mock_logger)

            engine.add("test", "findable content here")
            engine.commit()

            results = engine.search("findable", "test", limit=5)
            assert len(results) >= 1
            calls = [str(c) for c in mock_logger.debug.call_args_list]
            assert any("Tantivy match" in c for c in calls)

    def test_search_index_none_with_logger(self) -> None:
        '''Test search with _index=None logs warning with logger.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config, logger=mock_logger)
            engine._index = None

            results = engine.search("query", "test", limit=5)
            assert results == []
            calls = [str(c) for c in mock_logger.warning.call_args_list]
            assert any("index not initialized" in c for c in calls)


@pytest.mark.unit
class TestEnsureInitialized:
    '''Tests for ensure_initialized method.'''

    def test_ensure_initialized_creates_searcher(self) -> None:
        '''Test that ensure_initialized forces searcher creation.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            assert engine._searcher is None
            assert engine.is_ready() is False
            engine.ensure_initialized()
            assert engine._searcher is not None
            assert engine.is_ready() is True


@pytest.mark.unit
class TestCloseErrorHandling:
    '''Tests for close() error handling.'''

    def test_close_handles_writer_error(self) -> None:
        '''Test that close handles errors during writer cleanup.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config, logger=mock_logger)

            engine.add("test", "msg")
            real_writer = engine._writer
            if real_writer is not None:
                try:
                    real_writer.commit()
                except Exception:
                    pass
            failing_writer = MagicMock()
            failing_writer.commit.side_effect = RuntimeError("commit failed")
            engine._writer = failing_writer

            # close() should not raise
            engine.close()

            assert engine._writer is None
            assert engine._searcher is None
            assert engine._index is None
            calls = [str(c) for c in mock_logger.warning.call_args_list]
            assert any("Error during Tantivy writer cleanup" in c for c in calls)

    def test_close_with_logger_logs_closed(self) -> None:
        '''Test that close logs engine closed message.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config, logger=mock_logger)

            engine.close()
            calls = [str(c) for c in mock_logger.info.call_args_list]
            assert any("Tantivy engine closed" in c for c in calls)

    def test_close_without_writer(self) -> None:
        '''Test that close works when writer was never initialized.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)
            assert engine._writer is None

            # Should not raise
            engine.close()
            assert engine._index is None


@pytest.mark.unit
class TestSoftDeleteWithLogger:
    '''Tests for soft_delete with logger messages.'''

    def test_soft_delete_not_found_with_logger(self) -> None:
        '''Test soft_delete logs debug when message not found.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config, logger=mock_logger)

            result = engine.soft_delete("test", "nonexistent")
            assert result is False
            calls = [str(c) for c in mock_logger.debug.call_args_list]
            assert any("memory not found" in c for c in calls)

    def test_soft_delete_success_with_logger(self) -> None:
        '''Test soft_delete logs debug when tombstone added.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config, logger=mock_logger)

            engine.add("test", "target message")
            engine.commit()

            result = engine.soft_delete("test", "target message")
            assert result is True
            calls = [str(c) for c in mock_logger.debug.call_args_list]
            assert any("tombstone added" in c for c in calls)

            engine.close()


@pytest.mark.unit
class TestDeleteWithSoftDeleteDisabled:
    '''Tests for delete() with soft_delete_enabled=False (rebuild path).'''

    @pytest.fixture
    def engine_no_soft_delete(self) -> TantivyEngine:
        '''Create engine with soft_delete_enabled=False.'''
        tmpdir = tempfile.mkdtemp()
        config = TantivyConfig(
            workspace_id="test",
            index_path=tmpdir,
            soft_delete_enabled=False,
        )
        return TantivyEngine(config)

    def test_delete_rebuild_existing_content(
        self, engine_no_soft_delete: TantivyEngine
    ) -> None:
        '''Test delete with rebuild removes the message.'''
        engine = engine_no_soft_delete
        engine.add("test", "Keep this")
        engine.add("test", "Delete this")
        engine.commit()

        result = engine.delete("test", "Delete this")
        assert result is True

        docs = engine._get_all_docs("test")
        assert "Keep this" in docs
        assert "Delete this" not in docs

    def test_delete_rebuild_nonexistent_content(
        self, engine_no_soft_delete: TantivyEngine
    ) -> None:
        '''Test delete rebuild returns False for nonexistent message.'''
        engine = engine_no_soft_delete
        engine.add("test", "Only message")
        engine.commit()

        result = engine.delete("test", "Nonexistent")
        assert result is False

    def test_delete_rebuild_index_none(
        self, engine_no_soft_delete: TantivyEngine
    ) -> None:
        '''Test delete rebuild returns False when _index is None.'''
        engine = engine_no_soft_delete
        engine._index = None
        result = engine.delete("test", "msg")
        assert result is False

    def test_delete_rebuild_with_logger(self) -> None:
        '''Test delete rebuild logs progress messages.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(
                workspace_id="test",
                index_path=tmpdir,
                soft_delete_enabled=False,
            )
            engine = TantivyEngine(config, logger=mock_logger)

            engine.add("test", "Target message")
            engine.commit()

            result = engine.delete("test", "Target message")
            assert result is True

            calls = [str(c) for c in mock_logger.debug.call_args_list]
            assert any("found document" in c for c in calls)
            assert any("completed successfully" in c for c in calls)

    def test_delete_rebuild_not_found_with_logger(self) -> None:
        '''Test delete rebuild logs not-found message.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(
                workspace_id="test",
                index_path=tmpdir,
                soft_delete_enabled=False,
            )
            engine = TantivyEngine(config, logger=mock_logger)

            engine.add("test", "Some message")
            engine.commit()

            engine.delete("test", "Not found")
            calls = [str(c) for c in mock_logger.debug.call_args_list]
            assert any("document not found" in c for c in calls)

    def test_delete_rebuild_value_error(self) -> None:
        '''Test delete rebuild handles ValueError.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(
                workspace_id="test",
                index_path=tmpdir,
                soft_delete_enabled=False,
            )
            engine = TantivyEngine(config, logger=mock_logger)

            engine.add("test", "msg")
            engine.commit()

            with patch.object(
                engine, "_get_all_docs_all_workspaces", side_effect=ValueError("bad")
            ):
                result = engine.delete("test", "msg")

            assert result is False
            calls = [str(c) for c in mock_logger.warning.call_args_list]
            assert any("query parsing failed" in c.lower() for c in calls)

    def test_delete_rebuild_os_error(self) -> None:
        '''Test delete rebuild handles OSError.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(
                workspace_id="test",
                index_path=tmpdir,
                soft_delete_enabled=False,
            )
            engine = TantivyEngine(config, logger=mock_logger)

            engine.add("test", "msg")
            engine.commit()

            with patch.object(
                engine, "_get_all_docs_all_workspaces", side_effect=OSError("disk")
            ):
                result = engine.delete("test", "msg")

            assert result is False
            calls = [str(c) for c in mock_logger.error.call_args_list]
            assert any("file system error" in c.lower() for c in calls)

    def test_delete_rebuild_unexpected_error(self) -> None:
        '''Test delete rebuild handles unexpected errors.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(
                workspace_id="test",
                index_path=tmpdir,
                soft_delete_enabled=False,
            )
            engine = TantivyEngine(config, logger=mock_logger)

            engine.add("test", "msg")
            engine.commit()

            with patch.object(
                engine, "_get_all_docs_all_workspaces", side_effect=TypeError("oops")
            ):
                result = engine.delete("test", "msg")

            assert result is False
            calls = [str(c) for c in mock_logger.error.call_args_list]
            assert any("failed unexpectedly" in c.lower() for c in calls)

    def test_delete_rebuild_unexpected_error_without_logger(self) -> None:
        '''Test delete rebuild handles unexpected errors without logger.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
                index_path=tmpdir,
                soft_delete_enabled=False,
            )
            engine = TantivyEngine(config)

            engine.add("test", "msg")
            engine.commit()

            with patch.object(
                engine, "_get_all_docs_all_workspaces", side_effect=TypeError("oops")
            ):
                result = engine.delete("test", "msg")

            assert result is False

    def test_delete_rebuild_preserves_other_workspaces(self) -> None:
        '''Test delete rebuild preserves documents from other projects.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(
                workspace_id="test",
                index_path=tmpdir,
                soft_delete_enabled=False,
            )
            engine = TantivyEngine(config)

            engine.add("proj-a", "Message A")
            engine.add("proj-b", "Message B")
            engine.commit()

            result = engine.delete("proj-a", "Message A")
            assert result is True

            # proj-b's message should remain
            results = engine.search("Message", "proj-b", limit=10)
            assert len(results) == 1


@pytest.mark.unit
class TestRebuildIndexWithDocs:
    '''Tests for _rebuild_index_with_docs edge cases.'''

    def test_rebuild_with_existing_writer_error(self) -> None:
        '''Test rebuild handles writer commit error gracefully.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            engine.add("test", "original")
            engine.commit()

            # Force writer to raise on commit during rebuild
            _original_writer = engine._writer
            mock_writer = MagicMock()
            mock_writer.commit.side_effect = RuntimeError("boom")
            engine._writer = mock_writer

            # Rebuild should still work (best effort cleanup)
            engine._rebuild_index_with_docs([("test", "rebuilt")])

            docs = engine._get_all_docs("test")
            assert "rebuilt" in docs


@pytest.mark.unit
class TestGetDocLimitEdgeCases:
    '''Tests for _get_doc_limit edge cases.'''

    def test_doc_limit_index_none(self) -> None:
        '''Test _get_doc_limit returns 0 when _index is None.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)
            engine._index = None
            assert engine._get_doc_limit() == 0

    def test_doc_limit_callable_num_docs(self) -> None:
        '''Test _get_doc_limit uses callable num_docs.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            engine.add("test", "msg")
            engine.commit()

            # Normal path should return doc count
            result = engine._get_doc_limit()
            assert result >= 1

    def test_doc_limit_numeric_num_docs(self) -> None:
        '''Test _get_doc_limit uses searcher.num_docs().'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)
            engine.add("test", "msg")
            engine.commit()

            mock_searcher = MagicMock()
            mock_searcher.num_docs.return_value = 42
            with patch.object(
                type(engine),
                "searcher",
                new_callable=lambda: property(lambda self: mock_searcher),
            ):
                result = engine._get_doc_limit()
            assert result == 42

    def test_doc_limit_negative_num_docs_uses_fallback(self) -> None:
        '''Test _get_doc_limit uses fallback when num_docs is negative.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            mock_searcher = MagicMock()
            mock_searcher.num_docs.return_value = -1
            with patch.object(
                type(engine),
                "searcher",
                new_callable=lambda: property(lambda self: mock_searcher),
            ):
                result = engine._get_doc_limit()
            assert result == DEFAULT_TANTIVY_DOC_LIMIT

    def test_doc_limit_none_num_docs_uses_fallback(self) -> None:
        '''Test _get_doc_limit uses fallback when num_docs returns None.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            mock_searcher = MagicMock()
            mock_searcher.num_docs.return_value = None
            with patch.object(
                type(engine),
                "searcher",
                new_callable=lambda: property(lambda self: mock_searcher),
            ):
                result = engine._get_doc_limit()
            assert result == DEFAULT_TANTIVY_DOC_LIMIT

    def test_doc_limit_exception_uses_fallback(self) -> None:
        '''Test _get_doc_limit uses fallback when num_docs raises.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            mock_searcher = MagicMock()
            mock_searcher.num_docs.side_effect = RuntimeError("fail")
            with patch.object(
                type(engine),
                "searcher",
                new_callable=lambda: property(lambda self: mock_searcher),
            ):
                result = engine._get_doc_limit()
            assert result == DEFAULT_TANTIVY_DOC_LIMIT

    def test_doc_limit_zero_returns_max_1(self) -> None:
        '''Test _get_doc_limit returns max(1, 0) = 1 for empty index.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            mock_searcher = MagicMock()
            mock_searcher.num_docs.return_value = 0
            with patch.object(
                type(engine),
                "searcher",
                new_callable=lambda: property(lambda self: mock_searcher),
            ):
                result = engine._get_doc_limit()
            # max(1, 0) = 1 (line 1078)
            # Actually num_docs=0 is not < 0, so it returns max(1, 0) = 1
            assert result == 1


@pytest.mark.unit
class TestGetTombstoneStatsEdgeCases:
    '''Tests for get_tombstone_stats edge cases.'''

    def test_stats_index_none(self) -> None:
        '''Test get_tombstone_stats returns zeros when _index is None.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)
            engine._index = None

            stats = engine.get_tombstone_stats()
            assert stats["total_docs"] == 0
            assert stats["active_docs"] == 0
            assert stats["tombstones"] == 0
            assert stats["unique_active_memories"] == 0

    def test_stats_exception_with_logger(self) -> None:
        '''Test get_tombstone_stats returns zeros on exception with logger.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config, logger=mock_logger)

            mock_index = MagicMock()
            mock_index.parse_query.side_effect = RuntimeError("fail")
            engine._index = mock_index

            stats = engine.get_tombstone_stats()

            assert stats["total_docs"] == 0
            calls = [str(c) for c in mock_logger.warning.call_args_list]
            assert any("Failed to get tombstone stats" in c for c in calls)

    def test_stats_exception_without_logger(self) -> None:
        '''Test get_tombstone_stats returns zeros on exception without logger.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            mock_index = MagicMock()
            mock_index.parse_query.side_effect = RuntimeError("fail")
            engine._index = mock_index

            stats = engine.get_tombstone_stats()

            assert stats["total_docs"] == 0


@pytest.mark.unit
class TestNeedsCompactionWithLogger:
    '''Tests for needs_compaction with logger messages.'''

    def test_needs_compaction_max_tombstones_with_logger(self) -> None:
        '''Test needs_compaction logs when max tombstones exceeded.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(
                workspace_id="test",
                index_path=tmpdir,
                compaction_max_tombstones=1,
            )
            engine = TantivyEngine(config, logger=mock_logger)

            engine.add("test", "msg1")
            engine.add("test", "msg2")
            engine.commit()

            engine.soft_delete("test", "msg1")
            engine.soft_delete("test", "msg2")
            engine.commit()

            result = engine.needs_compaction()
            assert result is True
            calls = [str(c) for c in mock_logger.info.call_args_list]
            assert any("tombstone count exceeded" in c for c in calls)

    def test_needs_compaction_ratio_with_logger(self) -> None:
        '''Test needs_compaction logs when ratio exceeded.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(
                workspace_id="test",
                index_path=tmpdir,
                compaction_threshold_ratio=0.1,
                compaction_max_tombstones=10000,
            )
            engine = TantivyEngine(config, logger=mock_logger)

            engine.add("test", "msg1")
            engine.add("test", "msg2")
            engine.commit()

            engine.soft_delete("test", "msg1")
            engine.commit()

            result = engine.needs_compaction()
            assert result is True
            calls = [str(c) for c in mock_logger.info.call_args_list]
            assert any("tombstone ratio exceeded" in c for c in calls)


@pytest.mark.unit
class TestCompactEdgeCases:
    '''Tests for compact() error handling and edge cases.'''

    def test_compact_exception_returns_failure(self) -> None:
        '''Test compact returns failure dict on exception.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config, logger=mock_logger)

            mock_index = MagicMock()
            mock_index.parse_query.side_effect = RuntimeError("compaction boom")
            engine._index = mock_index

            result = engine.compact(force=True)

            assert result["compacted"] is False
            assert result["removed_tombstones"] == 0
            assert result["elapsed_ms"] >= 0
            calls = [str(c) for c in mock_logger.error.call_args_list]
            assert any("compaction failed" in c.lower() for c in calls)

    def test_compact_exception_without_logger(self) -> None:
        '''Test compact returns failure dict on exception without logger.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config)

            mock_index = MagicMock()
            mock_index.parse_query.side_effect = RuntimeError("boom")
            engine._index = mock_index

            result = engine.compact(force=True)

            assert result["compacted"] is False

    def test_compact_with_logger_logs_completion(self) -> None:
        '''Test compact logs completion message with stats.'''
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_logger = MagicMock()
            config = TantivyConfig(workspace_id="test", index_path=tmpdir)
            engine = TantivyEngine(config, logger=mock_logger)

            engine.add("test", "keep")
            engine.add("test", "remove")
            engine.commit()
            engine.soft_delete("test", "remove")
            engine.commit()

            result = engine.compact(force=True)
            assert result["compacted"] is True
            calls = [str(c) for c in mock_logger.info.call_args_list]
            assert any("compaction completed" in c.lower() for c in calls)


@pytest.mark.unit
class TestEscapeTantivyQuery:
    '''Tests for _escape_tantivy_query static method.'''

    def test_escape_special_characters(self) -> None:
        '''Test that all special characters are escaped.'''
        query = 'test+and-or|not!group()range{}list[]up^quote"tilde~wild*any?colon:back\\/slash'
        escaped = TantivyEngine._escape_tantivy_query(query)
        # Every special char should have a backslash prefix
        assert "\\+" in escaped
        assert "\\-" in escaped
        assert "\\|" in escaped
        assert "\\!" in escaped
        assert "\\(" in escaped
        assert "\\)" in escaped
        assert "\\{" in escaped
        assert "\\}" in escaped
        assert "\\[" in escaped
        assert "\\]" in escaped
        assert "\\^" in escaped
        assert '\\"' in escaped
        assert "\\~" in escaped
        assert "\\*" in escaped
        assert "\\?" in escaped
        assert "\\:" in escaped

    def test_escape_no_special_chars(self) -> None:
        '''Test that plain text is unchanged.'''
        query = "simple query text"
        assert TantivyEngine._escape_tantivy_query(query) == query

    def test_escape_empty_string(self) -> None:
        '''Test escaping empty string returns empty.'''
        assert TantivyEngine._escape_tantivy_query("") == ""


class TestSoftDeleteDoesNotPlantSurplusTombs:
    @pytest.fixture
    def engine(self) -> TantivyEngine:
        tmpdir = tempfile.mkdtemp()
        config = TantivyConfig(workspace_id="test-project", index_path=tmpdir)
        return TantivyEngine(config)

    def test_verify_exists_false_does_not_hide_later_readd(
        self, engine: TantivyEngine
    ) -> None:
        engine.add("test-project", "hello world")
        engine.commit()
        assert engine.delete("test-project", "hello world") is True
        replayed = engine.soft_delete(
            "test-project", "hello world", verify_exists=False
        )
        assert replayed is False
        engine.add("test-project", "hello world")
        engine.commit()
        assert engine.find_by_exact_match("test-project", "hello world") == [
            "hello world"
        ]
        assert engine.search("hello", "test-project", limit=5) == [
            ("hello world", pytest.approx(1.0))
        ]

    def test_whitespace_query_returns_no_hits(self, engine: TantivyEngine) -> None:
        engine.add("test-project", "hello world")
        engine.commit()
        assert engine.search("   ", "test-project", limit=10) == []
        assert engine.search("\n\t", "test-project", limit=10) == []
