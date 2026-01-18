"""Unit tests for MessageStore SQLite storage."""

import os
import tempfile
import threading
from unittest.mock import MagicMock

import pytest

from reflectlog.application.exceptions import StorageError
from reflectlog.infrastructure.message_store import MessageStore


class TestMessageStoreInitialization:
    """Tests for MessageStore initialization."""

    def test_creates_database_file(self) -> None:
        """Database file should be created on first access."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MessageStore(db_path=db_path)

            # Access connection to trigger creation
            _ = store.connection

            assert os.path.exists(db_path)
            store.close()

    def test_creates_parent_directories(self) -> None:
        """Parent directories should be created if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "nested", "path", "test.db")
            store = MessageStore(db_path=db_path)

            _ = store.connection

            assert os.path.exists(db_path)
            store.close()

    def test_lazy_initialization(self) -> None:
        """Connection should not be created until accessed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MessageStore(db_path=db_path)

            # Before accessing connection, file should not exist
            assert not os.path.exists(db_path)

            # Access connection
            _ = store.connection

            # Now file should exist
            assert os.path.exists(db_path)
            store.close()


class TestMessageStoreInsert:
    """Tests for MessageStore.insert method."""

    def test_insert_returns_id(self) -> None:
        """Insert should return auto-increment ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MessageStore(db_path=db_path)

            msg_id = store.insert("user1", "Hello world")

            assert msg_id == 1
            store.close()

    def test_insert_increments_id(self) -> None:
        """Subsequent inserts should return incrementing IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MessageStore(db_path=db_path)

            id1 = store.insert("user1", "Message 1")
            id2 = store.insert("user1", "Message 2")
            id3 = store.insert("user1", "Message 3")

            assert id1 == 1
            assert id2 == 2
            assert id3 == 3
            store.close()

    def test_insert_duplicate_raises_error(self) -> None:
        """Inserting duplicate message for same user should raise error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MessageStore(db_path=db_path)

            store.insert("user1", "Hello world")

            with pytest.raises(StorageError, match="Duplicate message"):
                store.insert("user1", "Hello world")

            store.close()

    def test_insert_same_message_different_users(self) -> None:
        """Same message for different users should be allowed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MessageStore(db_path=db_path)

            id1 = store.insert("user1", "Hello world")
            id2 = store.insert("user2", "Hello world")

            assert id1 == 1
            assert id2 == 2
            store.close()


class TestMessageStoreGet:
    """Tests for MessageStore.get method."""

    def test_get_existing_message(self) -> None:
        """Get should return MessageRecord for existing ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MessageStore(db_path=db_path)

            msg_id = store.insert("user1", "Hello world")
            record = store.get(msg_id)

            assert record is not None
            assert record.id == msg_id
            assert record.project_id == "user1"
            assert record.message == "Hello world"
            store.close()

    def test_get_nonexistent_message(self) -> None:
        """Get should return None for non-existent ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MessageStore(db_path=db_path)

            record = store.get(999)

            assert record is None
            store.close()


class TestMessageStoreGetAll:
    """Tests for MessageStore.get_all method."""

    def test_get_all_empty(self) -> None:
        """Get all should return empty list when no messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MessageStore(db_path=db_path)

            messages = store.get_all("user1")

            assert messages == []
            store.close()

    def test_get_all_single_user(self) -> None:
        """Get all should return messages for specific user."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MessageStore(db_path=db_path)

            store.insert("user1", "Message 1")
            store.insert("user1", "Message 2")
            store.insert("user2", "Other message")

            messages = store.get_all("user1")

            assert len(messages) == 2
            assert "Message 1" in messages
            assert "Message 2" in messages
            assert "Other message" not in messages
            store.close()

    def test_get_all_preserves_order(self) -> None:
        """Get all should return messages in insertion order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MessageStore(db_path=db_path)

            store.insert("user1", "First")
            store.insert("user1", "Second")
            store.insert("user1", "Third")

            messages = store.get_all("user1")

            assert messages == ["First", "Second", "Third"]
            store.close()


class TestMessageStoreDelete:
    """Tests for MessageStore.delete method."""

    def test_delete_existing_message(self) -> None:
        """Delete should return True for existing message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MessageStore(db_path=db_path)

            msg_id = store.insert("user1", "Hello world")
            deleted = store.delete(msg_id)

            assert deleted is True
            assert store.get(msg_id) is None
            store.close()

    def test_delete_nonexistent_message(self) -> None:
        """Delete should return False for non-existent message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MessageStore(db_path=db_path)

            deleted = store.delete(999)

            assert deleted is False
            store.close()


class TestMessageStoreExists:
    """Tests for MessageStore.exists method."""

    def test_exists_returns_true_for_existing(self) -> None:
        """Exists should return True for existing message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MessageStore(db_path=db_path)

            store.insert("user1", "Hello world")

            assert store.exists("user1", "Hello world") is True
            store.close()

    def test_exists_returns_false_for_nonexistent(self) -> None:
        """Exists should return False for non-existent message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MessageStore(db_path=db_path)

            assert store.exists("user1", "Hello world") is False
            store.close()

    def test_exists_checks_project_id(self) -> None:
        """Exists should check project_id, not just message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MessageStore(db_path=db_path)

            store.insert("user1", "Hello world")

            assert store.exists("user1", "Hello world") is True
            assert store.exists("user2", "Hello world") is False
            store.close()


class TestMessageStoreGetIdByMessage:
    """Tests for MessageStore.get_id_by_message method."""

    def test_get_id_by_message_existing(self) -> None:
        """Get ID by message should return ID for existing message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MessageStore(db_path=db_path)

            msg_id = store.insert("user1", "Hello world")
            found_id = store.get_id_by_message("user1", "Hello world")

            assert found_id == msg_id
            store.close()

    def test_get_id_by_message_nonexistent(self) -> None:
        """Get ID by message should return None for non-existent message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MessageStore(db_path=db_path)

            found_id = store.get_id_by_message("user1", "Hello world")

            assert found_id is None
            store.close()


class TestMessageStoreThreadSafety:
    """Tests for MessageStore thread safety."""

    @pytest.mark.skip(reason="Concurrent test causes system instability in CI")
    def test_concurrent_inserts(self) -> None:
        """Concurrent inserts should be thread-safe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MessageStore(db_path=db_path)

            results: list[int] = []
            lock = threading.Lock()

            def insert_message(i: int) -> None:
                msg_id = store.insert("user1", f"Message {i}")
                with lock:
                    results.append(msg_id)

            threads = [
                threading.Thread(target=insert_message, args=(i,)) for i in range(10)
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All inserts should succeed with unique IDs
            assert len(results) == 10
            assert len(set(results)) == 10
            store.close()


class TestMessageStoreLogging:
    """Tests for MessageStore logging."""

    def test_logs_insert_on_debug(self) -> None:
        """Insert should log on debug level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            logger = MagicMock()
            store = MessageStore(db_path=db_path, logger=logger)

            store.insert("user1", "Hello world")

            logger.debug.assert_called()
            store.close()

    def test_logs_delete_on_debug(self) -> None:
        """Delete should log on debug level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            logger = MagicMock()
            store = MessageStore(db_path=db_path, logger=logger)

            msg_id = store.insert("user1", "Hello world")
            logger.reset_mock()

            store.delete(msg_id)

            logger.debug.assert_called()
            store.close()
