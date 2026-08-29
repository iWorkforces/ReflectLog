"""Tests for RemoveTool implementation."""

from unittest.mock import MagicMock, patch

import pytest

from reflectlog.application.tools.remove import RemoveTool
from reflectlog.core.exceptions import StorageError


@pytest.mark.unit
class TestRemoveToolHappyPath:
    """Tests for successful remove operations."""

    async def test_remove_single_memory(
        self, remove_tool_instance, mock_memory_manager
    ):
        """Removing a single existing memory calls search_for_removal and delete_by_memory."""
        memory = "Remove this"
        mock_memory_manager.search_for_removal.return_value = [
            {"id": "1", "memory": memory, "score": 0.95}
        ]

        handler = remove_tool_instance.get_handler()
        result = await handler([memory])

        assert result is None
        mock_memory_manager.search_for_removal.assert_called_once_with(memory)
        mock_memory_manager.delete_by_memory.assert_called_once_with(memory)

    async def test_remove_multiple_memories(
        self, remove_tool_instance, mock_memory_manager
    ):
        """Removing multiple memories processes each sequentially."""

        def search_side_effect(query):
            return [{"id": "x", "memory": query, "score": 0.95}]

        mock_memory_manager.search_for_removal.side_effect = search_side_effect

        handler = remove_tool_instance.get_handler()
        await handler(["First", "Second"])

        assert mock_memory_manager.search_for_removal.call_count == 2
        assert mock_memory_manager.delete_by_memory.call_count == 2

    async def test_remove_logs_invocation(
        self, remove_tool_instance, mock_memory_manager, mock_tool_logger
    ):
        """Remove handler logs invocation with tool name."""
        mock_memory_manager.search_for_removal.return_value = [
            {"id": "1", "memory": "Hello", "score": 0.95}
        ]

        handler = remove_tool_instance.get_handler()
        await handler(["Hello"])

        info_calls = mock_tool_logger.info.call_args_list
        invocation_logged = any(
            "invoked" in str(c.args[0]).lower() for c in info_calls if c.args
        )
        assert invocation_logged

    async def test_remove_duplicate_occurrences(
        self, remove_tool_instance, mock_memory_manager
    ):
        """Multiple exact matches for a single memory are all deleted."""
        memory = "Dup memory"
        mock_memory_manager.search_for_removal.return_value = [
            {"id": "1", "memory": memory, "score": 0.95},
            {"id": "2", "memory": memory, "score": 0.90},
            {"id": "3", "memory": memory, "score": 0.85},
        ]

        handler = remove_tool_instance.get_handler()
        await handler([memory])

        assert mock_memory_manager.delete_by_memory.call_count == 3


@pytest.mark.unit
class TestRemoveToolEmptyInput:
    """Tests for empty list input handling."""

    async def test_remove_empty_list_noop(
        self, remove_tool_instance, mock_memory_manager
    ):
        """Empty list is a no-op that returns None without searching."""
        handler = remove_tool_instance.get_handler()
        result = await handler([])

        assert result is None
        mock_memory_manager.search_for_removal.assert_not_called()
        mock_memory_manager.delete_by_memory.assert_not_called()

    async def test_remove_empty_list_logs_message(
        self, remove_tool_instance, mock_tool_logger
    ):
        """Empty list remove logs skip message."""
        handler = remove_tool_instance.get_handler()
        await handler([])

        info_calls = [
            str(c.args[0]) for c in mock_tool_logger.info.call_args_list if c.args
        ]
        skip_logged = any("empty" in msg.lower() for msg in info_calls)
        assert skip_logged


@pytest.mark.unit
class TestRemoveToolNotFound:
    """Tests for memory not found scenarios."""

    async def test_remove_nonexistent_silently_ignored(
        self, remove_tool_instance, mock_memory_manager
    ):
        """Non-existent memory is silently ignored without error."""
        mock_memory_manager.search_for_removal.return_value = [
            {"id": "1", "memory": "Similar but different", "score": 0.80}
        ]

        handler = remove_tool_instance.get_handler()
        result = await handler(["Nonexistent memory"])

        assert result is None
        mock_memory_manager.delete_by_memory.assert_not_called()

    async def test_remove_no_candidates_at_all(
        self, remove_tool_instance, mock_memory_manager
    ):
        """Zero search candidates still succeeds silently."""
        mock_memory_manager.search_for_removal.return_value = []

        handler = remove_tool_instance.get_handler()
        result = await handler(["Ghost memory"])

        assert result is None
        mock_memory_manager.delete_by_memory.assert_not_called()

    async def test_remove_case_sensitive_matching(
        self, remove_tool_instance, mock_memory_manager
    ):
        """Remove uses case-sensitive exact matching."""
        mock_memory_manager.search_for_removal.return_value = [
            {"id": "1", "memory": "Hello", "score": 0.95},
            {"id": "2", "memory": "hello", "score": 0.90},
            {"id": "3", "memory": "HELLO", "score": 0.85},
        ]

        handler = remove_tool_instance.get_handler()
        await handler(["Hello"])

        # Only exact case match "Hello" should be deleted
        assert mock_memory_manager.delete_by_memory.call_count == 1
        mock_memory_manager.delete_by_memory.assert_called_with("Hello")


@pytest.mark.unit
class TestRemoveToolPartialRemoval:
    """Tests for partial removal scenarios."""

    async def test_mixed_existing_and_nonexistent(
        self, remove_tool_instance, mock_memory_manager
    ):
        """Mix of existing and non-existing memories: only existing ones are removed."""

        def search_side_effect(query):
            if query == "Exists":
                return [{"id": "1", "memory": "Exists", "score": 0.95}]
            return []

        mock_memory_manager.search_for_removal.side_effect = search_side_effect

        handler = remove_tool_instance.get_handler()
        await handler(["Exists", "Ghost"])

        assert mock_memory_manager.search_for_removal.call_count == 2
        assert mock_memory_manager.delete_by_memory.call_count == 1
        mock_memory_manager.delete_by_memory.assert_called_with("Exists")

    async def test_removal_summary_logging(
        self, remove_tool_instance, mock_memory_manager, mock_tool_logger
    ):
        """Partial removal logs summary with found and not-found counts."""

        def search_side_effect(query):
            if query == "Found":
                return [{"id": "1", "memory": "Found", "score": 0.95}]
            return []

        mock_memory_manager.search_for_removal.side_effect = search_side_effect

        handler = remove_tool_instance.get_handler()
        await handler(["Found", "Missing"])

        info_calls = [
            str(c.args[0]) for c in mock_tool_logger.info.call_args_list if c.args
        ]
        summary_logged = any("not found" in msg.lower() for msg in info_calls)
        assert summary_logged

    async def test_batch_delete_reports_missing_by_content(
        self, mock_config, mock_tool_logger
    ):
        """delete_memories results are compared by content, not list suffix."""

        class FakeManager:
            def delete_memories(self, memories: list[str]) -> list[str]:
                return [memory for memory in memories if memory == "Exists"]

        tool = RemoveTool(
            config=mock_config,
            memory_manager=FakeManager(),
            logger=mock_tool_logger,
        )
        handler = tool.get_handler()
        await handler(["Exists", "Ghost"])

        info_calls = [
            str(c.args[0]) for c in mock_tool_logger.info.call_args_list if c.args
        ]
        summary = next(msg for msg in info_calls if "not found" in msg.lower())
        assert "1 memory" in summary

    async def test_delete_memories_returns_deleted_contents(
        self, mock_config, mock_tool_logger
    ):
        """Production batch path reports the contents delete_memories returned."""

        class FakeManager:
            def delete_memories(self, memories: list[str]) -> list[str]:
                return ["Kept"]

        tool = RemoveTool(
            config=mock_config,
            memory_manager=FakeManager(),
            logger=mock_tool_logger,
        )
        handler = tool.get_handler()
        await handler(["Kept"])

        info_calls = [
            str(c.args[0]) for c in mock_tool_logger.info.call_args_list if c.args
        ]
        assert any("Kept" in msg or "1" in msg for msg in info_calls)

    async def test_delete_memories_non_list_fails_closed(
        self, mock_config, mock_tool_logger
    ):
        """A non-list delete_memories return must not look like success."""

        class FakeManager:
            def delete_memories(self, memories: list[str]) -> int:
                return 1

        tool = RemoveTool(
            config=mock_config,
            memory_manager=FakeManager(),
            logger=mock_tool_logger,
        )
        handler = tool.get_handler()
        with pytest.raises(StorageError, match="Failed to remove memories"):
            await handler(["Exists"])


@pytest.mark.unit
class TestRemoveToolErrorHandling:
    """Tests for remove error handling."""

    async def test_storage_error_wraps_with_from(
        self, remove_tool_instance, mock_memory_manager
    ):
        """Storage exception is wrapped as StorageError with 'from e' chaining."""
        original = RuntimeError("disk failure")
        mock_memory_manager.search_for_removal.side_effect = original

        handler = remove_tool_instance.get_handler()
        with pytest.raises(StorageError, match="Failed to remove memories") as exc_info:
            await handler(["Memory"])

        assert exc_info.value.__cause__ is original

    async def test_delete_failure_raises_storage_error(
        self, remove_tool_instance, mock_memory_manager
    ):
        """Delete failure during removal raises StorageError."""
        mock_memory_manager.search_for_removal.return_value = [
            {"id": "1", "memory": "Memory", "score": 0.95}
        ]
        mock_memory_manager.delete_by_memory.side_effect = RuntimeError("delete failed")

        handler = remove_tool_instance.get_handler()
        with pytest.raises(StorageError, match="Failed to remove memories"):
            await handler(["Memory"])

    async def test_error_logs_before_raising(
        self, remove_tool_instance, mock_memory_manager, mock_tool_logger
    ):
        """Error triggers log_error before raising StorageError."""
        mock_memory_manager.search_for_removal.side_effect = RuntimeError("boom")

        handler = remove_tool_instance.get_handler()
        with pytest.raises(StorageError):
            await handler(["Memory"])

        mock_tool_logger.error.assert_called()


@pytest.mark.unit
class TestRemoveToolMetadata:
    """Tests for RemoveTool metadata methods."""

    def test_get_name(self, remove_tool_instance):
        """RemoveTool.get_name() returns 'remove'."""
        assert remove_tool_instance.get_name() == "remove"

    def test_get_instruction_snippet(self, remove_tool_instance):
        """get_instruction_snippet contains expected signature."""
        snippet = remove_tool_instance.get_instruction_snippet()
        assert "remove(memories: list[str])" in snippet

    def test_get_handler_returns_callable(self, remove_tool_instance):
        """get_handler returns a callable."""
        handler = remove_tool_instance.get_handler()
        assert callable(handler)
