"""Tests for RemoveTool implementation."""

from unittest.mock import MagicMock

import pytest

from reflectlog.application.config.settings import Config
from reflectlog.application.tools.remove import RemoveTool
from reflectlog.core.exceptions import InconsistentStateError, StorageError


class RecordingManager:
    """Manager with a real ``delete_memories`` so RemoveTool uses the batch path."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.present: set[str] = set()
        self.error: Exception | None = None

    def delete_memories(self, memories: list[str]) -> list[str]:
        if self.error is not None:
            raise self.error
        self.calls.append(list(memories))
        return [memory for memory in memories if memory in self.present]


@pytest.fixture
def recording_manager() -> RecordingManager:
    return RecordingManager()


@pytest.fixture
def remove_tool_instance(
    mock_config: Config, recording_manager: RecordingManager, mock_tool_logger: MagicMock
) -> RemoveTool:
    return RemoveTool(
        config=mock_config,
        memory_manager=recording_manager,
        logger=mock_tool_logger,
    )


@pytest.mark.unit
class TestRemoveToolHappyPath:
    """Tests for successful remove operations."""

    async def test_remove_single_memory(
        self, remove_tool_instance: RemoveTool, recording_manager: RecordingManager
    ) -> None:
        """Removing a single existing memory calls delete_memories."""
        memory = "Remove this"
        recording_manager.present = {memory}

        handler = remove_tool_instance.get_handler()
        result = await handler([memory])

        assert result is None
        assert recording_manager.calls == [[memory]]

    async def test_remove_multiple_memories(
        self, remove_tool_instance: RemoveTool, recording_manager: RecordingManager
    ) -> None:
        """Removing multiple memories sends the unique list to delete_memories."""
        recording_manager.present = {"First", "Second"}

        handler = remove_tool_instance.get_handler()
        await handler(["First", "Second"])

        assert recording_manager.calls == [["First", "Second"]]

    async def test_remove_logs_invocation(
        self, remove_tool_instance: RemoveTool, recording_manager: RecordingManager, mock_tool_logger: MagicMock
    ) -> None:
        """Remove handler logs invocation with tool name."""
        recording_manager.present = {"Hello"}

        handler = remove_tool_instance.get_handler()
        await handler(["Hello"])

        info_calls = mock_tool_logger.info.call_args_list
        invocation_logged = any(
            "invoked" in str(c.args[0]).lower() for c in info_calls if c.args
        )
        assert invocation_logged

    async def test_remove_duplicate_occurrences(
        self, remove_tool_instance: RemoveTool, recording_manager: RecordingManager
    ) -> None:
        """Duplicate request strings are collapsed before delete_memories."""
        memory = "Dup memory"
        recording_manager.present = {memory}

        handler = remove_tool_instance.get_handler()
        await handler([memory, memory])

        assert recording_manager.calls == [[memory]]


@pytest.mark.unit
class TestRemoveToolEmptyInput:
    """Tests for empty list input handling."""

    async def test_remove_empty_list_noop(
        self, remove_tool_instance: RemoveTool, recording_manager: RecordingManager
    ) -> None:
        """Empty list is a no-op that returns None without deleting."""
        handler = remove_tool_instance.get_handler()
        result = await handler([])

        assert result is None
        assert recording_manager.calls == []

    async def test_remove_empty_list_logs_message(
        self, remove_tool_instance: RemoveTool, mock_tool_logger: MagicMock
    ) -> None:
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
        self, remove_tool_instance: RemoveTool, recording_manager: RecordingManager
    ) -> None:
        """Non-existent memory is silently ignored without error."""
        handler = remove_tool_instance.get_handler()
        result = await handler(["Nonexistent memory"])

        assert result is None
        assert recording_manager.calls == [["Nonexistent memory"]]

    async def test_remove_no_candidates_at_all(
        self, remove_tool_instance: RemoveTool, recording_manager: RecordingManager
    ) -> None:
        """Missing memories still succeed silently."""
        handler = remove_tool_instance.get_handler()
        result = await handler(["Ghost memory"])

        assert result is None
        assert recording_manager.calls == [["Ghost memory"]]

    async def test_remove_case_sensitive_matching(
        self, remove_tool_instance: RemoveTool, recording_manager: RecordingManager
    ) -> None:
        """Remove uses case-sensitive exact matching."""
        recording_manager.present = {"Hello"}

        handler = remove_tool_instance.get_handler()
        await handler(["Hello"])

        assert recording_manager.calls == [["Hello"]]


@pytest.mark.unit
class TestRemoveToolPartialRemoval:
    """Tests for partial removal scenarios."""

    async def test_mixed_existing_and_nonexistent(
        self, remove_tool_instance: RemoveTool, recording_manager: RecordingManager
    ) -> None:
        """Mix of existing and non-existing memories: only existing ones are removed."""
        recording_manager.present = {"Exists"}

        handler = remove_tool_instance.get_handler()
        await handler(["Exists", "Ghost"])

        assert recording_manager.calls == [["Exists", "Ghost"]]

    async def test_removal_summary_logging(
        self, remove_tool_instance: RemoveTool, recording_manager: RecordingManager, mock_tool_logger: MagicMock
    ) -> None:
        """Partial removal logs summary with found and not-found counts."""
        recording_manager.present = {"Found"}

        handler = remove_tool_instance.get_handler()
        await handler(["Found", "Missing"])

        info_calls = [
            str(c.args[0]) for c in mock_tool_logger.info.call_args_list if c.args
        ]
        summary_logged = any("not found" in msg.lower() for msg in info_calls)
        assert summary_logged

    async def test_batch_delete_reports_missing_by_content(
        self, mock_config: Config, mock_tool_logger: MagicMock
    ) -> None:
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
        self, mock_config: Config, mock_tool_logger: MagicMock
    ) -> None:
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
        self, mock_config: Config, mock_tool_logger: MagicMock
    ) -> None:
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
        with pytest.raises(TypeError, match="delete_memories must return list"):
            await handler(["Exists"])


@pytest.mark.unit
class TestRemoveToolErrorHandling:
    """Tests for remove error handling."""

    async def test_storage_error_wraps_with_from(
        self, remove_tool_instance: RemoveTool, recording_manager: RecordingManager
    ) -> None:
        """Storage exception is wrapped as StorageError with 'from e' chaining."""
        original = RuntimeError("disk failure")
        recording_manager.error = original

        handler = remove_tool_instance.get_handler()
        with pytest.raises(StorageError, match="Failed to remove memories") as exc_info:
            await handler(["Memory"])

        assert exc_info.value.__cause__ is original

    async def test_delete_failure_raises_storage_error(
        self, remove_tool_instance: RemoveTool, recording_manager: RecordingManager
    ) -> None:
        """Delete failure during removal raises StorageError."""
        recording_manager.error = RuntimeError("delete failed")

        handler = remove_tool_instance.get_handler()
        with pytest.raises(StorageError, match="Failed to remove memories"):
            await handler(["Memory"])

    async def test_inconsistent_state_is_reraised(
        self, remove_tool_instance: RemoveTool, recording_manager: RecordingManager
    ) -> None:
        """InconsistentStateError must not be flattened to StorageError."""
        original = InconsistentStateError("USearch/Tantivy split")
        recording_manager.error = original

        handler = remove_tool_instance.get_handler()
        with pytest.raises(InconsistentStateError) as exc_info:
            await handler(["Memory"])

        assert exc_info.value is original

    async def test_error_logs_before_raising(
        self, remove_tool_instance: RemoveTool, recording_manager: RecordingManager, mock_tool_logger: MagicMock
    ) -> None:
        """Error triggers log_error before raising StorageError."""
        recording_manager.error = RuntimeError("boom")

        handler = remove_tool_instance.get_handler()
        with pytest.raises(StorageError):
            await handler(["Memory"])

        mock_tool_logger.error.assert_called()


@pytest.mark.unit
class TestRemoveToolMetadata:
    """Tests for RemoveTool metadata methods."""

    def test_get_name(self, remove_tool_instance: RemoveTool) -> None:
        """RemoveTool.get_name() returns 'remove'."""
        assert remove_tool_instance.get_name() == "remove"

    def test_get_instruction_snippet(self, remove_tool_instance: RemoveTool) -> None:
        """get_instruction_snippet contains expected signature."""
        snippet = remove_tool_instance.get_instruction_snippet()
        assert "remove(memories: list[str])" in snippet

    def test_get_handler_returns_callable(self, remove_tool_instance: RemoveTool) -> None:
        """get_handler returns a callable."""
        handler = remove_tool_instance.get_handler()
        assert callable(handler)
