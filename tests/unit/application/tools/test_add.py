"""Tests for AddTool implementation."""

from unittest.mock import AsyncMock

import pytest

from reflectlog.application.memory.add_phases import AddResult, ReplacementInfo
from reflectlog.core.exceptions import StorageError


@pytest.mark.unit
class TestAddToolHappyPath:
    """Tests for successful add operations."""

    async def test_add_single_memory(self, add_tool_instance, mock_memory_manager):
        """Adding a single valid memory calls add_memories_async and returns None."""
        handler = add_tool_instance.get_handler()
        result = await handler(["Remember this"])

        assert result is None
        mock_memory_manager.add_memories_async.assert_awaited_once_with(
            ["Remember this"], dry_run=False
        )

    async def test_add_multiple_memories(self, add_tool_instance, mock_memory_manager):
        """Adding multiple memories passes entire list to add_memories_async."""
        mock_memory_manager.add_memories_async = AsyncMock(
            return_value=AddResult(
                stored_count=3, skipped_count=0, replaced_count=0, replacements=[]
            )
        )
        handler = add_tool_instance.get_handler()
        await handler(["First", "Second", "Third"])

        call_args = mock_memory_manager.add_memories_async.call_args
        assert call_args.args[0] == ["First", "Second", "Third"]

    async def test_add_with_dry_run(self, add_tool_instance, mock_memory_manager):
        """dry_run=True is forwarded to add_memories_async."""
        handler = add_tool_instance.get_handler()
        await handler(["Test memory"], dry_run=True)

        mock_memory_manager.add_memories_async.assert_awaited_once_with(
            ["Test memory"], dry_run=True
        )

    async def test_add_logs_invocation(self, add_tool_instance, mock_tool_logger):
        """Add handler calls log_invocation with tool name 'add'."""
        handler = add_tool_instance.get_handler()
        await handler(["Hello"])

        # log_invocation calls logger.info with tool='add'
        info_calls = mock_tool_logger.info.call_args_list
        invocation_logged = any(
            "add" in str(c.args[0]) and "invoked" in str(c.args[0]).lower()
            for c in info_calls
            if c.args
        )
        assert invocation_logged

    async def test_add_with_replacements(
        self, add_tool_instance, mock_memory_manager, mock_tool_logger
    ):
        """Replacement details are logged when memories are replaced."""
        replacement = ReplacementInfo(
            old_memory="Old preference for Python",
            new_memory="New preference for Rust",
            confidence=0.85,
            reason="Updated language preference",
        )
        mock_memory_manager.add_memories_async = AsyncMock(
            return_value=AddResult(
                stored_count=0,
                skipped_count=0,
                replaced_count=1,
                replacements=[replacement],
            )
        )
        handler = add_tool_instance.get_handler()
        await handler(["New preference for Rust"])

        # Verify replacement was logged
        info_calls = [str(c) for c in mock_tool_logger.info.call_args_list]
        replacement_logged = any("Replaced" in c for c in info_calls)
        assert replacement_logged


@pytest.mark.unit
class TestAddToolEmptyInput:
    """Tests for empty list input handling."""

    async def test_add_empty_list_returns_none(
        self, add_tool_instance, mock_memory_manager
    ):
        """Empty list is a no-op that returns None without calling storage."""
        handler = add_tool_instance.get_handler()
        result = await handler([])

        assert result is None
        mock_memory_manager.add_memories_async.assert_not_awaited()

    async def test_add_empty_list_logs_noop(self, add_tool_instance, mock_tool_logger):
        """Empty list add logs appropriate message about no-op."""
        handler = add_tool_instance.get_handler()
        await handler([])

        info_calls = [
            str(c.args[0]) for c in mock_tool_logger.info.call_args_list if c.args
        ]
        noop_logged = any("empty" in msg.lower() for msg in info_calls)
        assert noop_logged


@pytest.mark.unit
class TestAddToolValidation:
    """Tests for input validation errors."""

    async def test_add_empty_string_raises_value_error(self, add_tool_instance):
        """Empty string memory raises ValueError."""
        handler = add_tool_instance.get_handler()
        with pytest.raises(ValueError):
            await handler([""])

    async def test_add_whitespace_only_raises_value_error(self, add_tool_instance):
        """Whitespace-only memory raises ValueError."""
        handler = add_tool_instance.get_handler()
        with pytest.raises(ValueError):
            await handler(["   "])

    async def test_add_too_long_memory_raises_value_error(self, add_tool_instance):
        """Memory exceeding max length raises ValueError."""
        handler = add_tool_instance.get_handler()
        too_long = "x" * 30721
        with pytest.raises(ValueError, match="too long"):
            await handler([too_long])

    async def test_add_non_string_raises_value_error(self, add_tool_instance):
        """Non-string item in memories raises ValueError."""
        handler = add_tool_instance.get_handler()
        with pytest.raises(ValueError):
            await handler([123])

    async def test_add_validation_does_not_call_storage(
        self, add_tool_instance, mock_memory_manager
    ):
        """Validation errors prevent any storage call."""
        handler = add_tool_instance.get_handler()
        with pytest.raises(ValueError):
            await handler([""])
        mock_memory_manager.add_memories_async.assert_not_awaited()


@pytest.mark.unit
class TestAddToolErrorHandling:
    """Tests for storage error handling."""

    async def test_storage_error_wraps_with_from(
        self, add_tool_instance, mock_memory_manager
    ):
        """Storage exception is wrapped as StorageError with 'from e' chaining."""
        original = RuntimeError("disk full")
        mock_memory_manager.add_memories_async = AsyncMock(side_effect=original)

        handler = add_tool_instance.get_handler()
        with pytest.raises(StorageError, match="Failed to add memories") as exc_info:
            await handler(["Valid memory"])

        assert exc_info.value.__cause__ is original

    async def test_storage_error_logs_before_raising(
        self, add_tool_instance, mock_memory_manager, mock_tool_logger
    ):
        """Storage error triggers log_error before raising."""
        mock_memory_manager.add_memories_async = AsyncMock(
            side_effect=RuntimeError("write failed")
        )

        handler = add_tool_instance.get_handler()
        with pytest.raises(StorageError):
            await handler(["Valid memory"])

        mock_tool_logger.error.assert_called()


@pytest.mark.unit
class TestAddToolMetadata:
    """Tests for AddTool metadata methods."""

    def test_get_name(self, add_tool_instance):
        """AddTool.get_name() returns 'add'."""
        assert add_tool_instance.get_name() == "add"

    def test_get_instruction_snippet(self, add_tool_instance):
        """get_instruction_snippet contains expected signature."""
        snippet = add_tool_instance.get_instruction_snippet()
        assert "add(memories: list[str])" in snippet
