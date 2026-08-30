"""Tests for SearchTool implementation."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from reflectlog.application.tools.search import SearchTool
from reflectlog.core.exceptions import SearchError


@pytest.mark.unit
class TestSearchToolHappyPath:
    """Tests for successful search operations."""

    async def test_search_returns_results(
        self, search_tool_instance: SearchTool, mock_memory_manager: MagicMock
    ) -> None:
        """Search returns list of matching memories."""
        expected = ["Python tutorial", "JavaScript guide"]
        mock_memory_manager.search = AsyncMock(return_value=expected)

        handler = search_tool_instance.get_handler()
        result = await handler("programming")

        assert result == expected
        mock_memory_manager.search.assert_awaited_once()

    async def test_search_passes_query_and_limit(
        self, search_tool_instance: SearchTool, mock_memory_manager: MagicMock
    ) -> None:
        """Search forwards query and config limit to MemoryManager.search."""
        mock_memory_manager.search = AsyncMock(return_value=["result"])

        handler = search_tool_instance.get_handler()
        await handler("test query")

        call_kwargs = mock_memory_manager.search.call_args.kwargs
        assert call_kwargs["limit"] == search_tool_instance.config.search_limit

        call_args = mock_memory_manager.search.call_args.args
        assert call_args[0] == "test query"

    async def test_search_returns_ordered_results(
        self, search_tool_instance: SearchTool, mock_memory_manager: MagicMock
    ) -> None:
        """Search preserves result order from MemoryManager."""
        ordered = ["Most relevant", "Somewhat relevant", "Least relevant"]
        mock_memory_manager.search = AsyncMock(return_value=ordered)

        handler = search_tool_instance.get_handler()
        result = await handler("relevance test")

        assert result == ordered

    async def test_search_logs_invocation(
        self, search_tool_instance: SearchTool, mock_memory_manager: MagicMock, mock_tool_logger: MagicMock
    ) -> None:
        """Search logs invocation with query parameter."""
        mock_memory_manager.search = AsyncMock(return_value=[])

        handler = search_tool_instance.get_handler()
        await handler("hello")

        info_calls = mock_tool_logger.info.call_args_list
        invocation_logged = any(
            "invoked" in str(c.args[0]).lower() for c in info_calls if c.args
        )
        assert invocation_logged

    async def test_search_logs_completion(
        self, search_tool_instance: SearchTool, mock_memory_manager: MagicMock, mock_tool_logger: MagicMock
    ) -> None:
        """Search logs completion with result count."""
        mock_memory_manager.search = AsyncMock(return_value=["a", "b"])

        handler = search_tool_instance.get_handler()
        await handler("test")

        info_calls = mock_tool_logger.info.call_args_list
        completion_logged = any(
            "completed successfully" in str(c.args[0]).lower()
            for c in info_calls
            if c.args
        )
        assert completion_logged


@pytest.mark.unit
class TestSearchToolNoResults:
    """Tests for empty result scenarios."""

    async def test_search_returns_empty_list(
        self, search_tool_instance: SearchTool, mock_memory_manager: MagicMock
    ) -> None:
        """Search returns empty list when no matches found."""
        mock_memory_manager.search = AsyncMock(return_value=[])

        handler = search_tool_instance.get_handler()
        result = await handler("nonexistent topic")

        assert result == []

    async def test_search_no_results_logs_message(
        self, search_tool_instance: SearchTool, mock_memory_manager: MagicMock, mock_tool_logger: MagicMock
    ) -> None:
        """No-results search logs appropriate message."""
        mock_memory_manager.search = AsyncMock(return_value=[])

        handler = search_tool_instance.get_handler()
        await handler("nothing here")

        info_calls = [
            str(c.args[0]) for c in mock_tool_logger.info.call_args_list if c.args
        ]
        completed = any("complet" in msg.lower() for msg in info_calls)
        assert completed


@pytest.mark.unit
class TestSearchToolErrorHandling:
    """Tests for search error handling."""

    async def test_search_error_wraps_with_from(
        self, search_tool_instance: SearchTool, mock_memory_manager: MagicMock
    ) -> None:
        """Search exception is wrapped as SearchError with 'from e' chaining."""
        original = RuntimeError("index corrupted")
        mock_memory_manager.search = AsyncMock(side_effect=original)

        handler = search_tool_instance.get_handler()
        with pytest.raises(
            SearchError, match="Failed to search memory store"
        ) as exc_info:
            await handler("test")

        assert exc_info.value.__cause__ is original

    async def test_search_error_logs_before_raising(
        self, search_tool_instance: SearchTool, mock_memory_manager: MagicMock, mock_tool_logger: MagicMock
    ) -> None:
        """Search error triggers log_error before raising."""
        mock_memory_manager.search = AsyncMock(side_effect=RuntimeError("oops"))

        handler = search_tool_instance.get_handler()
        with pytest.raises(SearchError):
            await handler("test")

        mock_tool_logger.error.assert_called()

    async def test_search_error_includes_query_context(
        self, search_tool_instance: SearchTool, mock_memory_manager: MagicMock, mock_tool_logger: MagicMock
    ) -> None:
        """Search error log includes query context."""
        mock_memory_manager.search = AsyncMock(side_effect=ValueError("bad"))

        handler = search_tool_instance.get_handler()
        with pytest.raises(SearchError):
            await handler("my query")

        error_call = mock_tool_logger.error.call_args
        assert "search" in error_call.args[0].lower()


@pytest.mark.unit
class TestSearchToolMetadata:
    """Tests for SearchTool metadata methods."""

    def test_get_name(self, search_tool_instance: SearchTool) -> None:
        """SearchTool.get_name() returns 'search'."""
        assert search_tool_instance.get_name() == "search"

    def test_get_instruction_snippet(self, search_tool_instance: SearchTool) -> None:
        """get_instruction_snippet contains expected signature."""
        snippet = search_tool_instance.get_instruction_snippet()
        assert "search(query: str)" in snippet

    def test_get_handler_returns_callable(self, search_tool_instance: SearchTool) -> None:
        """get_handler returns a callable."""
        handler = search_tool_instance.get_handler()
        assert callable(handler)
