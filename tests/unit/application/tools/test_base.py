"""Tests for BaseTool abstract class."""

from unittest.mock import MagicMock

import pytest

from reflectlog.application.tools.base import BaseTool
from reflectlog.application.config.settings import Config
from reflectlog.application.memory.manager import MemoryManager


class ConcreteTool(BaseTool):
    """Concrete implementation for testing the abstract BaseTool."""

    def get_name(self) -> str:
        """Get the tool name."""
        return "test_tool"

    def get_handler(self):
        """Get a no-op handler."""

        async def handler() -> None:
            pass

        return handler

    def get_instruction_snippet(self) -> str:
        """Get the instruction snippet."""
        return "    • test_tool()\n      A test tool."


@pytest.mark.unit
class TestBaseToolInitialization:
    """Tests for BaseTool.__init__."""

    def test_init_with_valid_args(
        self, mock_config: Config, mock_memory_manager: MemoryManager, mock_tool_logger: MagicMock
    ) -> None:
        """BaseTool stores config, memory, and logger."""
        tool = ConcreteTool(
            config=mock_config,
            memory_manager=mock_memory_manager,
            logger=mock_tool_logger,
        )
        assert tool.config is mock_config
        assert tool.memory is mock_memory_manager
        assert tool.logger is mock_tool_logger

    def test_init_raises_on_none_logger(self, mock_config: Config, mock_memory_manager: MemoryManager) -> None:
        """BaseTool raises ValueError when logger is None."""
        with pytest.raises(ValueError, match="logger is required"):
            ConcreteTool(
                config=mock_config,
                memory_manager=mock_memory_manager,
                logger=None,
            )


@pytest.mark.unit
class TestBaseToolLogging:
    """Tests for BaseTool logging helper methods."""

    def test_log_invocation_basic(
        self, mock_config: Config, mock_memory_manager: MemoryManager, mock_tool_logger: MagicMock
    ) -> None:
        """log_invocation logs info with tool name."""
        tool = ConcreteTool(
            config=mock_config,
            memory_manager=mock_memory_manager,
            logger=mock_tool_logger,
        )
        tool.log_invocation("add")

        mock_tool_logger.info.assert_called_once()
        call_args = mock_tool_logger.info.call_args
        assert "add" in call_args.args[0]
        assert call_args.kwargs["extra"]["tool"] == "add"

    def test_log_invocation_with_kwargs(
        self, mock_config: Config, mock_memory_manager: MemoryManager, mock_tool_logger: MagicMock
    ) -> None:
        """log_invocation passes extra kwargs into the extra dict."""
        tool = ConcreteTool(
            config=mock_config,
            memory_manager=mock_memory_manager,
            logger=mock_tool_logger,
        )
        tool.log_invocation("search", query="hello", count=5)

        extra = mock_tool_logger.info.call_args.kwargs["extra"]
        assert extra["tool"] == "search"
        assert extra["query"] == "hello"
        assert extra["count"] == 5

    def test_log_completion(self, mock_config: Config, mock_memory_manager: MemoryManager, mock_tool_logger: MagicMock) -> None:
        """log_completion logs success with tool name."""
        tool = ConcreteTool(
            config=mock_config,
            memory_manager=mock_memory_manager,
            logger=mock_tool_logger,
        )
        tool.log_completion("remove", deleted=3)

        call_args = mock_tool_logger.info.call_args
        assert "remove" in call_args.args[0]
        assert "completed successfully" in call_args.args[0]
        assert call_args.kwargs["extra"]["deleted"] == 3

    def test_log_error(self, mock_config: Config, mock_memory_manager: MemoryManager, mock_tool_logger: MagicMock) -> None:
        """log_error logs error with tool name and exception message."""
        tool = ConcreteTool(
            config=mock_config,
            memory_manager=mock_memory_manager,
            logger=mock_tool_logger,
        )
        err = RuntimeError("disk full")
        tool.log_error("add", err, count=1)

        call_args = mock_tool_logger.error.call_args
        assert "add" in call_args.args[0]
        assert "disk full" in call_args.args[0]
        assert call_args.kwargs["extra"]["error"] == "disk full"
        assert call_args.kwargs["extra"]["count"] == 1

    def test_log_error_preserves_error_string(
        self, mock_config: Config, mock_memory_manager: MemoryManager, mock_tool_logger: MagicMock
    ) -> None:
        """log_error converts exception to string in extra dict."""
        tool = ConcreteTool(
            config=mock_config,
            memory_manager=mock_memory_manager,
            logger=mock_tool_logger,
        )
        err = ValueError("bad input")
        tool.log_error("search", err)

        extra = mock_tool_logger.error.call_args.kwargs["extra"]
        assert extra["error"] == "bad input"
        assert extra["tool"] == "search"


@pytest.mark.unit
class TestBaseToolAbstractMethods:
    """Tests for BaseTool abstract interface."""

    def test_get_name_returns_string(
        self, mock_config: Config, mock_memory_manager: MemoryManager, mock_tool_logger: MagicMock
    ) -> None:
        """Concrete tool get_name returns expected name."""
        tool = ConcreteTool(
            config=mock_config,
            memory_manager=mock_memory_manager,
            logger=mock_tool_logger,
        )
        assert tool.get_name() == "test_tool"

    def test_get_handler_returns_callable(
        self, mock_config: Config, mock_memory_manager: MemoryManager, mock_tool_logger: MagicMock
    ) -> None:
        """Concrete tool get_handler returns a callable."""
        tool = ConcreteTool(
            config=mock_config,
            memory_manager=mock_memory_manager,
            logger=mock_tool_logger,
        )
        handler = tool.get_handler()
        assert callable(handler)

    def test_get_instruction_snippet_returns_string(
        self, mock_config: Config, mock_memory_manager: MemoryManager, mock_tool_logger: MagicMock
    ) -> None:
        """Concrete tool get_instruction_snippet returns formatted string."""
        tool = ConcreteTool(
            config=mock_config,
            memory_manager=mock_memory_manager,
            logger=mock_tool_logger,
        )
        snippet = tool.get_instruction_snippet()
        assert "test_tool" in snippet
        assert snippet.startswith("    •")
