"""Base class for MCP tools."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from ..config import Config
from ..memory import MemoryManager
from ..utils import StructuredLogger


class BaseTool(ABC):
    """Abstract base class for MCP tools."""

    def __init__(
        self, config: Config, memory_manager: MemoryManager, logger: StructuredLogger
    ):
        """Initialize tool with dependencies.

        Args:
            config: Application configuration.
            memory_manager: Memory management instance.
            logger: Structured logger instance.
        """
        self.config = config
        self.memory = memory_manager
        self.logger = logger

    @abstractmethod
    def get_name(self) -> str:
        """Get the tool name for registration.

        Returns:
            Tool name as string.
        """
        pass

    @abstractmethod
    def get_handler(self) -> Callable:
        """Get the tool handler function.

        Returns:
            Callable that implements the tool logic.
        """
        pass

    @abstractmethod
    def get_instruction_snippet(self) -> str:
        """Get the instruction snippet for MCP_INSTRUCTIONS.

        Each tool must provide a formatted documentation snippet that will be
        included in the dynamic MCP_INSTRUCTIONS when the tool is registered.

        Returns:
            Formatted string for this tool's documentation.
            Expected format:
                '    • signature\\n      description line 1\\n      description line 2'

        Example:
            '    • add(messages: list[str])\\n      Add messages with semantic embeddings.'
        """
        pass

    def log_invocation(self, tool_name: str, **kwargs: Any) -> None:
        """Log tool invocation with context.

        Args:
            tool_name: Name of the tool being invoked.
            **kwargs: Additional context to log.
        """
        self.logger.info(
            f"Tool '{tool_name}' invoked", extra={"tool": tool_name, **kwargs}
        )

    def log_completion(self, tool_name: str, **kwargs: Any) -> None:
        """Log tool completion with results.

        Args:
            tool_name: Name of the tool that completed.
            **kwargs: Results or additional context to log.
        """
        self.logger.info(
            f"Tool '{tool_name}' completed successfully",
            extra={"tool": tool_name, **kwargs},
        )

    def log_error(self, tool_name: str, error: Exception, **kwargs: Any) -> None:
        """Log tool error with context.

        Args:
            tool_name: Name of the tool that failed.
            error: The exception that occurred.
            **kwargs: Additional error context.
        """
        self.logger.error(
            f"Tool '{tool_name}' failed: {error}",
            extra={"tool": tool_name, "error": str(error), **kwargs},
        )
