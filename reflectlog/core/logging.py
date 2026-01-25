"""Logging protocols for ReflectLogMCP.

This module defines protocols for logging and observability. These
abstractions enable different logging implementations while providing
a consistent interface for structured logging.
"""

from typing import Protocol, runtime_checkable, Optional, Any
from enum import Enum
from datetime import datetime


class LogLevel(Enum):
    """Log level enumeration matching Python logging levels."""

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@runtime_checkable
class ILogSink(Protocol):
    """Protocol for log destinations.

    This protocol defines the interface for log sinks that receive
    and process log records. Implementations can write to files,
    network services, or any other destination.
    """

    def emit(
        self,
        level: LogLevel,
        message: str,
        timestamp: datetime,
        **kwargs: Any,
    ) -> None:
        """Emit a log record.

        Args:
            level: Log level.
            message: Log message.
            timestamp: Event timestamp.
            **kwargs: Additional structured data.
        """
        ...

    def close(self) -> None:
        """Release sink resources."""
        ...


@runtime_checkable
class ILoggingService(Protocol):
    """Protocol for logging services.

    This protocol defines the interface for structured logging services
    that provide type-safe logging with consistent formatting and
    metadata handling.
    """

    def debug(
        self,
        message: str,
        **kwargs: Any,
    ) -> None:
        """Log at DEBUG level.

        Args:
            message: Log message.
            **kwargs: Additional context.
        """
        ...

    def info(
        self,
        message: str,
        **kwargs: Any,
    ) -> None:
        """Log at INFO level.

        Args:
            message: Log message.
            **kwargs: Additional context.
        """
        ...

    def warning(
        self,
        message: str,
        **kwargs: Any,
    ) -> None:
        """Log at WARNING level.

        Args:
            message: Log message.
            **kwargs: Additional context.
        """
        ...

    def error(
        self,
        message: str,
        **kwargs: Any,
    ) -> None:
        """Log at ERROR level.

        Args:
            message: Log message.
            **kwargs: Additional context.
        """
        ...

    def critical(
        self,
        message: str,
        **kwargs: Any,
    ) -> None:
        """Log at CRITICAL level.

        Args:
            message: Log message.
            **kwargs: Additional context.
        """
        ...

    def log(
        self,
        level: LogLevel,
        message: str,
        **kwargs: Any,
    ) -> None:
        """Log at specified level.

        Args:
            level: Log level.
            message: Log message.
            **kwargs: Additional context.
        """
        ...

    def bind(self, **kwargs: Any) -> "ILoggingService":
        """Create a bound logger with preset context.

        Args:
            **kwargs: Context to bind.

        Returns:
            New logger with bound context.
        """
        ...

    @property
    def level(self) -> LogLevel:
        """Current log level."""
        ...

    def close(self) -> None:
        """Release logging resources."""
        ...
