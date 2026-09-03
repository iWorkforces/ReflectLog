"""Structured logging utilities for ReflectLog Server."""

from contextlib import contextmanager
import logging
from typing import TYPE_CHECKING, Any, final, override

from reflectlog.core.logging import IStructuredLogger

if TYPE_CHECKING:
    from collections.abc import Generator


@final
class StructuredLogger(IStructuredLogger):
    """Wrapper for structured logging with consistent formatting."""

    def __init__(
        self, logger: logging.Logger, default_extra: dict[str, Any] | None = None
    ) -> None:
        """Initialize structured logger.

        Args:
            logger: The underlying Python logger.
            default_extra: Default extra fields to include in all log messages.
        """
        self.logger = logger
        self.default_extra = default_extra or {}

    def _merge_extra(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Merge default extra fields with provided extras.

        Performance optimized: avoids dict copying when possible.
        Automatically redacts sensitive patterns in extra fields.
        """
        if extra is None:
            # Return copy to prevent accidental mutation of default
            return self.default_extra.copy() if self.default_extra else {}

        # Merge and redact sensitive data
        merged = {**self.default_extra, **extra}
        return self._redact_sensitive_data(merged)

    def _redact_sensitive_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Redact sensitive patterns in log data.

        Integrates with redact_dict_secrets to automatically redact
        API keys, passwords, and other sensitive patterns.

        Args:
            data: Dictionary potentially containing sensitive data.

        Returns:
            Dictionary with sensitive values redacted.
        """
        from .security import redact_dict_secrets, sanitize_for_logging

        redacted = redact_dict_secrets(data)
        sanitized: dict[str, Any] = {}
        for key, value in redacted.items():
            if isinstance(value, str):
                sanitized[key] = sanitize_for_logging(value)
            else:
                sanitized[key] = value
        return sanitized

    @override
    def is_enabled_for(self, level: int) -> bool:
        """Check if logger is enabled for the given level.

        Use this to guard expensive logging operations.

        Example:
            if logger.is_enabled_for(logging.DEBUG):
                logger.debug("Expensive: %s", expensive_computation())
        """
        return self.logger.isEnabledFor(level)

    @override
    def info(
        self,
        message: str,
        extra: dict[str, Any] | None = None,
        exc_info: bool = False,
    ) -> None:
        """Log info message with structured data."""
        from .security import sanitize_for_logging

        self.logger.info(
            sanitize_for_logging(message),
            extra=self._merge_extra(extra),
            exc_info=exc_info,
        )

    @override
    def error(
        self,
        message: str,
        extra: dict[str, Any] | None = None,
        exc_info: bool = False,
    ) -> None:
        """Log error message with structured data."""
        from .security import sanitize_for_logging

        self.logger.error(
            sanitize_for_logging(message),
            extra=self._merge_extra(extra),
            exc_info=exc_info,
        )

    @override
    def warning(
        self,
        message: str,
        extra: dict[str, Any] | None = None,
        exc_info: bool = False,
    ) -> None:
        """Log warning message with structured data."""
        from .security import sanitize_for_logging

        self.logger.warning(
            sanitize_for_logging(message),
            extra=self._merge_extra(extra),
            exc_info=exc_info,
        )

    @override
    def debug(
        self,
        message: str,
        extra: dict[str, Any] | None = None,
        exc_info: bool = False,
    ) -> None:
        """Log debug message with structured data."""
        from .security import sanitize_for_logging

        self.logger.debug(
            sanitize_for_logging(message),
            extra=self._merge_extra(extra),
            exc_info=exc_info,
        )

    @contextmanager
    def operation(
        self, operation_name: str, **kwargs: Any
    ) -> Generator[dict[str, Any]]:
        """Context manager for logging operation start and completion.

        Args:
            operation_name: Name of the operation being performed.
            **kwargs: Additional fields to include in log messages.

        Yields:
            Dict that can be updated with operation results.

        Example:
            with logger.operation("add_messages", count=5) as ctx:
                # Do operation
                ctx["added"] = 3
            # Automatically logs completion with results
        """
        context = {"operation": operation_name, **kwargs}

        self.info(f"Starting {operation_name}", extra=context)

        try:
            yield context
            self.info(f"Completed {operation_name}", extra=context)
        except Exception as e:
            context["error"] = str(e)
            context["error_type"] = type(e).__name__
            self.error(f"Failed {operation_name}: {e}", extra=context, exc_info=True)
            # Re-raise with operation context in the exception message
            raise RuntimeError(f"Operation '{operation_name}' failed: {e}") from e


def create_logger(
    name: str, workspace_id: str, log_level: str = "INFO"
) -> StructuredLogger:
    """Create a structured logger with default configuration.

    This function wraps fastmcp's logger to provide consistent structured logging
    across the application. The wrapper adds project context to all log messages.

    Args:
        name: Logger name (usually __name__).
        workspace_id: Workspace identifier to include in all logs.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured StructuredLogger instance.

    Note:
        fastmcp Logger Integration: This function uses fastmcp's get_logger()
        and sets the log level directly on it. This approach works with the current
        fastmcp version but may conflict if fastmcp changes its logger management
        in future versions. The setLevel() call ensures our configuration is
        respected, but fastmcp may override this in some scenarios.
    """
    # Use fastmcp's logger which is already properly configured with handlers
    from fastmcp.utilities.logging import get_logger

    logger = get_logger(name)
    # Set log level: fastmcp logger may have its own level management
    # We set it here to ensure our configuration is respected
    logger.setLevel(logging.getLevelNamesMapping()[log_level.upper()])

    return StructuredLogger(logger, default_extra={"workspace_id": workspace_id})


def format_fusion_score_status(score: float, threshold: float) -> tuple[str, str]:
    """Format fusion score with status and interpretation.

    Args:
        score: The normalized RRF fusion score (0-1 range).
        threshold: The minimum fusion score threshold.

    Returns:
        Tuple of (status_indicator, interpretation).
    """
    status = "KEEP" if score >= threshold else "FILTER"

    if score >= 0.8:
        interpretation = "High fusion confidence"
    elif score >= 0.6:
        interpretation = "Moderate fusion confidence"
    elif score >= 0.4:
        interpretation = "Low fusion confidence"
    else:
        interpretation = "Minimal fusion overlap"

    return status, interpretation
