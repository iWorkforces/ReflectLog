"""Structured logging utilities for OpenMemoriesMCP MCP Server."""

import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional


class StructuredLogger:
    """Wrapper for structured logging with consistent formatting."""

    def __init__(
        self, logger: logging.Logger, default_extra: Optional[Dict[str, Any]] = None
    ):
        """Initialize structured logger.

        Args:
            logger: The underlying Python logger.
            default_extra: Default extra fields to include in all log messages.
        """
        self.logger = logger
        self.default_extra = default_extra or {}

    def _merge_extra(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Merge default extra fields with provided extras.

        Performance optimized: avoids dict copying when possible.
        """
        if extra is None:
            # Return reference directly - logging doesn't modify extra dict
            return self.default_extra
        # Use unpacking for efficient merge (faster than copy + update)
        return {**self.default_extra, **extra}

    def isEnabledFor(self, level: int) -> bool:
        """Check if logger is enabled for the given level.

        Use this to guard expensive logging operations.

        Example:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Expensive: %s", expensive_computation())
        """
        return self.logger.isEnabledFor(level)

    def info(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = False,
    ) -> None:
        """Log info message with structured data."""
        self.logger.info(message, extra=self._merge_extra(extra), exc_info=exc_info)

    def error(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = False,
    ) -> None:
        """Log error message with structured data."""
        self.logger.error(message, extra=self._merge_extra(extra), exc_info=exc_info)

    def warning(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = False,
    ) -> None:
        """Log warning message with structured data."""
        self.logger.warning(message, extra=self._merge_extra(extra), exc_info=exc_info)

    def debug(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = False,
    ) -> None:
        """Log debug message with structured data."""
        self.logger.debug(message, extra=self._merge_extra(extra), exc_info=exc_info)

    @contextmanager
    def operation(self, operation_name: str, **kwargs: Any):
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
            self.error(f"Failed {operation_name}: {e}", extra=context)
            raise


def create_logger(
    name: str, project_id: str, log_level: str = "INFO"
) -> StructuredLogger:
    """Create a structured logger with default configuration.

    Args:
        name: Logger name (usually __name__).
        project_id: Project identifier to include in all logs.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured StructuredLogger instance.
    """
    # Use fastmcp's logger which is already properly configured with handlers
    from fastmcp.utilities.logging import get_logger

    logger = get_logger(name)
    # Note: fastmcp logger may have its own level management, but we respect the config
    logger.setLevel(getattr(logging, log_level.upper()))

    return StructuredLogger(logger, default_extra={"project_id": project_id})


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
