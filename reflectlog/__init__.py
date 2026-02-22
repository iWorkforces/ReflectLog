"""ReflectLogMCP - An Agentic Memory Layer For Coding Agents.

This module provides the public API for ReflectLogMCP.
"""

from typing import Any

# Export exception hierarchy for error handling
from reflectlog.application.exceptions import (
    ConfigurationError,
    DuplicateError,
    EmbeddingError,
    InconsistentStateError,
    InitializationError,
    ReflectLogError,
    RerankerError,
    SearchError,
    StorageError,
    ValidationError,
)
from reflectlog.version import __version__


# Lazy import for main to avoid circular imports
# main is only needed when the module is used as a server entry point
# Using __getattr__ for lazy loading
def __getattr__(name: str) -> Any:
    """Lazy import for server.main to avoid circular imports."""
    if name == "main":
        from reflectlog.server import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ConfigurationError",
    "DuplicateError",
    "EmbeddingError",
    "InconsistentStateError",
    "InitializationError",
    "ReflectLogError",
    "RerankerError",
    "SearchError",
    "StorageError",
    "ValidationError",
    "__version__",
]
