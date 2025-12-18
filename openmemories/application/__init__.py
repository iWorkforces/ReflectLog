"""CCMemoriesMCP application layer.

This package contains the core business logic including:
- MCP server orchestration
- Memory management with hybrid search
- MCP tool implementations
- Configuration management
- Custom exception hierarchy
"""

from ccmemories.application.exceptions import (
    ConfigurationError,
    DuplicateError,
    EmbeddingError,
    InconsistentStateError,
    InitializationError,
    CCMemoriesError,
    RerankerError,
    SearchError,
    StorageError,
    ValidationError,
)

__all__ = [
    "CCMemoriesError",
    "ConfigurationError",
    "ValidationError",
    "InitializationError",
    "StorageError",
    "DuplicateError",
    "InconsistentStateError",
    "SearchError",
    "EmbeddingError",
    "RerankerError",
]
