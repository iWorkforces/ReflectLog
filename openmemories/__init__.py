__version__ = "0.1.0"

# Export main components for programmatic access
from ccmemories.server import main

# Export exception hierarchy for error handling
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
    "main",
    "__version__",
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
