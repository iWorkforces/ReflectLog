__version__ = "0.1.7"

# Export main components for programmatic access
from reflectlog.server import main

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

__all__ = [
    "main",
    "__version__",
    "ReflectLogError",
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
