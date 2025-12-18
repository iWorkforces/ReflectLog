__version__ = "0.1.0"

# Export main components for programmatic access
from openmemories.server import main

# Export exception hierarchy for error handling
from openmemories.application.exceptions import (
    ConfigurationError,
    DuplicateError,
    EmbeddingError,
    InconsistentStateError,
    InitializationError,
    OpenMemoriesError,
    RerankerError,
    SearchError,
    StorageError,
    ValidationError,
)

__all__ = [
    "main",
    "__version__",
    "OpenMemoriesError",
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
