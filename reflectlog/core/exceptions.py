"""Custom exception hierarchy for ReflectLog.

This module defines a structured exception hierarchy that allows callers
to distinguish between different types of errors and handle them appropriately.

Hierarchy:
    ReflectLogError (base)
    ├── ConfigurationError - Invalid or missing configuration
    ├── ValidationError - Input validation failures
    ├── InitializationError - Engine/client initialization failures
    ├── StorageError - Storage operation failures
    │   ├── DuplicateError - Duplicate entry detected
    │   └── InconsistentStateError - Dual-engine state mismatch
    ├── SearchError - Search operation failures
    └── EmbeddingError - Embedding generation failures
"""


class ReflectLogError(Exception):
    """Base exception for all ReflectLog errors.

    All custom exceptions in this project should inherit from this class
    to enable broad exception catching when needed.

    Example:
        try:
            memory_manager.add_memories(memories)
        except ReflectLogError as e:
            # Handle any ReflectLog error
            logger.error(f"Operation failed: {e}")
    """

    pass


class ConfigurationError(ReflectLogError):
    """Raised when configuration is invalid or missing.

    This includes missing required environment variables,
    invalid configuration values, and configuration validation failures.

    Example:
        if not workspace_id:
            raise ConfigurationError("WORKSPACE_ID environment variable is required")
    """

    pass


class ValidationError(ReflectLogError):
    """Raised when input validation fails.

    This includes memory length validation, format validation,
    and any other input-related validation failures.

    Example:
        if len(memory) > max_length:
            raise ValidationError(f"Memory exceeds maximum length of {max_length}")
    """

    pass


class InitializationError(ReflectLogError):
    """Raised when engine or client initialization fails.

    This includes failures to initialize search engines, embedding clients,
    database connections, and other infrastructure components.

    Example:
        if self._index is None:
            raise InitializationError("Tantivy index not initialized")
    """

    pass


class StorageError(ReflectLogError):
    """Raised when a storage operation fails.

    This is the base class for storage-related errors including
    insert, update, delete, and persistence failures.

    Example:
        try:
            self.message_store.insert(workspace_id, memory)
        except Exception as e:
            raise StorageError(f"Failed to store memory: {e}") from e
    """

    pass


class DuplicateError(StorageError):
    """Raised when a duplicate entry is detected.

    This is raised when attempting to insert a memory that already
    exists in the storage (when deduplication is enabled).

    Example:
        if self.message_store.exists(workspace_id, memory):
            raise DuplicateError(f"Memory already exists: {memory[:50]}...")
    """

    pass


class InconsistentStateError(StorageError):
    """Raised when dual-engine state becomes inconsistent.

    This is a critical error that occurs when an operation succeeds
    on one engine (e.g., USearch) but fails on another (e.g., Tantivy),
    leaving the system in an inconsistent state.

    Example:
        # USearch deletion succeeded but Tantivy deletion failed
        raise InconsistentStateError(
            "USearch deletion succeeded but Tantivy deletion failed: "
            "system is in inconsistent state"
        )
    """

    pass


class SearchError(ReflectLogError):
    """Raised when a search operation fails.

    This includes semantic search failures, full-text search failures,
    and hybrid search fusion failures.

    Example:
        try:
            results = self._semantic_engine.search(query, workspace_id, limit)
        except Exception as e:
            raise SearchError(f"Semantic search failed: {e}") from e
    """

    pass


class EmbeddingError(ReflectLogError):
    """Raised when embedding generation fails.

    This includes API errors, timeout errors, and other failures
    during the embedding generation process.

    Example:
        try:
            embedding = embedder.embed_query(text)
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embedding: {e}") from e
    """

    pass


class RerankerError(ReflectLogError):
    """Raised when reranking fails.

    This is raised when the reranker encounters an unrecoverable error.
    Individual document scoring failures fall back to fusion scores;
    this exception is for systemic failures.
    """

    pass
