"""Custom exception hierarchy for CCMemoriesMCP.

This module defines a structured exception hierarchy that allows callers
to distinguish between different types of errors and handle them appropriately.

Hierarchy:
    CCMemoriesError (base)
    ├── ConfigurationError - Invalid or missing configuration
    ├── ValidationError - Input validation failures
    ├── InitializationError - Engine/client initialization failures
    ├── StorageError - Storage operation failures
    │   ├── DuplicateError - Duplicate entry detected
    │   └── InconsistentStateError - Dual-engine state mismatch
    ├── SearchError - Search operation failures
    └── EmbeddingError - Embedding generation failures
"""


class CCMemoriesError(Exception):
    """Base exception for all CCMemoriesMCP errors.

    All custom exceptions in this project should inherit from this class
    to enable broad exception catching when needed.

    Example:
        try:
            memory_manager.add_messages(messages)
        except CCMemoriesError as e:
            # Handle any CCMemoriesMCP error
            logger.error(f"Operation failed: {e}")
    """

    pass


class ConfigurationError(CCMemoriesError):
    """Raised when configuration is invalid or missing.

    This includes missing required environment variables,
    invalid configuration values, and configuration validation failures.

    Example:
        if not project_id:
            raise ConfigurationError("PROJECT_ID environment variable is required")
    """

    pass


class ValidationError(CCMemoriesError):
    """Raised when input validation fails.

    This includes message length validation, format validation,
    and any other input-related validation failures.

    Example:
        if len(message) > max_length:
            raise ValidationError(f"Message exceeds maximum length of {max_length}")
    """

    pass


class InitializationError(CCMemoriesError):
    """Raised when engine or client initialization fails.

    This includes failures to initialize search engines, embedding clients,
    database connections, and other infrastructure components.

    Example:
        if self._index is None:
            raise InitializationError("Tantivy index not initialized")
    """

    pass


class StorageError(CCMemoriesError):
    """Raised when a storage operation fails.

    This is the base class for storage-related errors including
    insert, update, delete, and persistence failures.

    Example:
        try:
            self.message_store.insert(project_id, message)
        except Exception as e:
            raise StorageError(f"Failed to store message: {e}") from e
    """

    pass


class DuplicateError(StorageError):
    """Raised when a duplicate entry is detected.

    This is raised when attempting to insert a message that already
    exists in the storage (when deduplication is enabled).

    Example:
        if self.message_store.exists(project_id, message):
            raise DuplicateError(f"Message already exists: {message[:50]}...")
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


class SearchError(CCMemoriesError):
    """Raised when a search operation fails.

    This includes semantic search failures, full-text search failures,
    and hybrid search fusion failures.

    Example:
        try:
            results = self._semantic_engine.search(query, project_id, limit)
        except Exception as e:
            raise SearchError(f"Semantic search failed: {e}") from e
    """

    pass


class EmbeddingError(CCMemoriesError):
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


class RerankerError(CCMemoriesError):
    """Raised when LLM reranking fails.

    This is raised when the LLM reranker encounters an unrecoverable error.
    Note: Individual document scoring failures fall back gracefully to fusion scores;
    this exception is for systemic failures.

    Example:
        if self._client is None:
            raise RerankerError("LLM reranker client is not initialized")
    """

    pass
