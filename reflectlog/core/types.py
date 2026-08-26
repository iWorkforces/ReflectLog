"""Shared type definitions for ReflectLog.

This module defines types that are shared across architectural boundaries
(core, application, infrastructure). Moving these types here breaks the
circular dependency where core/ imported from application/.

Types defined here:
    MemoryRecord: TypedDict for memory entries.
    Embeddings: Protocol for embedding providers.
    IArchiveMemoryStore: Protocol for archive operations.
    ISemanticSearchEngine: Protocol for semantic search engines.
"""

from typing import (
    Protocol,
    TypedDict,
    runtime_checkable,
)


class MemoryRecord(TypedDict, total=False):
    id: str
    memory: str
    score: float
    project_id: str
    content: str
    created_at: str
    metadata: dict[str, object]


class IArchiveMemoryStore(Protocol):
    def archive(
        self,
        memory_id: int,
        project_id: str,
        content: str,
        replaced_by: str,
        reason: str,
        confidence: float,
    ) -> int | None: ...


@runtime_checkable
class Embeddings(Protocol):
    """Interface for embedding providers used by semantic search engines.

    This protocol defines the contract for embedding text into vector representations.
    Implementations can use various backends (OpenAI, Langchain/Qwen, etc.)
    as long as they conform to this interface.

    All embedding methods should return lists of floats representing the
    embedding vector for the given text(s).
    """

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string synchronously.

        Args:
            text: The query text to embed.

        Returns:
            A list of floats representing the embedding vector.
            The length of the list depends on the embedding model dimensions.

        Raises:
            RuntimeError: If the embedding operation fails.
        """
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents synchronously.

        Args:
            texts: List of document texts to embed.

        Returns:
            A list of embedding vectors (one per input text).
            Each vector is a list of floats with length equal to the
            embedding model dimensions.

        Raises:
            RuntimeError: If the embedding operation fails.
        """
        ...

    async def aembed_query(self, text: str) -> list[float]:
        """Embed a single query string asynchronously.

        Args:
            text: The query text to embed.

        Returns:
            A list of floats representing the embedding vector.
            The length of the list depends on the embedding model dimensions.

        Raises:
            RuntimeError: If the embedding operation fails.
        """
        ...

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents asynchronously.

        Args:
            texts: List of document texts to embed.

        Returns:
            A list of embedding vectors (one per input text).
            Each vector is a list of floats with length equal to the
            embedding model dimensions.

        Raises:
            RuntimeError: If the embedding operation fails.
        """
        ...


class ISemanticSearchEngine(Protocol):
    """Interface for semantic search engine operations.

    Following Clean Architecture principles, this protocol defines the contract
    that the application layer needs from a semantic search engine. The
    infrastructure layer's USearchEngine explicitly inherits from this protocol
    using a custom metaclass to resolve Pydantic BaseModel + Protocol conflicts.

    Type checkers (mypy) verify structural compatibility at static analysis time.

    This abstraction allows:
    - Application layer to depend on capabilities, not implementations
    - Easy mocking in unit tests
    - Swapping implementations without changing application code
    - Following Interface Segregation Principle (ISP) from SOLID

    Example:
        ```python
        # Application layer code depends on the protocol
        def search_memories(
            engine: ISemanticSearchEngine, query: str
        ) -> list[tuple[str, float]]:
            return engine.search(query=query, project_id="project", limit=5)
        ```
    """

    @property
    def name(self) -> str:
        """Engine name for identification."""
        ...

    def add(self, project_id: str, content: str, infer: bool) -> None:
        """Add a memory to the semantic index.

        Args:
            project_id: Project identifier for filtering.
            content: Memory content to index.
            infer: Whether to enable LLM-based memory inference.

        Returns:
            None. The memory is stored in the index if successful.

        Raises:
            RuntimeError: If add operation fails.
        """
        ...

    def add_batch(self, project_id: str, contents: list[str], infer: bool) -> list[str]:
        """Add multiple memories to the semantic index in a single batch.

        Args:
            project_id: Project identifier for filtering.
            contents: List of memory texts to index.
            infer: Whether to enable LLM-based memory inference.

        Returns:
            List of memories successfully added (duplicates skipped).
        """
        ...

    def search(
        self,
        query: str,
        project_id: str,
        limit: int,
    ) -> list[tuple[str, float, str]]:
        """Execute semantic search.

        Args:
            query: Search query string.
            project_id: Filter results by project_id.
            limit: Maximum number of results.

        Returns:
            List of (memory, score, created_at) tuples sorted by relevance.

            - memory (str): The memory text
            - score (float): Similarity score (higher = more relevant)
            - created_at (str): ISO 8601 timestamp in UTC when the memory was stored.
              May be empty string ("") for backward compatibility with older data
              that was stored before timestamp tracking was implemented.

            Timestamp Format:
                ISO 8601 format: "YYYY-MM-DDTHH:MM:SS.ffffffZ"
                Example: "2025-01-24T10:30:45.123456Z"

            The timestamp is used by rerankers for recency-aware scoring when
            ENABLE_RECENCY_BOOST is enabled. Newer memories may be preferred
            over older memories when they contain contradictory information.

        Raises:
            RuntimeError: If search operation fails.
        """
        ...

    def get_all(self, project_id: str) -> list[str]:
        """Retrieve all stored memories for a project.

        Args:
            project_id: Project identifier for filtering.

        Returns:
            List of all memories stored for the project.

        Raises:
            RuntimeError: If retrieval operation fails.
        """
        ...

    def delete(self, memory_id: str) -> None:
        """Delete a memory entry by its ID.

        Args:
            memory_id: The ID of the memory to delete.

        Returns:
            None. The memory is removed from the index if it exists.

        Raises:
            RuntimeError: If deletion fails.
        """
        ...

    def commit(self) -> None:
        """Commit pending changes to the index.

        This ensures any buffered writes are persisted to storage.

        Note: May be a no-op for engines that commit automatically.

        Returns:
            None. Changes are persisted if successful.

        Raises:
            RuntimeError: If commit operation fails.
        """
        ...

    def ensure_initialized(self) -> None:
        """Ensure the engine is fully initialized (thread-safe).

        This method forces any lazy initialization to complete before returning.
        Use this before parallel operations to prevent race conditions in
        lazy initialization.

        Note: May be a no-op for eagerly initialized engines.

        Returns:
            None. The engine is guaranteed to be initialized after this call.

        Raises:
            RuntimeError: If initialization fails.
        """
        ...

    def get_id_by_content(self, project_id: str, content: str) -> int | None:
        """Get the ID of a memory by its content.

        Args:
            project_id: Project identifier.
            content: Memory text to look up.

        Returns:
            The memory ID if found, None otherwise.
        """
        ...

    def close(self) -> None:
        """Close resources and cleanup.

        This method should be called during graceful shutdown to ensure
        all resources are properly released (e.g., database connections,
        file handles).

        Note: May be a no-op for engines without persistent resources.

        Returns:
            None. Resources are released and the engine should not be used
            after calling this method.

        Raises:
            RuntimeError: If cleanup operation fails.
        """
        ...

    @property
    def memory_store(self) -> IArchiveMemoryStore:
        """Get the underlying memory store for archive operations.

        Returns:
            The MemoryStore instance used for memory text storage.
        """
        ...
