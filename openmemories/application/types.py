"""Type definitions for CCMemoriesMCP Server."""

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    TypeAlias,
    runtime_checkable,
)

if TYPE_CHECKING:
    from ccmemories.infrastructure.message_store import MessageStore

# Memory operation types
MemoryRecord: TypeAlias = Dict[str, Any]
SearchResult: TypeAlias = Dict[str, List[MemoryRecord]]
MessageList: TypeAlias = List[str]

# Tool result types
ToolResult: TypeAlias = None | MessageList

# Extra logging context
LogContext: TypeAlias = Dict[str, Any]


# Clean Architecture: Application layer defines interfaces
# Infrastructure layer implements them via structural subtyping (duck typing)
@runtime_checkable
class ISemanticSearchConfig(Protocol):
    """Interface for semantic search engine configuration.

    Following Clean Architecture principles, this protocol defines the contract
    that the application layer needs from semantic search configuration. The
    infrastructure layer's SemanticConfig implements this via structural subtyping.

    The @runtime_checkable decorator enables isinstance() checks at runtime.
    Type checkers (mypy) verify structural compatibility at static analysis time.

    This allows the application layer to depend on an abstraction (this protocol)
    rather than a concrete implementation (SemanticConfig), following the
    Dependency Inversion Principle (DIP) from SOLID.

    Attributes:
        project_id: Unique project identifier for filtering.
        index_path: Path to the semantic index storage.
        embedding_model: Embedding model identifier.
        embedding_dims: Embedding vector dimensions.
        embedder_provider: Embedding provider name.
        enable_llm_infer: Whether to enable LLM-based message inference.
        openrouter_api_key: API key for embeddings.
        openrouter_base_url: API base URL.
        qwen_embedding_dims: Qwen embedding dimensions.
    """

    @property
    def project_id(self) -> str: ...

    @property
    def index_path(self) -> str: ...

    @property
    def embedding_model(self) -> str: ...

    @property
    def embedding_dims(self) -> int: ...

    @property
    def embedder_provider(self) -> str: ...

    @property
    def enable_llm_infer(self) -> bool: ...

    @property
    def openrouter_api_key(self) -> str: ...

    @property
    def openrouter_base_url(self) -> str: ...

    @property
    def qwen_embedding_dims(self) -> int: ...


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
        def search_memories(engine: ISemanticSearchEngine, query: str) -> List[Tuple[str, float]]:
            return engine.search(query=query, project_id="project", limit=5)
        ```
    """

    @property
    def name(self) -> str:
        """Engine name for identification."""
        ...

    def add(self, project_id: str, message: str, infer: bool) -> None:
        """Add a message to the semantic index.

        Args:
            project_id: Project identifier for filtering.
            message: Message content to index.
            infer: Whether to enable LLM-based message inference.

        Raises:
            RuntimeError: If add operation fails.
        """
        ...

    def search(
        self,
        query: str,
        project_id: str,
        limit: int,
    ) -> List[tuple[str, float]]:
        """Execute semantic search.

        Args:
            query: Search query string.
            project_id: Filter results by project_id.
            limit: Maximum number of results.

        Returns:
            List of (message, score) tuples sorted by relevance.

        Raises:
            RuntimeError: If search operation fails.
        """
        ...

    def get_all(self, project_id: str) -> List[str]:
        """Retrieve all stored messages for a project.

        Args:
            project_id: Project identifier for filtering.

        Returns:
            List of all messages stored for the project.

        Raises:
            RuntimeError: If retrieval operation fails.
        """
        ...

    def delete(self, memory_id: str) -> None:
        """Delete a memory entry by its ID.

        Args:
            memory_id: The ID of the memory to delete.

        Raises:
            RuntimeError: If deletion fails.
        """
        ...

    def commit(self) -> None:
        """Commit pending changes to the index.

        Note: May be a no-op for engines that commit automatically.
        """
        ...

    def ensure_initialized(self) -> None:
        """Ensure the engine is fully initialized (thread-safe).

        This method forces any lazy initialization to complete before returning.
        Use this before parallel operations to prevent race conditions in
        lazy initialization.

        Note: May be a no-op for eagerly initialized engines.
        """
        ...

    def get_id_by_message(self, project_id: str, message: str) -> Optional[int]:
        """Get the ID of a message by its content.

        Args:
            project_id: Project identifier.
            message: Message text to look up.

        Returns:
            The message ID if found, None otherwise.
        """
        ...

    def close(self) -> None:
        """Close resources and cleanup.

        This method should be called during graceful shutdown to ensure
        all resources are properly released (e.g., database connections,
        file handles).

        Note: May be a no-op for engines without persistent resources.
        """
        ...

    @property
    def message_store(self) -> "MessageStore":
        """Get the underlying message store for archive operations.

        Returns:
            The MessageStore instance used for message text storage.
        """
        ...
