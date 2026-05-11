"""Application-layer type definitions for ReflectLogMCP Server.

Types shared across architectural boundaries (MemoryRecord, Embeddings,
ISemanticSearchEngine, IArchiveMemoryStore) live in reflectlog.core.types.
This module contains application-specific types only.
"""

from typing import (
    Protocol,
    TypedDict,
    runtime_checkable,
)

from reflectlog.core.types import MemoryRecord

# Timestamp handling protocol
# All timestamps in ReflectLogMCP use ISO 8601 format (UTC timezone)
# Format: "YYYY-MM-DDTHH:MM:SS.ffffffZ" or "YYYY-MM-DDTHH:MM:SSZ"
# Example: "2025-01-24T10:30:45.123456Z"
#
# Timestamp flow:
# 1. MemoryStore stores created_at as ISO string when adding memories
# 2. USearchEngine.search() returns (memory, score, created_at) tuples
# 3. MemoryManager builds timestamp_map: Dict[str, str] for rerankers
# 4. Rerankers use timestamps for recency decay calculations
#
# Empty string ("") is used for backward compatibility with data stored
# before timestamp tracking was implemented.


# Memory search result types
type SearchResult = dict[str, list[MemoryRecord]]
type MemoryList = list[str]

# Tool result types
type ToolResult = None | MemoryList


# Extra logging context
class LogContext(TypedDict, total=False):
    operation: str
    project_id: str
    tool: str
    error: str
    error_type: str


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
        enable_llm_infer: Whether to enable LLM-based memory inference.
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
