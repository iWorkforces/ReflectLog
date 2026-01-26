"""Configuration protocols for ReflectLogMCP.

This module defines protocols that abstract configuration sources from
configuration consumers. Components depend on these protocols rather than
concrete configuration classes, enabling:

- Configuration from environment variables, files, or remote services
- Testing with fixture configurations
- Runtime configuration updates without component modification

Example:
    def search_memories(
        query: str,
        config: ISearchConfig,
        store: IMemoryStore,
    ) -> list[str]:
        limit = config.search_limit
        ...
"""

from typing import Protocol, runtime_checkable, Literal


@runtime_checkable
class IServerConfig(Protocol):
    """Protocol for server-level configuration."""

    @property
    def transport(self) -> Literal["stdio", "http", "sse", "streamable-http"]:
        """Transport mode for MCP server."""
        ...

    @property
    def host(self) -> str:
        """Server host for network transports."""
        ...

    @property
    def port(self) -> int:
        """Server port for network transports."""
        ...

    @property
    def path(self) -> str:
        """Server path for network transports."""
        ...

    @property
    def log_level(self) -> str:
        """Logging level: DEBUG, INFO, WARNING, ERROR."""
        ...

    @property
    def project_id(self) -> str:
        """Unique project identifier."""
        ...


@runtime_checkable
class ISearchConfig(Protocol):
    """Protocol for search-related configuration."""

    @property
    def search_limit(self) -> int:
        """Maximum number of results to return."""
        ...

    @property
    def enable_hybrid_search(self) -> bool:
        """Enable combining semantic and full-text search."""
        ...

    @property
    def enable_rrf_fusion(self) -> bool:
        """Enable Reciprocal Rank Fusion for result ranking."""
        ...

    @property
    def fusion_rrf_k(self) -> int:
        """RRF constant k for fusion ranking."""
        ...

    @property
    def fusion_threshold(self) -> float:
        """Minimum fusion score to include result."""
        ...

    @property
    def reranker_engine(self) -> Literal["llm", "cross_encoder", "none"]:
        """Reranking engine type."""
        ...

    @property
    def search_score_threshold(self) -> float:
        """Minimum relevance score for LLM reranking."""
        ...

    @property
    def enable_recency_boost(self) -> bool:
        """Include memory age in reranking."""
        ...

    @property
    def recency_decay_rate(self) -> float:
        """Exponential decay rate per hour."""
        ...


@runtime_checkable
class IStorageConfig(Protocol):
    """Protocol for storage-related configuration."""

    @property
    def storage_path(self) -> str:
        """Base path for index storage."""
        ...

    @property
    def usearch_index_path(self) -> str:
        """Path to USearch index files."""
        ...

    @property
    def tantivy_index_path(self) -> str:
        """Path to Tantivy index files."""
        ...

    @property
    def embedding_dims(self) -> int:
        """Embedding vector dimensions."""
        ...

    @property
    def metric(self) -> Literal["cosine", "euclidean", "inner_product"]:
        """Similarity metric for vector search."""
        ...


@runtime_checkable
class IRerankerConfig(Protocol):
    """Protocol for reranker configuration."""

    @property
    def llm_model(self) -> str:
        """LLM model for reranking."""
        ...

    @property
    def llm_api_base_url(self) -> str:
        """API base URL for LLM provider."""
        ...

    @property
    def cross_encoder_model(self) -> str:
        """Cross-encoder model name."""
        ...

    @property
    def cross_encoder_device(self) -> str:
        """Device for cross-encoder: cpu, cuda, mps."""
        ...

    @property
    def reranker_batch_normalize(self) -> bool:
        """Enable batch normalization for reranker scores."""
        ...


@runtime_checkable
class IEmbedderConfig(Protocol):
    """Protocol for embedding provider configuration."""

    @property
    def embedding_model(self) -> str:
        """Embedding model name."""
        ...

    @property
    def embedder_provider(self) -> str:
        """Embedder provider: langchain or openai."""
        ...

    @property
    def qwen_embedding_dims(self) -> int:
        """Qwen embedding dimensions."""
        ...

    @property
    def embedding_batch_size(self) -> int:
        """Batch size for embedding operations."""
        ...

    @property
    def embedding_max_concurrent_batches(self) -> int:
        """Maximum concurrent embedding batches."""
        ...

    @property
    def embedding_cache_enabled(self) -> bool:
        """Enable embedding cache."""
        ...

    @property
    def embedding_cache_size(self) -> int:
        """Maximum cache entries."""
        ...


@runtime_checkable
class IReplacementConfig(Protocol):
    """Protocol for smart replacement configuration."""

    @property
    def enable_smart_replace(self) -> bool:
        """Enable smart memory replacement detection."""
        ...

    @property
    def smart_replace_threshold(self) -> float:
        """Minimum confidence for replacement detection."""
        ...

    @property
    def smart_replace_min_similarity(self) -> float:
        """Minimum embedding similarity for candidates."""
        ...

    @property
    def smart_replace_candidate_limit(self) -> int:
        """Maximum candidates to check."""
        ...


@runtime_checkable
class IAppConfig(
    IServerConfig,
    ISearchConfig,
    IStorageConfig,
    IRerankerConfig,
    IEmbedderConfig,
    IReplacementConfig,
    Protocol,
):
    """Combined protocol for all application configuration.

    Components can depend on this protocol to access all configuration
    without needing to inject multiple configuration objects.
    """
