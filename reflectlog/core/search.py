"""Search engine protocols for ReflectLog.

This module defines protocols for search operations. The search abstraction
enables different search backends—vector search, full-text search, hybrid
approaches—while presenting a consistent interface to the search pipeline.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


def _empty_metadata() -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class ISearchResult:
    """Result from a search operation.

    This class represents a single search result with its content,
    relevance score, and metadata.
    """

    content: str
    score: float
    memory_id: str
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)


@dataclass
class SearchContext:
    """Context for a search operation.

    This class carries all context needed by the search pipeline,
    including query, limits, and configuration options.
    """

    query: str
    limit: int
    overfetch_limit: int
    enable_hybrid_search: bool
    enable_rrf_fusion: bool
    reranker_engine: str
    project_id: str


@runtime_checkable
class ISearchBackend(Protocol):
    """Protocol for search engine implementations.

    This protocol defines the interface that all search backends must
    implement. Backends can be vector databases, full-text search engines,
    hybrid systems, or any other retrieval mechanism.

    Attributes:
        name: Backend identifier for logging and configuration.
    """

    @property
    def name(self) -> str:
        """Backend identifier for logging and configuration."""
        ...

    async def search(
        self,
        query: str,
        project_id: str,
        limit: int,
    ) -> list[ISearchResult]:
        """Execute search and return results.

        Args:
            query: Search query string.
            project_id: Project identifier for filtering.
            limit: Maximum number of results.

        Returns:
            List of search results sorted by relevance.
        """
        ...

    async def add(
        self,
        project_id: str,
        documents: list[str],
    ) -> None:
        """Add documents to the index.

        Args:
            project_id: Project identifier.
            documents: List of documents to index.
        """
        ...

    async def delete(
        self,
        document_id: str,
    ) -> None:
        """Remove document from the index.

        Args:
            document_id: Document identifier to remove.
        """
        ...

    async def commit(self) -> None:
        """Commit pending changes to storage."""
        ...

    async def close(self) -> None:
        """Release resources."""
        ...

    def ensure_initialized(self) -> None:
        """Ensure backend is initialized (thread-safe)."""
        ...


@runtime_checkable
class IFusionAlgorithm(Protocol):
    """Protocol for result fusion algorithms.

    This protocol defines the interface for combining results from
    multiple search backends. Implementations can use RRF, CombMNZ,
    semantic combination, or any other fusion strategy.

    Attributes:
        name: Algorithm identifier for configuration.
    """

    @property
    def name(self) -> str:
        """Algorithm identifier for configuration."""
        ...

    def fuse(
        self,
        results: dict[str, list[ISearchResult]],
        limit: int,
    ) -> list[ISearchResult]:
        """Combine results from multiple backends.

        Args:
            results: Dict mapping backend name to results.
            limit: Maximum results to return.

        Returns:
            Combined and ranked results.
        """
        ...

    def normalize_scores(
        self,
        results: list[ISearchResult],
    ) -> list[ISearchResult]:
        """Normalize scores to a common scale.

        Args:
            results: List of results to normalize.

        Returns:
            Results with normalized scores.
        """
        ...


@runtime_checkable
class ISearchPipeline(Protocol):
    """Protocol for search pipeline orchestration.

    This protocol defines the interface for the search pipeline that
    coordinates backends, fusion, and reranking.
    """

    async def execute(
        self,
        context: SearchContext,
    ) -> list[str]:
        """Execute the full search pipeline.

        Args:
            context: Search context with query and configuration.

        Returns:
            List of matching message strings.
        """
        ...
