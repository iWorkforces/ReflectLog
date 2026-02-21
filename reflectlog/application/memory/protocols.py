"""Protocols for search engine implementations."""

from typing import Protocol


class SearchEngine(Protocol):
    """Protocol for search engines (USearch, Tantivy, etc.).

    This protocol defines the interface that search engines must implement
    to work with the hybrid search system. Both semantic (USearch) and full-text
    (Tantivy) engines should conform to this interface.
    """

    def add(self, project_id: str, content: str) -> None:
        """Add a memory to the index.

        Args:
            project_id: Project identifier for filtering.
            content: Memory content to index.
        """
        ...

    def search(
        self, query: str, project_id: str, limit: int
    ) -> list[tuple[str, float]]:
        """Search and return (memory, score) tuples.

        Args:
            query: Search query string.
            project_id: Filter results by project_id.
            limit: Maximum number of results to return.

        Returns:
            List of (memory, score) tuples sorted by relevance.
        """
        ...

    def commit(self) -> None:
        """Commit pending changes to the index."""
        ...

    @property
    def name(self) -> str:
        """Engine name for logging purposes."""
        ...
