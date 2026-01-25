"""Base classes for search engines.

This module provides the base class that all search engine implementations
should inherit from. It provides common functionality and ensures
conformance to the ISearchBackend protocol.
"""

from typing import Protocol, runtime_checkable, Optional
from reflectlog.core.search import ISearchBackend, ISearchResult


class SearchEngineBase:
    """Base class for search engine implementations.

    This class provides common functionality for search engines and
    ensures conformance to the ISearchBackend protocol. Implementations
    should override the async methods with backend-specific logic.

    Attributes:
        _name: Backend identifier for logging.
    """

    _name: str = "base"

    @property
    def name(self) -> str:
        """Backend identifier for logging."""
        return self._name

    async def search(
        self,
        query: str,
        project_id: str,
        limit: int,
    ) -> list[ISearchResult]:
        """Default search implementation returns empty results.

        Args:
            query: Search query string.
            project_id: Project identifier for filtering.
            limit: Maximum number of results.

        Returns:
            Empty list.
        """
        return []

    async def add(
        self,
        project_id: str,
        documents: list[str],
    ) -> None:
        """Default add implementation does nothing.

        Args:
            project_id: Project identifier.
            documents: Documents to add.
        """
        pass

    async def delete(
        self,
        document_id: str,
    ) -> None:
        """Default delete implementation does nothing.

        Args:
            document_id: Document identifier.
        """
        pass

    async def commit(self) -> None:
        """Default commit does nothing."""
        pass

    async def close(self) -> None:
        """Default close does nothing."""
        pass

    def ensure_initialized(self) -> None:
        """Default initialization does nothing."""
        pass

    async def search(
        self,
        query: str,
        project_id: str,
        limit: int,
    ) -> list[ISearchResult]:
        """Default search implementation returns empty results.

        Args:
            query: Search query string.
            project_id: Project identifier for filtering.
            limit: Maximum number of results.

        Returns:
            Empty list.
        """
        return []

    async def add(
        self,
        project_id: str,
        documents: list[str],
    ) -> None:
        """Default add implementation does nothing.

        Args:
            project_id: Project identifier.
            documents: Documents to add.
        """
        pass

    async def delete(
        self,
        document_id: str,
    ) -> None:
        """Default delete implementation does nothing.

        Args:
            document_id: Document identifier.
        """
        pass

    async def commit(self) -> None:
        """Default commit does nothing."""
        pass

    async def close(self) -> None:
        """Default close does nothing."""
        pass

    def ensure_initialized(self) -> None:
        """Default initialization does nothing."""
        pass
