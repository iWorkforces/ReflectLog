"""Type stubs for usearch.index module.

This module provides type annotations for the USearch HNSW vector index library.
Only APIs used by ReflectLogMCP are stubbed.

Reference: https://github.com/unum-cloud/usearch
"""

from typing import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

class Match:
    """A single search result match.

    Attributes:
        key: The integer key associated with the matched vector.
        distance: The distance from the query vector (lower is more similar for cosine).
    """

    @property
    def key(self) -> int:
        """The key (ID) of the matched vector."""
        ...

    @property
    def distance(self) -> float:
        """The distance from the query vector."""
        ...

class Matches:
    """Collection of search result matches.

    Iterable container of Match objects returned by Index.search().
    Supports iteration and length.
    """

    def __iter__(self) -> Iterator[Match]:
        """Iterate over match results."""
        ...

    def __len__(self) -> int:
        """Return number of matches."""
        ...

    def __getitem__(self, index: int) -> Match:
        """Get match at index."""
        ...

class BatchMatches:
    """Collection of search result matches for batch queries.

    Iterable container of Match objects returned by Index.search()
    when searching with a single vector. Supports iteration and length.
    """

    def __iter__(self) -> Iterator[Match]:
        """Iterate over match results."""
        ...

    def __len__(self) -> int:
        """Return number of matches."""
        ...

    def __getitem__(self, index: int) -> Match:
        """Get match at index."""
        ...

class Index:
    """USearch HNSW (Hierarchical Navigable Small World) vector index.

    A high-performance approximate nearest neighbor search index supporting
    various distance metrics and efficient disk persistence.

    Example:
        ```python
        from usearch.index import Index

        # Create new index
        index = Index(
            ndim=128,
            metric="cos",
            dtype="f32",
        )

        # Add vectors
        index.add(1, vector1)
        index.add(2, vector2)

        # Search
        matches = index.search(query_vector, 10)
        for match in matches:
            print(f"Key: {match.key}, Distance: {match.distance}")

        # Save and restore
        index.save("index.usearch")
        restored = Index.restore("index.usearch")
        ```
    """

    def __init__(
        self,
        ndim: int,
        metric: str = "cos",
        dtype: str = "f32",
        connectivity: int = 16,
        expansion_add: int = 128,
        expansion_search: int = 64,
    ) -> None:
        """Initialize a new USearch index.

        Args:
            ndim: Number of dimensions for vectors.
            metric: Distance metric - "cos" (cosine), "l2" (Euclidean), "ip" (inner product).
            dtype: Data type - "f32" (float32), "f16" (float16), "i8" (int8).
            connectivity: HNSW M parameter - max edges per node (default: 16).
            expansion_add: HNSW efConstruction - search width during insertion (default: 128).
            expansion_search: HNSW ef - search width during queries (default: 64).
        """
        ...

    @classmethod
    def restore(cls, path: str) -> "Index | None":
        """Load an index from disk.

        Args:
            path: Path to the saved index file.

        Returns:
            Restored Index instance, or None if loading fails.

        Raises:
            FileNotFoundError: If the index file doesn't exist.
            RuntimeError: If the index file is corrupted.
        """
        ...

    def add(
        self,
        key: int,
        vector: NDArray[np.float32] | Sequence[float],
    ) -> None:
        """Add a vector to the index.

        Args:
            key: Unique integer key for the vector.
            vector: The vector to add (must match index ndim).

        Raises:
            ValueError: If vector dimensions don't match index ndim.
        """
        ...

    def search(
        self,
        vector: NDArray[np.float32] | Sequence[float],
        count: int,
        exact: bool = False,
    ) -> BatchMatches:
        """Search for nearest neighbors.

        Args:
            vector: Query vector to find neighbors for.
            count: Maximum number of results to return.
            exact: If True, performs exact brute-force search bypassing HNSW.
                   If False (default), uses approximate HNSW search.

        Returns:
            BatchMatches object containing search results ordered by distance.
        """
        ...

    def remove(self, key: int) -> None:
        """Remove a vector from the index by its key.

        Args:
            key: The key of the vector to remove.

        Note:
            USearch marks vectors as deleted but doesn't reclaim space
            until the index is compacted or rebuilt.
        """
        ...

    def save(self, path: str) -> None:
        """Save the index to disk.

        Args:
            path: Path where the index should be saved.

        Raises:
            IOError: If the file cannot be written.
        """
        ...

    def __len__(self) -> int:
        """Return the number of vectors in the index."""
        ...

    def __contains__(self, key: int) -> bool:
        """Check if a key exists in the index.

        Args:
            key: The key to check.

        Returns:
            True if key exists, False otherwise.
        """
        ...
