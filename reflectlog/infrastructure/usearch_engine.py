"""USearch-based semantic search engine.

This module provides a semantic search engine using USearch for vector
similarity search, with SQLite for memory content storage. It implements
the ISemanticSearchEngine protocol for compatibility with MemoryManager.

Clean Architecture Compliance:
    This module implements the ISemanticSearchEngine protocol defined in
    ``reflectlog.core.types.ISemanticSearchEngine``, following the
    Dependency Inversion Principle from SOLID.
"""

from dataclasses import dataclass
import os
import threading
from typing import TYPE_CHECKING, Any, Self, final

if TYPE_CHECKING:
    from typing import TypeGuard

    from reflectlog.infrastructure.memory_store import MemoryRecord, MemoryStore

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr
from usearch.index import BatchMatches, Index

from reflectlog.core.config import IAppConfig
from reflectlog.core.exceptions import StorageError
from reflectlog.core.logging import IStructuredLogger
from reflectlog.core.types import Embeddings
from reflectlog.utility.scoring import distance_to_similarity_cosine
from reflectlog.utility.security import validate_project_id


def _is_dict_config(config: object) -> TypeGuard[dict[str, Any]]:
    """Type guard to check if config is a dict."""
    return isinstance(config, dict)


@dataclass(frozen=True)
class USearchConfig:
    """Configuration for USearchEngine.

    This dataclass defines the settings needed for USearch vector search
    with SQLite memory storage.

    Attributes:
        project_id: Project identifier for filtering.
        index_path: Path to the USearch index file.
        db_path: Path to the SQLite memory database.
        embedding_dims: Vector embedding dimensions.
        metric: Distance metric (cos, l2, ip).
        connectivity: HNSW M parameter.
        expansion_add: HNSW efConstruction parameter.
        expansion_search: HNSW ef search parameter.
        exact_search: Force exact brute-force search instead of HNSW approximation.
            When True, bypasses HNSW indexing and uses SIMD-optimized similarity
            metrics from SimSIMD for guaranteed exact results. Best for small
            collections (< 10k vectors). Default: False.
        exact_search_threshold: Auto-switch to exact search when index size is
            below this threshold. Set to 0 to disable auto-switching.
            When index size < threshold, exact search is used regardless of
            exact_search setting. Default: 0 (disabled).
    """

    project_id: str
    index_path: str
    db_path: str
    embedding_dims: int
    metric: str = "cos"
    connectivity: int = 16
    expansion_add: int = 128
    expansion_search: int = 64
    exact_search: bool = True
    exact_search_threshold: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> USearchConfig:
        """Create USearchConfig from a dictionary with validation.

        This method properly validates dict contents and returns a typed instance,
        avoiding the type: ignore needed when using **dict unpacking.

        Args:
            data: Dictionary with configuration values.

        Returns:
            Validated USearchConfig instance.
        """
        return cls(
            project_id=data.get("project_id", "") or "",
            index_path=data.get("index_path", "") or "",
            db_path=data.get("db_path", "") or "",
            embedding_dims=int(data.get("embedding_dims", 3072)),
            metric=data.get("metric", "cos") or "cos",
            connectivity=int(data.get("connectivity", 16)),
            expansion_add=int(data.get("expansion_add", 128)),
            expansion_search=int(data.get("expansion_search", 64)),
            exact_search=bool(data.get("exact_search", True)),
            exact_search_threshold=int(data.get("exact_search_threshold", 0)),
        )

    @classmethod
    def from_config(cls, config: IAppConfig) -> USearchConfig:
        """Create USearchConfig from IAppConfig protocol.

        Args:
            config: Configuration satisfying IAppConfig protocol.

        Returns:
            USearchConfig with extracted settings.
        """
        # Validate project_id to prevent path traversal attacks
        project_id = validate_project_id(config.project_id)
        base_path = os.path.join(os.getcwd(), "indexes", project_id, "usearch")

        # Determine embedding dims based on provider
        embedding_dims = (
            config.qwen_embedding_dims
            if config.embedder_provider == "langchain"
            else config.embedding_dims
        )

        # Get exact search settings with safe defaults for missing attributes
        exact_search = getattr(config, "usearch_exact_search", False)
        exact_search_threshold = getattr(
            config, "usearch_exact_search_threshold", 10000
        )

        return cls(
            project_id=project_id,
            index_path=os.path.join(base_path, "vectors.usearch"),
            db_path=os.path.join(base_path, "memories.db"),
            embedding_dims=embedding_dims,
            exact_search=exact_search,
            exact_search_threshold=exact_search_threshold,
        )


@final
class USearchEngine(BaseModel):
    """USearch-based semantic search engine.

    Implements ISemanticSearchEngine protocol via structural subtyping.
    Uses USearch for vector similarity search and SQLite for memory content storage.

    Uses lazy initialization for both USearch index and SQLite connection
    with thread-safe double-checked locking.

    Example:
        ```python
        from reflectlog.infrastructure.usearch_engine import USearchConfig, USearchEngine
        from langchain_openai import OpenAIEmbeddings

        config = USearchConfig(
            project_id="my-project",
            index_path="indexes/my-project/usearch/vectors.usearch",
            db_path="indexes/my-project/usearch/memories.db",
            embedding_dims=3072,
        )
        embedder = OpenAIEmbeddings(model="text-embedding-3-large")
        engine = USearchEngine(config=config, embedder=embedder, logger=logger)

        engine.add("my-project", "Hello world", infer=False)
        results = engine.search("hello", "my-project", limit=5)
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: USearchConfig
    embedder: Embeddings
    logger: IStructuredLogger | None = None

    _index: Index | None = PrivateAttr(default=None)
    _memory_store: MemoryStore | None = PrivateAttr(default=None)
    # Instance-level lock for thread-safe lazy initialization (prevents cross-instance contention)
    _init_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def __init__(
        self,
        config: USearchConfig | dict[str, Any],
        embedder: Embeddings,
        logger: IStructuredLogger | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize USearchEngine.

        Args:
            config: USearchConfig or dict with engine settings.
            embedder: Embedding provider implementing Embeddings interface.
            logger: Optional StructuredLogger instance.
            **kwargs: Additional arguments passed to BaseModel.
        """
        if _is_dict_config(config):
            config = USearchConfig.from_dict(config)

        super().__init__(config=config, embedder=embedder, logger=logger, **kwargs)

    @property
    def name(self) -> str:
        """Engine name for identification."""
        return "usearch"

    @property
    def index(self) -> Index:
        """Get USearch index (thread-safe lazy initialization).

        Returns:
            USearch Index instance.

        Raises:
            RuntimeError: If index initialization fails.
        """
        if self._index is not None:
            return self._index

        with self._init_lock:
            if self._index is None:
                try:
                    # Ensure directory exists
                    index_dir = os.path.dirname(self.config.index_path)
                    if index_dir:
                        os.makedirs(index_dir, exist_ok=True)

                    # Optimization: Try restore first, avoid extra os.path.exists() call
                    # This is faster for existing indices (one less syscall)
                    try:
                        loaded_index = Index.restore(self.config.index_path)
                        if loaded_index is None:
                            raise RuntimeError(
                                f"Index.restore() returned None for {self.config.index_path}"
                            )
                        self._index = loaded_index
                        if self.logger:
                            self.logger.info(
                                "Loaded existing USearch index",
                                extra={
                                    "project_id": self.config.project_id,
                                    "index_path": self.config.index_path,
                                    "size": len(loaded_index),
                                },
                            )
                    except RuntimeError, FileNotFoundError, OSError:
                        # Index doesn't exist or is corrupted, create new one
                        if self.logger:
                            self.logger.debug(
                                "USearch index not found, creating new index",
                                extra={
                                    "project_id": self.config.project_id,
                                    "index_path": self.config.index_path,
                                },
                            )
                        new_index = Index(
                            ndim=self.config.embedding_dims,
                            metric=self.config.metric,
                            dtype="f32",
                            connectivity=self.config.connectivity,
                            expansion_add=self.config.expansion_add,
                            expansion_search=self.config.expansion_search,
                        )
                        self._index = new_index
                        if self.logger:
                            self.logger.info(
                                "Created new USearch index",
                                extra={
                                    "project_id": self.config.project_id,
                                    "index_path": self.config.index_path,
                                    "dims": self.config.embedding_dims,
                                },
                            )

                except Exception as exc:
                    if self.logger:
                        self.logger.error(
                            "Failed to initialize USearch index",
                            extra={
                                "project_id": self.config.project_id,
                                "error": str(exc),
                            },
                            exc_info=True,
                        )
                    raise RuntimeError(
                        f"Failed to initialize USearch index: {exc}"
                    ) from exc

        # _index is guaranteed to be non-None after successful initialization
        # An exception would have been raised above if initialization failed
        return self._index

    @property
    def memory_store(self) -> MemoryStore:
        """Get MemoryStore (thread-safe lazy initialization).

        Returns:
            MemoryStore instance.
        """
        if self._memory_store is not None:
            return self._memory_store

        with self._init_lock:
            if self._memory_store is None:
                from reflectlog.infrastructure.memory_store import MemoryStore

                self._memory_store = MemoryStore(
                    db_path=self.config.db_path,
                    logger=self.logger,
                )

        return self._memory_store

    def add(
        self,
        project_id: str,
        content: str,
        infer: bool,
    ) -> None:
        """Add a memory to the USearch index.

        Args:
            project_id: Project identifier for filtering.
            content: Memory content to index.
            infer: Whether to enable LLM-based memory inference (not supported).

        Raises:
            RuntimeError: If add operation fails.
        """
        mem_id = None  # Track mem_id for rollback on embedding failure
        try:
            if infer and self.logger:
                self.logger.warning(
                    "infer=True not supported by USearchEngine, proceeding as infer=False",
                    extra={"project_id": self.config.project_id},
                )

            # Insert into SQLite (relies on UNIQUE INDEX for dedup - no pre-check needed)
            mem_id = self.memory_store.insert(project_id, content)

            # Generate embedding with rollback on failure
            try:
                vector = self.embedder.embed_query(content)
                vector_np = np.array(vector, dtype=np.float32)
            except Exception as embed_error:
                # Rollback SQLite insert if embedding fails to prevent desynchronization
                _ = self.memory_store.delete(mem_id)
                if self.logger:
                    self.logger.error(
                        "Embedding generation failed, rolled back SQLite insert",
                        extra={
                            "project_id": self.config.project_id,
                            "error": str(embed_error),
                        },
                    )
                raise RuntimeError(
                    f"Failed to generate embedding: {embed_error}"
                ) from embed_error

            # Add to USearch index
            self.index.add(mem_id, vector_np)

            if self.logger:
                self.logger.debug(
                    "Memory added to USearch index",
                    extra={
                        "project_id": self.config.project_id,
                        "memory_id": mem_id,
                        "memory_length": len(content),
                    },
                )

        except (RuntimeError, StorageError) as e:
            # Handle duplicate memory (detected by database UNIQUE constraint)
            if "Duplicate memory" in str(e):
                if self.logger:
                    self.logger.debug(
                        "Skipping duplicate memory (detected by DB constraint)",
                        extra={"project_id": self.config.project_id},
                    )
                return
            # Re-raise other errors
            if self.logger:
                self.logger.error(
                    "Failed to add memory to USearch index",
                    extra={
                        "project_id": self.config.project_id,
                        "error": str(e),
                    },
                )
            raise
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Failed to add memory to USearch index",
                    extra={
                        "project_id": self.config.project_id,
                        "error": str(e),
                    },
                )
            raise RuntimeError(f"Failed to add memory: {e}") from e

    def add_batch(
        self,
        project_id: str,
        contents: list[str],
        infer: bool,
    ) -> list[str]:
        """Add multiple memories to the USearch index in a single batch.

        Args:
            project_id: Project identifier for filtering.
            contents: List of memory texts to index.
            infer: Whether to enable LLM-based memory inference (not supported).

        Returns:
            List of memory contents successfully added (duplicates skipped).
        """
        if not contents:
            return []

        if infer and self.logger:
            self.logger.warning(
                "infer=True not supported by USearchEngine, proceeding as infer=False",
                extra={"project_id": self.config.project_id},
            )

        inserted = []  # Track successfully inserted (content, mem_id) pairs
        inserted_ids = []  # Track IDs for rollback on embedding failure

        try:
            # First, insert all memories into SQLite
            inserted = self.memory_store.insert_many(project_id, contents)
            if not inserted:
                return []

            inserted_contents = [content for content, _ in inserted]
            inserted_ids = [mem_id for _, mem_id in inserted]

            # Generate embeddings with rollback on failure
            try:
                vectors = self.embedder.embed_documents(inserted_contents)
                if len(vectors) != len(inserted_contents):
                    raise RuntimeError(
                        "Embedding batch size mismatch for USearch add_batch"
                    )
            except Exception as embed_error:
                # Rollback all SQLite inserts if embedding fails
                if self.logger:
                    self.logger.error(
                        "Embedding generation failed, rolling back batch",
                        extra={
                            "project_id": self.config.project_id,
                            "count": len(inserted_ids),
                            "error": str(embed_error),
                        },
                    )
                for mem_id in inserted_ids:
                    _ = self.memory_store.delete(mem_id)
                raise RuntimeError(
                    f"Failed to generate embeddings for batch: {embed_error}"
                ) from embed_error

            # Add vectors to USearch index
            for (_, mem_id), vector in zip(inserted, vectors, strict=True):
                vector_np = np.array(vector, dtype=np.float32)
                self.index.add(mem_id, vector_np)

            if self.logger:
                self.logger.debug(
                    "Batch added memories to USearch index",
                    extra={
                        "project_id": self.config.project_id,
                        "memory_count": len(inserted_contents),
                    },
                )

            return inserted_contents

        except Exception as e:
            # Clean up on any other error (shouldn't happen if above is correct)
            if self.logger:
                self.logger.error(
                    "Failed to add memory batch to USearch index",
                    extra={
                        "project_id": self.config.project_id,
                        "error": str(e),
                    },
                )
            raise RuntimeError(f"Failed to add memory batch: {e}") from e

    def _should_use_exact_search(self) -> bool:
        """Determine if exact search should be used based on config and index size.

        Returns:
            True if exact (brute-force) search should be used, False for HNSW approximate.
        """
        # If exact search is explicitly forced, use it
        if self.config.exact_search:
            return True

        # If threshold is set and index size is below it, auto-switch to exact
        if self.config.exact_search_threshold > 0:
            index_size = len(self.index)
            if index_size < self.config.exact_search_threshold:
                return True

        # Default to approximate search (HNSW)
        return False

    def _rank_scores(self, count: int) -> np.ndarray:
        """Create rank-based scores from 1.0 to 0.0.

        Used when distance-to-similarity conversion is ambiguous.
        """
        if count <= 0:
            return np.array([], dtype=np.float32)
        if count == 1:
            return np.array([1.0], dtype=np.float32)
        return np.linspace(1.0, 0.0, num=count, dtype=np.float32)

    def _distances_to_scores(self, distances: np.ndarray) -> np.ndarray:
        """Convert index distances to similarity scores.

        Cosine distances are mapped to [0, 1] similarity.
        L2 distances are converted with a monotonic 1 / (1 + d) transform.
        Other metrics fall back to rank-based scoring to preserve ordering.
        """
        metric = self.config.metric.lower()
        if metric in ("cos", "cosine"):
            return distance_to_similarity_cosine(distances)
        if metric in ("l2", "euclidean"):
            return np.float32(1.0) / (np.float32(1.0) + distances)

        if self.logger:
            self.logger.warning(
                "USearch metric has ambiguous distance scale; using rank-based scores",
                extra={"metric": self.config.metric},
            )
        return self._rank_scores(len(distances))

    def search(
        self,
        query: str,
        project_id: str,
        limit: int,
    ) -> list[tuple[str, float, str]]:
        """Execute semantic search.

        Supports both exact (brute-force) and approximate (HNSW) search modes:
        - **Exact search**: Bypasses HNSW indexing and performs brute-force search
          using SIMD-optimized similarity metrics from SimSIMD. Best for small
          collections (< 10k vectors) where guaranteed exact results are needed.
        - **Approximate search**: Uses HNSW algorithm for fast nearest neighbor
          search. Best for larger collections where slight accuracy tradeoff
          for speed is acceptable.

        Search mode is determined by:
        1. `exact_search=True` in config: Always use exact search
        2. `exact_search_threshold > 0` and index size < threshold: Auto-switch to exact
        3. Otherwise: Use approximate (HNSW) search

        Args:
            query: Search query string.
            project_id: Filter results by project_id.
            limit: Maximum number of results.

        Returns:
            List of (content, score, created_at) tuples sorted by relevance.
            created_at is an ISO format timestamp string (may be empty for
            backward compatibility with older data).
        """
        try:
            if len(self.index) == 0:
                if self.logger:
                    self.logger.debug(
                        "USearch index is empty",
                        extra={"project_id": self.config.project_id},
                    )
                return []

            use_exact = self._should_use_exact_search()
            self._log_search_mode(use_exact)

            matches = self._execute_index_search(query, limit, use_exact)
            filtered_matches = self._filter_matches_by_project(
                matches, project_id, limit
            )
            results = self._build_search_results(filtered_matches)

            if self.logger:
                self.logger.debug(
                    "USearch search completed",
                    extra={
                        "project_id": self.config.project_id,
                        "query": query[:100],
                        "search_mode": "exact" if use_exact else "approximate",
                        "matches_found": len(matches),
                        "results_after_filter": len(results),
                        "numba_batch_size": len(filtered_matches),
                    },
                )

            return results

        except Exception as e:
            if self.logger:
                self.logger.error(
                    "USearch search failed",
                    extra={
                        "project_id": self.config.project_id,
                        "query": query[:100],
                        "error": str(e),
                    },
                )
            raise RuntimeError(f"USearch search failed: {e}") from e

    def _log_search_mode(self, use_exact: bool) -> None:
        """Log the selected search mode with configuration details."""
        if not self.logger:
            return
        search_mode = "exact (brute-force)" if use_exact else "approximate (HNSW)"
        self.logger.debug(
            f"USearch search mode: {search_mode}",
            extra={
                "project_id": self.config.project_id,
                "search_mode": "exact" if use_exact else "approximate",
                "index_size": len(self.index),
                "exact_search_config": self.config.exact_search,
                "exact_search_threshold": self.config.exact_search_threshold,
            },
        )

    def _execute_index_search(
        self, query: str, limit: int, use_exact: bool
    ) -> BatchMatches:
        """Generate query embedding and execute index search with overfetch."""
        query_vector = self.embedder.embed_query(query)
        query_np = np.array(query_vector, dtype=np.float32)
        overfetch_limit = min(limit * 3, len(self.index))
        return self.index.search(query_np, overfetch_limit, exact=use_exact)

    def _filter_matches_by_project(
        self,
        matches: BatchMatches,
        project_id: str,
        limit: int,
    ) -> list[tuple[MemoryRecord, Any]]:
        """Filter matches by project_id using batch record fetch."""
        keys = [int(match.key) for match in matches]
        records = self.memory_store.get_batch(keys)

        filtered: list[tuple[MemoryRecord, Any]] = []
        for match in matches:
            key = int(match.key)
            record = records.get(key)
            if record is not None and record.project_id == project_id:
                filtered.append((record, match))
                if len(filtered) >= limit:
                    break
        return filtered

    def _build_search_results(
        self, filtered_matches: list[tuple[MemoryRecord, Any]]
    ) -> list[tuple[str, float, str]]:
        """Convert filtered matches to scored results using numba distance conversion."""
        if not filtered_matches:
            return []

        distances = np.array(
            [match.distance for _, match in filtered_matches],
            dtype=np.float32,
        )
        similarities = self._distances_to_scores(distances)

        return [
            (record.content, float(similarities[i]), record.created_at)
            for i, (record, _) in enumerate(filtered_matches)
        ]

    def get_all(self, project_id: str) -> list[str]:
        """Retrieve all stored memories for a project.

        Args:
            project_id: Project identifier for filtering.

        Returns:
            List of all memories stored for the project.
        """
        try:
            memories = self.memory_store.get_all(project_id)

            if self.logger:
                self.logger.debug(
                    "Retrieved all memories",
                    extra={
                        "project_id": self.config.project_id,
                        "count": len(memories),
                    },
                )

            return memories

        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Failed to retrieve memories",
                    extra={
                        "project_id": self.config.project_id,
                        "error": str(e),
                    },
                )
            raise RuntimeError(f"Failed to retrieve memories: {e}") from e

    def delete(self, memory_id: str) -> None:
        """Delete a memory entry by its ID.

        Args:
            memory_id: The ID of the memory to delete (SQLite row ID as string).

        Raises:
            RuntimeError: If deletion fails.
        """
        try:
            mem_id = int(memory_id)

            # Check if key exists in index
            if mem_id in self.index:
                self.index.remove(mem_id)

            # Delete from SQLite
            deleted = self.memory_store.delete(mem_id)

            if self.logger:
                if deleted:
                    self.logger.debug(
                        "Memory deleted from USearch index",
                        extra={
                            "project_id": self.config.project_id,
                            "memory_id": memory_id,
                        },
                    )
                else:
                    self.logger.warning(
                        "Memory not found for deletion",
                        extra={
                            "project_id": self.config.project_id,
                            "memory_id": memory_id,
                        },
                    )

        except ValueError as e:
            if self.logger:
                self.logger.error(
                    "Invalid memory_id format",
                    extra={
                        "project_id": self.config.project_id,
                        "memory_id": memory_id,
                        "error": str(e),
                    },
                )
            raise RuntimeError(f"Invalid memory_id format: {e}") from e
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Failed to delete memory",
                    extra={
                        "project_id": self.config.project_id,
                        "memory_id": memory_id,
                        "error": str(e),
                    },
                )
            raise RuntimeError(f"Failed to delete memory: {e}") from e

    def commit(self) -> None:
        """Commit pending changes to the index.

        Saves the USearch index to disk. SQLite auto-commits.
        """
        try:
            if self._index is not None:
                self.index.save(self.config.index_path)
                if self.logger:
                    self.logger.debug(
                        "USearch index saved",
                        extra={
                            "project_id": self.config.project_id,
                            "index_path": self.config.index_path,
                            "size": len(self.index),
                        },
                    )
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Failed to save USearch index",
                    extra={
                        "project_id": self.config.project_id,
                        "error": str(e),
                    },
                )
            raise RuntimeError(f"Failed to save USearch index: {e}") from e

    def ensure_initialized(self) -> None:
        """Ensure the engine is fully initialized (thread-safe).

        Forces the lazy initialization of USearch index and MemoryStore
        to complete. Call this before parallel operations.
        """
        _ = self.index
        self.memory_store.ensure_initialized()

    def is_ready(self) -> bool:
        """Return True if the USearch index has already been loaded."""
        return self._index is not None

    def get_id_by_content(self, project_id: str, content: str) -> int | None:
        """Get the ID of a memory by its content.

        Args:
            project_id: Project identifier.
            content: Memory text to look up.

        Returns:
            The memory ID if found, None otherwise.
        """
        return self.memory_store.get_id_by_content(project_id, content)

    def close(self) -> None:
        """Close resources and cleanup.

        Closes the MemoryStore SQLite connection.
        """
        if self._memory_store is not None:
            self._memory_store.close()

    def __enter__(self) -> Self:
        """Enter context manager.

        Ensures the engine is initialized before use.

        Returns:
            self
        """
        self.ensure_initialized()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        """Exit context manager.

        Ensures resources are cleaned up.

        Args:
            exc_type: Exception type if raised
            exc_val: Exception value if raised
            exc_tb: Exception traceback if raised

        Returns:
            False to not suppress exceptions.
        """
        self.close()
        return False  # Don't suppress exceptions
