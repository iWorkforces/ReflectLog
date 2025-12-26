"""USearch-based semantic search engine.

This module provides a semantic search engine using USearch for vector
similarity search, with SQLite for message text storage. It implements
the ISemanticSearchEngine protocol for compatibility with MemoryManager.

Clean Architecture Compliance:
    This module implements the ISemanticSearchEngine protocol defined in
    the application layer (ccmemories.application.types), following the
    Dependency Inversion Principle from SOLID.
"""

import os
import threading
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union, cast

import numpy as np
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, PrivateAttr
from usearch.index import Index, BatchMatches

from ccmemories.application.exceptions import StorageError
from ccmemories.application.types import ISemanticSearchEngine
from ccmemories.application.utils.numba_utils import distance_to_similarity_cosine
from ccmemories.application.utils.security import validate_project_id
from ccmemories.infrastructure.message_store import MessageStore


@dataclass(frozen=True)
class USearchConfig:
    """Configuration for USearchEngine.

    This dataclass defines the settings needed for USearch vector search
    with SQLite message storage.

    Attributes:
        project_id: Project identifier for filtering.
        index_path: Path to the USearch index file.
        db_path: Path to the SQLite message database.
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
    def from_app_config(cls, config: Any) -> "USearchConfig":
        """Create USearchConfig from application Config.

        Args:
            config: Application Config instance.

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
            db_path=os.path.join(base_path, "messages.db"),
            embedding_dims=embedding_dims,
            exact_search=exact_search,
            exact_search_threshold=exact_search_threshold,
        )


class USearchEngine(BaseModel):
    """USearch-based semantic search engine.

    Implements ISemanticSearchEngine protocol via structural subtyping.
    Uses USearch for vector similarity search and SQLite for message text storage.

    Uses lazy initialization for both USearch index and SQLite connection
    with thread-safe double-checked locking.

    Example:
        ```python
        from ccmemories.infrastructure import USearchConfig, USearchEngine
        from langchain_openai import OpenAIEmbeddings

        config = USearchConfig(
            project_id="my-project",
            index_path="indexes/my-project/usearch/vectors.usearch",
            db_path="indexes/my-project/usearch/messages.db",
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
    logger: Any = None

    _index: Optional[Index] = PrivateAttr(default=None)
    _message_store: Optional[MessageStore] = PrivateAttr(default=None)
    # Instance-level lock for thread-safe lazy initialization (prevents cross-instance contention)
    _init_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def __init__(
        self,
        config: USearchConfig | dict[str, Any],
        embedder: Embeddings,
        logger: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize USearchEngine.

        Args:
            config: USearchConfig or dict with engine settings.
            embedder: Embedding provider implementing Embeddings interface.
            logger: Optional StructuredLogger instance.
            **kwargs: Additional arguments passed to BaseModel.
        """
        if isinstance(config, dict):
            config = USearchConfig(**cast(dict[str, Any], config))

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

                    # Try to load existing index
                    if os.path.exists(self.config.index_path):
                        loaded_index = Index.restore(self.config.index_path)
                        if loaded_index is None:
                            raise RuntimeError(
                                f"Failed to load USearch index from {self.config.index_path}"
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
                    else:
                        # Create new index
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

        # After initialization, _index is guaranteed to be non-None
        assert self._index is not None
        return self._index

    @property
    def message_store(self) -> MessageStore:
        """Get MessageStore (thread-safe lazy initialization).

        Returns:
            MessageStore instance.
        """
        if self._message_store is not None:
            return self._message_store

        with self._init_lock:
            if self._message_store is None:
                self._message_store = MessageStore(
                    db_path=self.config.db_path,
                    logger=self.logger,
                )

        return self._message_store

    def add(self, project_id: str, message: str, infer: bool) -> None:
        """Add a message to the USearch index.

        Args:
            project_id: Project identifier for filtering.
            message: Message content to index.
            infer: Whether to enable LLM-based message inference (not supported).

        Raises:
            RuntimeError: If add operation fails.
        """
        try:
            if infer:
                if self.logger:
                    self.logger.warning(
                        "infer=True not supported by USearchEngine, proceeding as infer=False",
                        extra={"project_id": self.config.project_id},
                    )

            # Insert into SQLite (relies on UNIQUE INDEX for dedup - no pre-check needed)
            msg_id = self.message_store.insert(project_id, message)

            # Generate embedding
            vector = self.embedder.embed_query(message)
            vector_np = np.array(vector, dtype=np.float32)

            # Add to USearch index
            self.index.add(msg_id, vector_np)

            if self.logger:
                self.logger.debug(
                    "Message added to USearch index",
                    extra={
                        "project_id": self.config.project_id,
                        "message_id": msg_id,
                        "message_length": len(message),
                    },
                )

        except (RuntimeError, StorageError) as e:
            # Handle duplicate message (detected by database UNIQUE constraint)
            # Note: StorageError is raised by MessageStore for duplicates
            if "Duplicate message" in str(e):
                if self.logger:
                    self.logger.debug(
                        "Skipping duplicate message (detected by DB constraint)",
                        extra={"project_id": self.config.project_id},
                    )
                return
            # Re-raise other errors
            if self.logger:
                self.logger.error(
                    "Failed to add message to USearch index",
                    extra={
                        "project_id": self.config.project_id,
                        "error": str(e),
                    },
                )
            raise
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Failed to add message to USearch index",
                    extra={
                        "project_id": self.config.project_id,
                        "error": str(e),
                    },
                )
            raise RuntimeError(f"Failed to add message: {e}") from e

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

    def search(
        self,
        query: str,
        project_id: str,
        limit: int,
    ) -> List[Tuple[str, float, str]]:
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
            List of (message, score, created_at) tuples sorted by relevance.
            created_at is an ISO format timestamp string (may be empty for
            backward compatibility with older data).
        """
        try:
            # Check if index is empty
            if len(self.index) == 0:
                if self.logger:
                    self.logger.debug(
                        "USearch index is empty",
                        extra={"project_id": self.config.project_id},
                    )
                return []

            # Determine search mode (exact vs approximate)
            use_exact = self._should_use_exact_search()
            index_size = len(self.index)

            if self.logger:
                search_mode = (
                    "exact (brute-force)" if use_exact else "approximate (HNSW)"
                )
                self.logger.debug(
                    f"USearch search mode: {search_mode}",
                    extra={
                        "project_id": self.config.project_id,
                        "search_mode": "exact" if use_exact else "approximate",
                        "index_size": index_size,
                        "exact_search_config": self.config.exact_search,
                        "exact_search_threshold": self.config.exact_search_threshold,
                    },
                )

            # Generate query embedding
            query_vector = self.embedder.embed_query(query)
            query_np = np.array(query_vector, dtype=np.float32)

            # Over-fetch to account for project_id filtering
            overfetch_limit = min(limit * 3, len(self.index))
            matches: BatchMatches = self.index.search(
                query_np, overfetch_limit, exact=use_exact
            )

            # Batch fetch all message records in single query (eliminates N+1 pattern)
            keys = [int(match.key) for match in matches]
            records = self.message_store.get_batch(keys)

            # Collect distances for batch conversion (numba-accelerated)
            filtered_matches: List[Tuple[Any, Any]] = []  # (record, match)
            for match in matches:
                key = int(match.key)
                record = records.get(key)
                if record is not None and record.project_id == project_id:
                    filtered_matches.append((record, match))
                    if len(filtered_matches) >= limit:
                        break

            # Batch convert distances to similarities using numba
            if filtered_matches:
                distances = np.array(
                    [match.distance for _, match in filtered_matches],
                    dtype=np.float32,
                )
                similarities = distance_to_similarity_cosine(distances)

                # Build results with converted scores and created_at timestamps
                results: List[Tuple[str, float, str]] = [
                    (record.message, float(similarities[i]), record.created_at)
                    for i, (record, _) in enumerate(filtered_matches)
                ]
            else:
                results = []

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

    def get_all(self, project_id: str) -> List[str]:
        """Retrieve all stored messages for a project.

        Args:
            project_id: Project identifier for filtering.

        Returns:
            List of all messages stored for the project.
        """
        try:
            messages = self.message_store.get_all(project_id)

            if self.logger:
                self.logger.debug(
                    "Retrieved all messages",
                    extra={
                        "project_id": self.config.project_id,
                        "count": len(messages),
                    },
                )

            return messages

        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Failed to retrieve messages",
                    extra={
                        "project_id": self.config.project_id,
                        "error": str(e),
                    },
                )
            raise RuntimeError(f"Failed to retrieve messages: {e}") from e

    def delete(self, memory_id: str) -> None:
        """Delete a memory entry by its ID.

        Args:
            memory_id: The ID of the memory to delete (SQLite row ID as string).

        Raises:
            RuntimeError: If deletion fails.
        """
        try:
            msg_id = int(memory_id)

            # Check if key exists in index
            if msg_id in self.index:
                self.index.remove(msg_id)

            # Delete from SQLite
            deleted = self.message_store.delete(msg_id)

            if self.logger:
                if deleted:
                    self.logger.debug(
                        "Message deleted from USearch index",
                        extra={
                            "project_id": self.config.project_id,
                            "memory_id": memory_id,
                        },
                    )
                else:
                    self.logger.warning(
                        "Message not found for deletion",
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
                    "Failed to delete message",
                    extra={
                        "project_id": self.config.project_id,
                        "memory_id": memory_id,
                        "error": str(e),
                    },
                )
            raise RuntimeError(f"Failed to delete message: {e}") from e

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

        Forces the lazy initialization of USearch index and MessageStore
        to complete. Call this before parallel operations.
        """
        _ = self.index
        self.message_store.ensure_initialized()

    def get_id_by_message(self, project_id: str, message: str) -> Union[int, None]:
        """Get the ID of a message by its content.

        Args:
            project_id: Project identifier.
            message: Message text to look up.

        Returns:
            The message ID if found, None otherwise.
        """
        return self.message_store.get_id_by_message(project_id, message)

    def close(self) -> None:
        """Close resources and cleanup.

        Closes the MessageStore SQLite connection.
        """
        if self._message_store is not None:
            self._message_store.close()


# Static type assertion: verify USearchEngine implements ISemanticSearchEngine protocol
# This line has no runtime effect but causes mypy to verify structural compatibility
_: type[ISemanticSearchEngine] = USearchEngine
