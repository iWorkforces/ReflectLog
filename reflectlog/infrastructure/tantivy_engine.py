"""Tantivy full-text search engine wrapper.

This module provides a wrapper around the Tantivy full-text search library,
following the same patterns as qwen3_embedding.py for consistency.
"""

from collections import OrderedDict
from dataclasses import dataclass
import os
import threading
import time
from typing import TYPE_CHECKING, Any, final

if TYPE_CHECKING:
    from typing import TypeGuard


import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr
import tantivy

from reflectlog.core.exceptions import SearchError
from reflectlog.core.logging import IStructuredLogger


def _is_dict_config(config: object) -> TypeGuard[dict[str, Any]]:
    """Type guard to check if config is a dict."""
    return isinstance(config, dict)


@dataclass(frozen=True)
class TantivyConfig:
    """Configuration for TantivyEngine.

    Attributes:
        project_id: Unique project identifier for filtering.
        index_path: Path to the Tantivy index directory.
        soft_delete_enabled: Use O(1) tombstone marking instead of O(n) rebuild.
        compaction_threshold_ratio: Compact when tombstones > this ratio of docs.
        compaction_max_tombstones: Force compaction above this tombstone count.
        tombstone_ttl_days: Days before tombstones are eligible for removal.
        tombstone_cache_max_size: Maximum number of project IDs to cache tombstones for.
        normalize_scores: Normalize BM25 scores to 0-1 range (batch min-max).
    """

    project_id: str
    index_path: str
    soft_delete_enabled: bool = True
    compaction_threshold_ratio: float = 0.2
    compaction_max_tombstones: int = 10000
    tombstone_ttl_days: int = 7
    tombstone_cache_max_size: int = 100
    normalize_scores: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TantivyConfig:
        """Create TantivyConfig from a dictionary with validation.

        This method properly validates dict contents and returns a typed instance,
        avoiding the type: ignore needed when using **dict unpacking.

        Args:
            data: Dictionary with configuration values.

        Returns:
            Validated TantivyConfig instance.
        """
        return cls(
            project_id=data.get("project_id", "") or "",
            index_path=data.get("index_path", "") or "",
            soft_delete_enabled=bool(data.get("soft_delete_enabled", True)),
            compaction_threshold_ratio=float(
                data.get("compaction_threshold_ratio", 0.2)
            ),
            compaction_max_tombstones=int(data.get("compaction_max_tombstones", 10000)),
            tombstone_ttl_days=int(data.get("tombstone_ttl_days", 7)),
            tombstone_cache_max_size=int(data.get("tombstone_cache_max_size", 100)),
            normalize_scores=bool(data.get("normalize_scores", True)),
        )


# Schema version constant (V2 only - soft-delete support)
TANTIVY_SCHEMA_VERSION = 2
DEFAULT_TANTIVY_DOC_LIMIT = 100000  # Fallback if searcher doc count unavailable


@final
class TantivyEngine(BaseModel):
    """Tantivy full-text search engine wrapper.

    Implements a SearchEngine-compatible interface for hybrid search integration.
    Uses Pydantic BaseModel for consistency with LangchainQwenEmbeddings.

    The engine provides:
    - Persistent index storage on disk
    - Lazy initialization of writer and searcher
    - Stemmed full-text search (en_stem tokenizer)
    - Project-level filtering via project_id field
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: TantivyConfig
    logger: IStructuredLogger | None = None

    _index: tantivy.Index | None = PrivateAttr(default=None)
    _writer: tantivy.IndexWriter | None = PrivateAttr(default=None)
    _searcher: tantivy.Searcher | None = PrivateAttr(default=None)
    # Instance-level locks for thread-safe operations
    # Note: Using RLock (re-entrant) because add() holds lock and
    # calls self.writer property
    _writer_lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)
    # Bounded in-memory tombstone cache for O(1) lookup
    # after first search
    # Uses OrderedDict for LRU eviction when size exceeds tombstone_cache_max_size
    # Key: project_id, Value: set of tombstoned memory contents
    _tombstone_cache: OrderedDict[str, set[str]] = PrivateAttr(
        default_factory=lambda: OrderedDict[str, set[str]]()
    )
    _tombstone_cache_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _searcher_lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)

    def __init__(
        self,
        config: TantivyConfig | dict[str, Any],
        logger: IStructuredLogger | None = None,
        **kwargs: Any,
    ):
        """Initialize TantivyEngine.

        Args:
            config: TantivyConfig or dict with project_id and index_path.
            logger: Optional StructuredLogger instance for logging.
            **kwargs: Additional arguments passed to BaseModel.

        Raises:
            RuntimeError: If index initialization fails.
        """
        if _is_dict_config(config):
            config = TantivyConfig.from_dict(config)

        super().__init__(config=config, logger=logger, **kwargs)
        self._initialize_index()

    @property
    def name(self) -> str:
        """Engine name for logging purposes."""
        return "tantivy"

    def _build_schema(self) -> tantivy.Schema:
        """Build Tantivy schema with project_id, content, and soft-delete fields.

        Returns:
            Tantivy schema with:
            - project_id: Stored text field with raw tokenizer (exact match)
            - content: Stored text field with en_stem tokenizer (full-text search)
            - is_deleted: Stored unsigned field for soft-delete (0=active, 1=deleted)
            - deleted_at: Stored integer field for deletion timestamp in ms
        """
        schema_builder = tantivy.SchemaBuilder()
        _ = schema_builder.add_text_field(
            "project_id", stored=True, tokenizer_name="raw"
        )
        _ = schema_builder.add_text_field(
            "content", stored=True, tokenizer_name="en_stem"
        )

        # Soft-delete fields
        # is_deleted: 0 = active, 1 = deleted
        # indexed=True for filtering, fast=True for fast columnar access
        _ = schema_builder.add_unsigned_field(
            "is_deleted", stored=True, indexed=True, fast=True
        )
        # deleted_at: Unix timestamp in milliseconds (0 if not deleted)
        # indexed=True for range queries during compaction
        _ = schema_builder.add_integer_field(
            "deleted_at", stored=True, indexed=True, fast=True
        )

        return schema_builder.build()

    def _initialize_index(self) -> None:
        """Initialize or load persistent Tantivy index.

        Creates the index directory if it doesn't exist.
        Attempts to load existing index, creates new one if not found.

        Raises:
            RuntimeError: If index creation fails.
        """
        index_path = self.config.index_path
        os.makedirs(index_path, exist_ok=True)

        try:
            # Try to open existing index
            self._index = tantivy.Index.open(index_path)
            if self.logger:
                self.logger.info(
                    "Loaded existing Tantivy index",
                    extra={
                        "project_id": self.config.project_id,
                        "tantivy_index_path": index_path,
                    },
                )
        except Exception:
            # Create new index
            schema = self._build_schema()
            self._index = tantivy.Index(schema, path=index_path, reuse=True)
            if self.logger:
                self.logger.info(
                    "Created new Tantivy index",
                    extra={
                        "project_id": self.config.project_id,
                        "tantivy_index_path": index_path,
                    },
                )

    @property
    def writer(self) -> tantivy.IndexWriter:
        """Get index writer (thread-safe lazy initialization).

        Uses double-checked locking pattern for thread-safe initialization.

        Returns:
            Tantivy IndexWriter for adding documents.

        Raises:
            RuntimeError: If index is not initialized.
        """
        # Fast path: already initialized
        if self._writer is not None:
            return self._writer

        # Slow path: need to initialize (acquire lock)
        with self._writer_lock:
            # Double-check after acquiring lock
            if self._writer is None:
                if self._index is None:
                    raise RuntimeError("Tantivy index not initialized")
                self._writer = self._index.writer()
        return self._writer

    @property
    def searcher(self) -> tantivy.Searcher:
        """Get index searcher (thread-safe lazy initialization).

        Uses double-checked locking pattern for thread-safe initialization.

        Returns:
            Tantivy Searcher for searching documents.

        Raises:
            RuntimeError: If index is not initialized.
        """
        # Fast path: already initialized
        if self._searcher is not None:
            return self._searcher

        # Slow path: need to initialize (acquire lock)
        with self._searcher_lock:
            # Double-check after acquiring lock
            if self._searcher is None:
                if self._index is None:
                    raise RuntimeError("Tantivy index not initialized")
                self._searcher = self._index.searcher()
        return self._searcher

    def _recreate_writer_if_needed(self) -> None:
        """Recreate the IndexWriter if it's in an invalid state (thread-safe).

        The Tantivy-py IndexWriter becomes invalid after certain operations
        like delete_documents(). This method safely recreates the writer.

        Must be called with _writer_lock already held.
        """
        if self._index is not None:
            self._writer = self._index.writer()

    def _get_all_docs(self, project_id: str) -> list[str]:
        """Get all active (non-tombstoned) documents for a project.

        Uses cached tombstone set for O(1) post-filtering after first call.

        Args:
            project_id: Project identifier to filter by.

        Returns:
            List of memory strings for the given project (excluding tombstoned).
        """
        if self._index is None:
            return []

        try:
            # Get cached tombstoned memories (O(1) after first call)
            tombstoned_memories = self._get_tombstoned_memories(project_id)
            doc_limit = self._get_doc_limit()

            # Query all docs for this project
            escaped_project_id = self._escape_tantivy_query(project_id)
            query = self._index.parse_query(
                query=f'project_id:"{escaped_project_id}"',
                default_field_names=["project_id"],
            )

            # Get all results (use index doc count to avoid truncation)
            top_docs = self.searcher.search(query=query, limit=doc_limit)

            results: list[str] = []
            seen: set[str] = set()  # Track seen memories to avoid duplicates

            for _, doc_addr in top_docs.hits:
                doc = self.searcher.doc(doc_addr)
                memory = doc.get_first("content")
                if memory is not None:
                    msg_str = memory
                    # Skip tombstoned memories and duplicates
                    if msg_str not in tombstoned_memories and msg_str not in seen:
                        results.append(msg_str)
                        seen.add(msg_str)

            return results

        except Exception as e:
            if self.logger:
                self.logger.warning(
                    "Failed to get all docs from Tantivy",
                    extra={
                        "project_id": project_id,
                        "error": str(e),
                    },
                )
            return []

    def find_by_exact_match(self, project_id: str, content: str) -> list[str]:
        """Find all memories that exactly match the given memory text.

        Uses Python-level string comparison after fetching all docs for the project.
        This works around the en_stem tokenizer's stemming behavior which prevents
        exact phrase matching via Tantivy queries.

        Args:
            project_id: Project identifier to filter by.
            content: Exact memory text to find.

        Returns:
            List of matching memory strings (may contain duplicates if stored multiple times).
        """
        all_docs = self._get_all_docs(project_id)
        return [doc for doc in all_docs if doc == content]

    def _get_all_docs_all_projects(self) -> list[tuple[str, str]]:
        """Get all documents from all projects.

        Returns:
            List of (project_id, content) tuples for all documents in the index.
        """
        if self._index is None:
            return []

        try:
            doc_limit = self._get_doc_limit()

            # Use match-all query by searching for common patterns
            # Tantivy doesn't have a built-in match-all, so we use the searcher directly
            searcher = self.searcher

            # Get all documents by iterating through segment readers
            results: list[tuple[str, str]] = []

            # Use a query that matches everything - project_id always exists
            # Search for project_id:* doesn't work, so we'll try multiple common project patterns
            # Actually, let's use the searcher's doc method with addresses

            # Alternative approach: search for any document with project_id field
            # We'll use a very broad search since all docs have project_id
            query = self._index.parse_query(
                query="*",
                default_field_names=["content"],
            )

            top_docs = searcher.search(query=query, limit=doc_limit)

            for _, doc_addr in top_docs.hits:
                doc = searcher.doc(doc_addr)
                project_id = doc.get_first("project_id")
                memory = doc.get_first("content")
                if project_id is not None and memory is not None:
                    results.append((project_id, memory))

            return results

        except Exception as e:
            if self.logger:
                self.logger.warning(
                    "Failed to get all docs from Tantivy (all projects)",
                    extra={"error": str(e)},
                )
            return []

    def add(self, project_id: str, content: str) -> None:
        """Add a document to the Tantivy index (thread-safe).

        Thread-safe: Uses writer lock since Tantivy IndexWriter is NOT thread-safe.

        Args:
            project_id: Project identifier for filtering.
            content: Memory content to index.
        """
        with self._writer_lock:
            doc = tantivy.Document()
            doc.add_text("project_id", project_id)
            doc.add_text("content", content)

            # Add soft-delete fields with default values
            doc.add_unsigned("is_deleted", 0)  # 0 = active
            doc.add_integer("deleted_at", 0)  # 0 = not deleted

            _ = self.writer.add_document(doc)

    def commit(self) -> None:
        """Commit pending changes and refresh searcher (thread-safe).

        Thread-safe: Uses both writer and searcher locks to ensure
        consistent state during commit and searcher refresh.

        This commits all pending writes to disk and refreshes
        the searcher to see the newly added documents.

        The writer remains valid after commit() for subsequent add() calls.
        Call flush() or close() when done to wait for merging threads and
        release resources.

        Note: In tantivy-py, commit() takes a mutable reference (&mut self)
        and does NOT consume the writer. Only wait_merging_threads() consumes
        the writer by taking ownership (self). This allows reusing the same
        writer across multiple add-commit cycles for better performance.
        """
        with self._writer_lock:
            if self._writer:
                self._writer.commit()
                # NOTE: Don't call wait_merging_threads() here - it consumes the writer!
                # Writer remains valid for subsequent add() operations.
                # Background merge threads continue running asynchronously.
                if self._index:
                    self._index.reload()
                    with self._searcher_lock:
                        self._searcher = self._index.searcher()
                # Invalidate tombstone cache since searcher was refreshed
                # New tombstones may now be visible
                self._invalidate_tombstone_cache()
                if self.logger:
                    self.logger.debug(
                        "Tantivy index committed (writer reusable)",
                        extra={"project_id": self.config.project_id},
                    )

    def flush(self) -> None:
        """Commit and wait for all background merging to complete (thread-safe).

        Unlike commit(), this also waits for segment merging threads.
        The writer becomes invalid after this call and will be recreated
        on the next write operation.

        Use this when:
        - Before reading index from another process
        - Before backup operations
        - When you need guaranteed durability of all segments
        - Before application shutdown (prefer close() instead)

        For normal operations, prefer commit() which is non-blocking and
        allows writer reuse.
        """
        with self._writer_lock:
            if self._writer:
                self._writer.commit()
                self._writer.wait_merging_threads()
                self._writer = None  # Writer consumed by wait_merging_threads
                if self._index:
                    self._index.reload()
                    with self._searcher_lock:
                        self._searcher = self._index.searcher()
                # Invalidate tombstone cache since searcher was refreshed
                self._invalidate_tombstone_cache()
                if self.logger:
                    self.logger.debug(
                        "Tantivy index flushed (writer invalidated)",
                        extra={"project_id": self.config.project_id},
                    )

    def _invalidate_tombstone_cache(self, project_id: str | None = None) -> None:
        """Invalidate tombstone cache for a project or all projects.

        Thread-safe cache invalidation. Call this after any operation that
        modifies tombstones (soft_delete, commit, compact).

        Args:
            project_id: Specific project to invalidate. If None, clears entire cache.
        """
        with self._tombstone_cache_lock:
            if project_id is None:
                self._tombstone_cache.clear()
            elif project_id in self._tombstone_cache:
                del self._tombstone_cache[project_id]

    def _get_tombstoned_memories(self, project_id: str) -> set[str]:
        """Get set of memories that have tombstones for a project.

        Uses bounded in-memory caching with LRU eviction for O(1) lookup.
        Cache is populated by querying is_deleted=1 documents directly.
        When cache size exceeds tombstone_cache_max_size, oldest entries are evicted.

        Per-project tombstone sets are also bounded to prevent memory exhaustion
        in projects with many deletions (max 10000 tombstones per project).

        Args:
            project_id: Project identifier to filter by.

        Returns:
            Set of memory strings that have tombstones.
        """
        # Fast path: check cache first (thread-safe read)
        with self._tombstone_cache_lock:
            if project_id in self._tombstone_cache:
                # Move to end (most recently used)
                self._tombstone_cache.move_to_end(project_id)
                return self._tombstone_cache[project_id]

        if self._index is None:
            return set()

        try:
            doc_limit = self._get_doc_limit()

            # Cache miss: query ONLY tombstoned documents directly (is_deleted=1)
            # This is O(tombstones) instead of O(all_docs)
            # Use range syntax [1 TO 1] for numeric field exact match
            escaped_project_id = self._escape_tantivy_query(project_id)
            query = self._index.parse_query(
                query=f'project_id:"{escaped_project_id}" AND is_deleted:[1 TO 1]',
                default_field_names=["project_id"],
            )

            top_docs = self.searcher.search(query=query, limit=doc_limit)
            tombstoned: set[str] = set()
            # Per-project tombstone limit to prevent memory exhaustion
            max_tombstones_per_project = 10000

            for _, doc_addr in top_docs.hits:
                doc = self.searcher.doc(doc_addr)
                memory = doc.get_first("content")
                if memory is not None:
                    # Enforce per-project tombstone limit
                    if len(tombstoned) >= max_tombstones_per_project:
                        if self.logger:
                            self.logger.warning(
                                "Tombstone cache per-project limit reached, truncating",
                                extra={
                                    "project_id": project_id,
                                    "tombstone_count": len(tombstoned),
                                    "max_per_project": max_tombstones_per_project,
                                },
                            )
                        break
                    tombstoned.add(memory)

            # Store in cache with LRU eviction (thread-safe write)
            with self._tombstone_cache_lock:
                # Remove oldest entry if cache is at max capacity
                if len(self._tombstone_cache) >= self.config.tombstone_cache_max_size:
                    _ = self._tombstone_cache.popitem(last=False)
                # Add new entry and move to end (most recently used)
                self._tombstone_cache[project_id] = tombstoned
                # Ensure this entry is at the end (most recent)
                self._tombstone_cache.move_to_end(project_id)

            return tombstoned

        except ValueError as e:
            # Query parsing errors - expected failure, log as warning
            if self.logger:
                self.logger.warning(
                    "Failed to get tombstoned memories (query parse error)",
                    extra={"project_id": project_id, "error": str(e)},
                )
            return set()
        except Exception as e:
            # Unexpected errors - log as warning with more context
            if self.logger:
                self.logger.warning(
                    "Failed to get tombstoned memories",
                    extra={"project_id": project_id, "error": str(e)},
                )
            return set()

    def _normalize_scores(
        self, results: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        """Normalize BM25 scores to 0-1 range using batch min-max.

        Uses the existing JIT-optimized normalize_scores_minmax function.

        Args:
            results: List of (memory, score) tuples with raw BM25 scores.

        Returns:
            List of (memory, normalized_score) tuples with scores in 0-1 range.
            Single result returns score 1.0 (best by definition).
            All equal scores return 1.0 (equally good).
        """
        if len(results) <= 1:
            # Single result or empty: return with score 1.0
            return [(msg, 1.0) for msg, _ in results]

        # Import here to avoid circular dependency
        from reflectlog.utility.scoring import normalize_scores_minmax

        memories = [msg for msg, _ in results]
        scores = np.array([score for _, score in results], dtype=np.float64)

        # Use existing JIT-optimized normalization
        normalized = normalize_scores_minmax(scores)

        return list(zip(memories, normalized.tolist(), strict=True))

    def search(
        self, query: str, project_id: str, limit: int
    ) -> list[tuple[str, float]]:
        """Execute full-text search.

        Uses cached tombstone set for O(1) post-filtering after first search.
        Tombstone cache is populated lazily and invalidated on writes.

        Args:
            query: Search query string.
            project_id: Filter results by project_id.
            limit: Maximum number of results to return.

        Returns:
            List of (memory, score) tuples sorted by relevance.
            Empty list if search fails or no results found.
        """
        try:
            if self._index is None:
                if self.logger:
                    self.logger.warning(
                        "Tantivy search skipped: index not initialized",
                        extra={"project_id": self.config.project_id},
                    )
                return []

            tombstoned_memories = self._get_tombstoned_memories(project_id)
            search_limit = limit * 3 if tombstoned_memories else limit

            parsed_query = self._build_search_query(query, project_id)
            results = self._collect_search_results(
                parsed_query, search_limit, limit, tombstoned_memories
            )

            if self.config.normalize_scores and results:
                results = self._normalize_scores(results)

            return results

        except ValueError as e:
            if self.logger:
                self.logger.warning(
                    "Tantivy query parsing failed",
                    extra={
                        "project_id": self.config.project_id,
                        "query": query[:100],
                        "error": str(e),
                        "error_type": "QueryParseError",
                    },
                )
            return []

        except OSError as e:
            if self.logger:
                self.logger.error(
                    "Tantivy file system error during search",
                    extra={
                        "project_id": self.config.project_id,
                        "index_path": self.config.index_path,
                        "error": str(e),
                        "error_type": "FileSystemError",
                    },
                )
            return []

        except Exception as e:
            raise SearchError(f"Tantivy search failed: {e}") from e

    def _build_search_query(self, query: str, project_id: str) -> tantivy.Query:
        """Build and parse a Tantivy query with project_id filter.

        Falls back to escaped query text if initial parsing fails.
        """
        escaped_project_id = self._escape_tantivy_query(project_id)
        query_text = query.strip()
        if query_text:
            combined_query = f'({query_text}) AND project_id:"{escaped_project_id}"'
        else:
            combined_query = f'project_id:"{escaped_project_id}"'

        assert self._index is not None
        try:
            return self._index.parse_query(
                query=combined_query, default_field_names=["content"]
            )
        except ValueError:
            if not query_text:
                raise
            escaped_query_text = self._escape_tantivy_query(query_text)
            combined_query = (
                f'({escaped_query_text}) AND project_id:"{escaped_project_id}"'
            )
            parsed = self._index.parse_query(
                query=combined_query, default_field_names=["content"]
            )
            if self.logger:
                self.logger.debug(
                    "Escaped Tantivy query after parse failure",
                    extra={
                        "project_id": self.config.project_id,
                        "original_query": query_text[:100],
                        "escaped_query": escaped_query_text[:100],
                    },
                )
            return parsed

    def _collect_search_results(
        self,
        parsed_query: tantivy.Query,
        search_limit: int,
        result_limit: int,
        tombstoned_memories: set[str],
    ) -> list[tuple[str, float]]:
        """Execute query and collect results, filtering tombstoned memories."""
        top_docs = self.searcher.search(query=parsed_query, limit=search_limit)
        results: list[tuple[str, float]] = []

        for score, doc_addr in top_docs.hits:
            doc = self.searcher.doc(doc_addr)
            memory = doc.get_first("content")
            if not isinstance(memory, str):
                continue

            if memory in tombstoned_memories:
                continue

            if self.logger:
                self.logger.debug(f"Tantivy match: {memory[:100]}...")
            results.append((memory, score))

            if len(results) >= result_limit:
                break

        return results

    def ensure_initialized(self) -> None:
        """Ensure the engine is fully initialized (thread-safe).

        Forces the lazy initialization of the searcher to complete.
        Call this before parallel operations to prevent race conditions.
        """
        # Access the searcher property to trigger thread-safe lazy initialization
        _ = self.searcher

    def is_ready(self) -> bool:
        """Return True if the Tantivy searcher has already been created."""
        return self._searcher is not None

    def close(self) -> None:
        """Close the engine and release resources (thread-safe).

        Commits any pending changes, waits for merging threads to complete,
        and releases the index writer to free file locks.

        Call this method during shutdown to ensure clean resource cleanup.
        If not called, resources will be released by Python's garbage collector,
        but file locks may persist until GC runs.
        """
        with self._writer_lock:
            if self._writer is not None:
                try:
                    # Commit any pending changes
                    self._writer.commit()
                    # Wait for background merge operations to complete
                    self._writer.wait_merging_threads()
                except Exception as e:
                    if self.logger:
                        self.logger.warning(
                            "Error during Tantivy writer cleanup",
                            extra={
                                "project_id": self.config.project_id,
                                "error": str(e),
                            },
                        )
                finally:
                    # Release the writer reference to allow GC
                    self._writer = None

            with self._searcher_lock:
                # Release searcher reference
                self._searcher = None

            # Clear index reference INSIDE writer_lock to prevent race with search()
            # Note: Index itself doesn't need explicit close in Tantivy Python bindings
            # but we clear the reference for consistency
            self._index = None

    def soft_delete(self, project_id: str, content: str) -> bool:
        """Mark a document as deleted by adding a tombstone (O(1) operation).

        Thread-safe. This is much faster than rebuilding the entire index (O(n) for rebuild vs O(1)).

        The tombstone approach:
        - Original document remains in index (is_deleted=0, deleted_at=0)
        - Tombstone document added (is_deleted=1, deleted_at=<timestamp>)
        - Search filters out memories that have tombstones
        - Compaction removes both original and tombstone

        Args:
            project_id: Project identifier for filtering.
            content: Exact memory content to soft-delete.

        Returns:
            True if tombstone was added, False if memory wasn't found.
        """
        # First, verify the memory exists (only check non-deleted documents)
        existing = self.find_by_exact_match(project_id, content)
        if not existing:
            if self.logger:
                self.logger.debug(
                    "Soft-delete: memory not found",
                    extra={
                        "project_id": project_id,
                        "memory_preview": content[:50] if content else "",
                    },
                )
            return False

        # Add tombstone document
        with self._writer_lock:
            doc = tantivy.Document()
            doc.add_text("project_id", project_id)
            doc.add_text("content", content)
            doc.add_unsigned("is_deleted", 1)  # 1 = deleted (tombstone)
            doc.add_integer("deleted_at", int(time.time() * 1000))  # Unix timestamp ms

            _ = self.writer.add_document(doc)

        if self.logger:
            self.logger.debug(
                "Soft-delete: tombstone added",
                extra={
                    "project_id": project_id,
                    "memory_preview": content[:50] if content else "",
                },
            )

        return True

    def delete(self, project_id: str, content: str) -> bool:
        """Delete a document from the Tantivy index by exact memory match (thread-safe).

        When soft-delete is enabled (default), uses O(1) tombstone marking.
        Otherwise, falls back to O(n) rebuild approach.

        Soft-delete approach:
        - Adds a tombstone document with is_deleted=1
        - Search automatically filters out tombstoned memories
        - Compaction removes tombstones and originals periodically

        Rebuild approach (when soft-delete disabled):
        - Gets all documents from all projects
        - Filters out the target memory
        - Destroys and recreates the index
        - Re-adds all remaining documents

        Thread-safe: Uses writer lock since Tantivy IndexWriter is NOT thread-safe.

        Args:
            project_id: Project identifier for filtering.
            content: Exact memory content to delete.

        Returns:
            True if document was found and deleted, False otherwise.
        """
        # Use soft-delete if enabled (O(1) vs O(n))
        if self.config.soft_delete_enabled:
            result = self.soft_delete(project_id, content)
            if result:
                self.commit()
            return result

        # Fall back to rebuild approach when soft-delete disabled
        try:
            return self._delete_via_rebuild(project_id, content)
        except ValueError as e:
            self._log_delete_error(project_id, content, e, "warning", "QueryParseError")
            return False
        except OSError as e:
            self._log_delete_error(
                project_id,
                content,
                e,
                "error",
                "FileSystemError",
                include_index_path=True,
            )
            return False
        except Exception as e:
            self._log_delete_error(project_id, content, e, "error", type(e).__name__)
            return False

    def _delete_via_rebuild(self, project_id: str, content: str) -> bool:
        """Delete a document using the full index rebuild approach."""
        if self._index is None:
            return False

        all_docs = self._get_all_docs_all_projects()

        target_exists = any(
            pid == project_id and msg == content for pid, msg in all_docs
        )
        if not target_exists:
            if self.logger:
                self.logger.debug(
                    "Tantivy delete: document not found",
                    extra={
                        "project_id": project_id,
                        "memory_preview": content[:50] if content else "",
                    },
                )
            return False

        docs_to_keep = [
            (pid, msg)
            for pid, msg in all_docs
            if not (pid == project_id and msg == content)
        ]
        deleted_count = len(all_docs) - len(docs_to_keep)

        if self.logger:
            self.logger.debug(
                "Tantivy delete: found document(s) to delete",
                extra={
                    "project_id": project_id,
                    "memory_preview": content[:50] if content else "",
                    "deleted_count": deleted_count,
                    "remaining_count": len(docs_to_keep),
                },
            )

        self._rebuild_index_with_docs(docs_to_keep)

        if self.logger:
            self.logger.debug(
                "Tantivy delete: completed successfully",
                extra={
                    "project_id": project_id,
                    "memory_preview": content[:50] if content else "",
                    "deleted_count": deleted_count,
                },
            )

        return True

    def _log_delete_error(
        self,
        project_id: str,
        content: str,
        error: Exception,
        level: str,
        error_type: str,
        *,
        include_index_path: bool = False,
    ) -> None:
        """Log a delete operation error at the specified level."""
        if not self.logger:
            return

        extra: dict[str, object] = {
            "project_id": project_id,
            "memory_preview": content[:50] if content else "",
            "error": str(error),
            "error_type": error_type,
        }
        if include_index_path:
            extra["index_path"] = self.config.index_path

        msg = (
            "Tantivy delete query parsing failed"
            if level == "warning"
            else "Tantivy delete failed unexpectedly"
            if error_type != "FileSystemError"
            else "Tantivy delete file system error"
        )
        log_fn = self.logger.warning if level == "warning" else self.logger.error
        log_fn(msg, extra=extra)

    def _rebuild_index_with_docs(self, docs_to_keep: list[tuple[str, str]]) -> None:
        """Rebuild the index with the specified documents from all projects.

        This destroys the existing index and creates a new one with all
        the documents in docs_to_keep. This is necessary because Tantivy-py's
        delete_documents() consumes the IndexWriter.

        Args:
            docs_to_keep: List of (project_id, content) tuples to preserve.
        """
        import shutil

        index_path = self.config.index_path

        # Step 1: Properly finalize existing writer before destroying index
        with self._writer_lock:
            if self._writer is not None:
                try:
                    self._writer.commit()
                    self._writer.wait_merging_threads()  # Wait before destroying
                except Exception:
                    pass  # Best effort - index will be destroyed anyway
                self._writer = None

        with self._searcher_lock:
            self._searcher = None
        self._index = None

        # Clear tombstone cache since index is being destroyed
        self._invalidate_tombstone_cache()

        # Step 2: Delete the index directory
        if os.path.exists(index_path):
            shutil.rmtree(index_path)

        # Step 3: Recreate the index
        self._initialize_index()

        # Step 4: Re-add the kept documents (all projects)
        for project_id, content in docs_to_keep:
            self.add(project_id, content)

        # Step 5: Commit the changes
        self.commit()

    @staticmethod
    def _escape_tantivy_query(query: str) -> str:
        """Escape special characters for Tantivy query syntax.

        Args:
            query: Raw query string that may contain special characters.

        Returns:
            Escaped query string safe for use in Tantivy queries.
        """
        special_chars = r'+-&|!(){}[]^"~*?:\/'
        escaped: list[str] = []
        for char in query:
            if char in special_chars:
                escaped.append(f"\\{char}")
            else:
                escaped.append(char)
        return "".join(escaped)

    def _get_doc_limit(self) -> int:
        """Get total document count for safe full scans.

        Uses Tantivy's searcher num_docs when available to avoid truncation.
        Falls back to a conservative limit if not accessible.
        """
        if self._index is None:
            return 0

        searcher = self.searcher
        num_docs: int | None = None

        try:
            num_docs_attr = getattr(searcher, "num_docs", None)
            if callable(num_docs_attr):
                num_docs_result = num_docs_attr()
                if isinstance(num_docs_result, (int, float)):
                    num_docs = int(num_docs_result)
            elif isinstance(num_docs_attr, (int, float)):
                num_docs = int(num_docs_attr)
        except Exception as e:
            if self.logger:
                self.logger.debug(
                    "Failed to read Tantivy num_docs",
                    extra={"error": str(e)},
                )

        if num_docs is None or num_docs < 0:
            if self.logger:
                self.logger.warning(
                    "Tantivy searcher num_docs unavailable; using fallback limit",
                    extra={"fallback_limit": DEFAULT_TANTIVY_DOC_LIMIT},
                )
            return DEFAULT_TANTIVY_DOC_LIMIT

        return max(1, num_docs)

    def get_tombstone_stats(self) -> dict[str, int]:
        """Get statistics about tombstones in the index.

        Returns dictionary with:
        - total_docs: Total documents in index (including tombstones)
        - active_docs: Documents with is_deleted=0
        - tombstones: Documents with is_deleted=1
        - unique_active_memories: Unique memories after filtering tombstones
        """
        if self._index is None:
            return {
                "total_docs": 0,
                "active_docs": 0,
                "tombstones": 0,
                "unique_active_memories": 0,
            }

        try:
            doc_limit = self._get_doc_limit()

            # Get all documents across all projects
            query = self._index.parse_query(
                query="*",
                default_field_names=["content"],
            )
            top_docs = self.searcher.search(query=query, limit=doc_limit)

            total_docs = 0
            active_docs = 0
            tombstones = 0
            active_memories: set[str] = set()
            tombstoned_memories: set[str] = set()

            for _, doc_addr in top_docs.hits:
                doc = self.searcher.doc(doc_addr)
                total_docs += 1
                memory = doc.get_first("content")

                is_deleted_val = doc.get_first("is_deleted")
                is_deleted = int(is_deleted_val) if is_deleted_val is not None else 0
                if is_deleted == 1:
                    tombstones += 1
                    if memory is not None:
                        tombstoned_memories.add(memory)
                else:
                    active_docs += 1
                    if memory is not None:
                        active_memories.add(memory)

            # Unique active memories = active memories NOT in tombstoned set
            unique_active = active_memories - tombstoned_memories

            return {
                "total_docs": total_docs,
                "active_docs": active_docs,
                "tombstones": tombstones,
                "unique_active_memories": len(unique_active),
            }

        except Exception as e:
            if self.logger:
                self.logger.warning(
                    "Failed to get tombstone stats",
                    extra={"error": str(e)},
                )
            return {
                "total_docs": 0,
                "active_docs": 0,
                "tombstones": 0,
                "unique_active_memories": 0,
            }

    def needs_compaction(self) -> bool:
        """Check if the index needs compaction based on configured thresholds.

        Compaction is needed when:
        - Tombstones > compaction_max_tombstones (absolute threshold), OR
        - Tombstones / total_docs > compaction_threshold_ratio (percentage threshold)
        """
        stats = self.get_tombstone_stats()
        tombstones = stats["tombstones"]
        total_docs = stats["total_docs"]

        # Check absolute threshold
        if tombstones >= self.config.compaction_max_tombstones:
            if self.logger:
                self.logger.info(
                    "Compaction needed: tombstone count exceeded threshold",
                    extra={
                        "tombstones": tombstones,
                        "threshold": self.config.compaction_max_tombstones,
                    },
                )
            return True

        # Check ratio threshold
        if total_docs > 0:
            ratio = tombstones / total_docs
            if ratio >= self.config.compaction_threshold_ratio:
                if self.logger:
                    self.logger.info(
                        "Compaction needed: tombstone ratio exceeded threshold",
                        extra={
                            "tombstones": tombstones,
                            "total_docs": total_docs,
                            "ratio": round(ratio, 3),
                            "threshold": self.config.compaction_threshold_ratio,
                        },
                    )
                return True

        return False

    def compact(self, force: bool = False) -> dict[str, int]:
        """Compact the index by removing tombstoned memories.

        This physically removes both tombstone documents (is_deleted=1) and their
        corresponding original documents (is_deleted=0) from the index. After
        compaction, only truly active memories remain.

        Args:
            force: If True, compact even if thresholds aren't exceeded.
                   Default False (only compact if needed).

        Returns:
            Dictionary with compaction statistics:
            - compacted: Whether compaction was performed
            - removed_tombstones: Number of tombstone documents removed
            - removed_originals: Number of original documents removed (had tombstones)
            - remaining_docs: Total documents after compaction
            - elapsed_ms: Time taken in milliseconds

        When compaction not needed (and force=False), returns compacted=False
        with all other values as 0.
        """
        start_time = time.time()

        if not force and not self.needs_compaction():
            return self._compaction_result(compacted=False)

        try:
            stats_before = self.get_tombstone_stats()

            index = self._index
            if index is None:
                return self._compaction_result(compacted=False)

            docs_to_keep, removed_tombstones, removed_originals = (
                self._collect_active_docs(index)
            )

            self._rebuild_index_with_docs(docs_to_keep)
            self._try_garbage_collect()

            elapsed_ms = int((time.time() - start_time) * 1000)

            if self.logger:
                self.logger.info(
                    "Tantivy compaction completed",
                    extra={
                        "removed_tombstones": removed_tombstones,
                        "removed_originals": removed_originals,
                        "remaining_docs": len(docs_to_keep),
                        "elapsed_ms": elapsed_ms,
                        "docs_before": stats_before["total_docs"],
                    },
                )

            return self._compaction_result(
                compacted=True,
                removed_tombstones=removed_tombstones,
                removed_originals=removed_originals,
                remaining_docs=len(docs_to_keep),
                elapsed_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            if self.logger:
                self.logger.error(
                    "Tantivy compaction failed",
                    extra={"error": str(e), "elapsed_ms": elapsed_ms},
                )
            return self._compaction_result(compacted=False, elapsed_ms=elapsed_ms)

    @staticmethod
    def _compaction_result(
        *,
        compacted: bool,
        removed_tombstones: int = 0,
        removed_originals: int = 0,
        remaining_docs: int = 0,
        elapsed_ms: int = 0,
    ) -> dict[str, int]:
        """Build a compaction statistics dictionary."""
        return {
            "compacted": compacted,
            "removed_tombstones": removed_tombstones,
            "removed_originals": removed_originals,
            "remaining_docs": remaining_docs,
            "elapsed_ms": elapsed_ms,
        }

    def _collect_active_docs(
        self, index: tantivy.Index
    ) -> tuple[list[tuple[str, str]], int, int]:
        """Collect active documents, filtering out tombstoned entries.

        Returns:
            Tuple of (docs_to_keep, removed_tombstones, removed_originals).
        """
        query = index.parse_query(
            query="*",
            default_field_names=["content"],
        )
        top_docs = self.searcher.search(query=query, limit=100000)

        all_docs: list[tuple[str, str, int]] = []
        tombstoned_memories: set[str] = set()

        for _, doc_addr in top_docs.hits:
            doc = self.searcher.doc(doc_addr)
            project_id = doc.get_first("project_id")
            memory = doc.get_first("content")
            is_deleted_val = doc.get_first("is_deleted")
            is_deleted = int(is_deleted_val) if is_deleted_val is not None else 0

            if project_id is not None and memory is not None:
                all_docs.append((project_id, memory, is_deleted))
                if is_deleted == 1:
                    tombstoned_memories.add(memory)

        docs_to_keep: list[tuple[str, str]] = []
        removed_originals = 0
        removed_tombstones = 0

        for project_id, memory, is_deleted in all_docs:
            if memory in tombstoned_memories:
                if is_deleted == 1:
                    removed_tombstones += 1
                else:
                    removed_originals += 1
            else:
                docs_to_keep.append((project_id, memory))

        return docs_to_keep, removed_tombstones, removed_originals

    def _try_garbage_collect(self) -> None:
        """Best-effort garbage collection of old segment files."""
        try:
            if self._writer is not None:
                self._writer.garbage_collect_files()
        except Exception:
            pass  # Files will be cleaned up eventually
