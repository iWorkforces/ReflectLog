"""Tantivy full-text search engine wrapper.

This module provides a wrapper around the Tantivy full-text search library,
following the same patterns as qwen3_embedding.py for consistency.
"""

from collections import OrderedDict
from collections.abc import Generator
from contextlib import contextmanager
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

from reflectlog.core.config import IAppConfig
from reflectlog.core.exceptions import InitializationError, SearchError
from reflectlog.core.logging import IStructuredLogger
from reflectlog.core.storage_coordination import IStorageCoordinator, LeaseMode


def _is_dict_config(config: object) -> TypeGuard[dict[str, Any]]:
    """Type guard to check if config is a dict."""
    return isinstance(config, dict)


@dataclass(frozen=True)
class TantivyConfig:
    """Configuration for TantivyEngine.

    Attributes:
        workspace_id: Unique workspace identifier for filtering.
        index_path: Path to the Tantivy index directory.
        soft_delete_enabled: Use O(1) tombstone marking instead of O(n) rebuild.
        compaction_threshold_ratio: Compact when tombstones > this ratio of docs.
        compaction_max_tombstones: Force compaction above this tombstone count.
        tombstone_ttl_days: Days before tombstones are eligible for removal.
        tombstone_cache_max_size: Maximum number of workspace IDs to cache tombstones for.
        normalize_scores: Normalize BM25 scores to 0-1 range (batch min-max).
    """

    workspace_id: str
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
            workspace_id=data.get("workspace_id", "") or "",
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

    @classmethod
    def from_config(cls, config: IAppConfig) -> TantivyConfig:
        """Create TantivyConfig from IAppConfig protocol."""
        return cls(
            workspace_id=config.workspace_id,
            index_path=config.tantivy_index_path,
            normalize_scores=config.tantivy_normalize_scores,
            soft_delete_enabled=config.tantivy_soft_delete_enabled,
            compaction_threshold_ratio=config.tantivy_compaction_threshold_ratio,
            compaction_max_tombstones=config.tantivy_compaction_max_tombstones,
            tombstone_ttl_days=config.tantivy_tombstone_ttl_days,
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
    - Workspace-level filtering via workspace_id field
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: TantivyConfig
    logger: IStructuredLogger | None = None
    coordinator: IStorageCoordinator | None = None

    _index: tantivy.Index | None = PrivateAttr(default=None)
    _writer: tantivy.IndexWriter | None = PrivateAttr(default=None)
    _searcher: tantivy.Searcher | None = PrivateAttr(default=None)
    _lease_depth: int = PrivateAttr(default=0)
    _seen_generation: int = PrivateAttr(default=0)
    # Instance-level locks for thread-safe operations
    # Note: Using RLock (re-entrant) because add() holds lock and
    # calls self.writer property
    _writer_lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)
    # Bounded in-memory tombstone cache for O(1) lookup
    # after first search
    # Uses OrderedDict for LRU eviction when size exceeds tombstone_cache_max_size
    # Key: workspace_id, Value: set of tombstoned memory contents
    _tombstone_cache: OrderedDict[str, set[str]] = PrivateAttr(
        default_factory=lambda: OrderedDict[str, set[str]]()
    )
    _tombstone_cache_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _searcher_lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)
    _closed: bool = PrivateAttr(default=False)

    def __init__(
        self,
        config: TantivyConfig | dict[str, Any],
        logger: IStructuredLogger | None = None,
        **kwargs: Any,
    ):
        """Initialize TantivyEngine.

        Args:
            config: TantivyConfig or dict with workspace_id and index_path.
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
        """Build Tantivy schema with workspace_id, content, and soft-delete fields.

        Returns:
            Tantivy schema with:
            - workspace_id: Stored text field with raw tokenizer (exact match)
            - content: Stored text field with en_stem tokenizer (full-text search)
            - is_deleted: Stored unsigned field for soft-delete (0=active, 1=deleted)
            - deleted_at: Stored integer field for deletion timestamp in ms
        """
        schema_builder = tantivy.SchemaBuilder()
        _ = schema_builder.add_text_field(
            "workspace_id", stored=True, tokenizer_name="raw"
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

    def _rebuild_backup_path(self, index_path: str) -> str:
        """Return the sibling rebuild-backup directory for ``index_path``."""
        return f"{index_path}.rebuild-bak"

    def _index_is_openable(self, index_path: str) -> bool:
        """Return True when Tantivy can open the directory as an index."""
        return self._index_num_docs(index_path) is not None

    def _index_num_docs(self, index_path: str) -> int | None:
        """Return committed document count, or None when the path is unopenable."""
        try:
            index = tantivy.Index.open(index_path)
            return int(index.searcher().num_docs())
        except Exception:
            return None

    def _restore_rebuild_backup_if_needed(self, index_path: str) -> None:
        """Move leftover rebuild backup into place when the live index is unusable.

        Restore when live is missing, unopenable, or an empty schema left by a
        crashed rebuild. Never destroy a populated live index, and never move
        a backup that cannot be opened.
        """
        import shutil

        backup_path = self._rebuild_backup_path(index_path)
        if not os.path.exists(backup_path):
            return
        live_docs = self._index_num_docs(index_path)
        if live_docs is not None and live_docs > 0:
            return
        bak_meta = os.path.join(backup_path, "meta.json")
        if not os.path.exists(bak_meta):
            return
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
        _ = shutil.move(backup_path, index_path)

    def _initialize_index(self, *, restore_backup: bool = True) -> None:
        """Initialize or load persistent Tantivy index.

        Creates the index directory if it doesn't exist.
        Attempts to load existing index, creates new one if not found.

        Raises:
            RuntimeError: If index creation fails.
        """
        index_path = self.config.index_path
        os.makedirs(index_path, exist_ok=True)
        if restore_backup:
            self._restore_rebuild_backup_if_needed(index_path)

        try:
            # Try to open existing index
            self._index = tantivy.Index.open(index_path)
            if self.logger:
                self.logger.info(
                    "Loaded existing Tantivy index",
                    extra={
                        "workspace_id": self.config.workspace_id,
                        "tantivy_index_path": index_path,
                    },
                )
        except Exception as open_error:
            if restore_backup:
                self._restore_rebuild_backup_if_needed(index_path)
                try:
                    self._index = tantivy.Index.open(index_path)
                    if self.logger:
                        self.logger.info(
                            "Restored Tantivy index from rebuild backup",
                            extra={
                                "workspace_id": self.config.workspace_id,
                                "tantivy_index_path": index_path,
                            },
                        )
                    return
                except Exception:
                    pass
            meta_path = os.path.join(index_path, "meta.json")
            if os.path.exists(meta_path):
                raise InitializationError(
                    f"Failed to open existing Tantivy index at {index_path}"
                ) from open_error
            schema = self._build_schema()
            self._index = tantivy.Index(schema, path=index_path, reuse=True)
            if self.logger:
                self.logger.info(
                    "Created new Tantivy index",
                    extra={
                        "workspace_id": self.config.workspace_id,
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

    @contextmanager
    def _lease(self, mode: LeaseMode) -> Generator[None]:
        """Acquire a coordinator lease, reusing a nest already held by this engine."""
        coordinator = self.coordinator
        if coordinator is None or self._lease_depth > 0:
            self._lease_depth += 1
            try:
                yield
            finally:
                self._lease_depth -= 1
            return
        with coordinator.acquire(self.config.workspace_id, mode):
            self._lease_depth += 1
            try:
                yield
            finally:
                self._lease_depth -= 1

    def _refresh_reader(self) -> None:
        """Reload committed segments so an external writer becomes visible."""
        if self._index is None:
            return
        self._index.reload()
        with self._searcher_lock:
            self._searcher = self._index.searcher()
        coordinator = self.coordinator
        if coordinator is not None:
            self._seen_generation = coordinator.read_generation(
                self.config.workspace_id
            )

    def _finalize_writer(self, *, relinquish: bool) -> None:
        """Commit the writer, optionally wait/drop it, then refresh the reader."""
        if self._writer is None:
            return
        self._writer.commit()
        if relinquish:
            self._writer.wait_merging_threads()
            self._writer = None
        if self._index is not None:
            self._index.reload()
            with self._searcher_lock:
                self._searcher = self._index.searcher()
        self._invalidate_tombstone_cache()

    def _rewrite_index_in_place(self, docs_to_keep: list[tuple[str, str]]) -> None:
        """Replace live documents without swapping the index directory."""
        with self._lease(LeaseMode.EXCLUSIVE):
            with self._writer_lock:
                writer = self.writer
                writer.delete_all_documents()
                for workspace_id, content in docs_to_keep:
                    doc = tantivy.Document()
                    doc.add_text("workspace_id", workspace_id)
                    doc.add_text("content", content)
                    doc.add_unsigned("is_deleted", 0)
                    doc.add_integer("deleted_at", 0)
                    _ = writer.add_document(doc)
                self._finalize_writer(relinquish=True)

    def _get_all_docs(self, workspace_id: str) -> list[str]:
        """Get all active (non-tombstoned) documents for a workspace.

        Uses cached tombstone set for O(1) post-filtering after first call.

        Args:
            workspace_id: Workspace identifier to filter by.

        Returns:
            List of memory strings for the given project (excluding tombstoned).
        """
        self._require_open()
        if self._index is None:
            return []

        try:
            pinned = self.searcher
            # Get cached tombstoned memories (O(1) after first call)
            tombstoned_memories = self._get_tombstoned_memories(
                workspace_id, searcher=pinned
            )
            doc_limit = self._get_doc_limit()

            # Query all docs for this project
            escaped_workspace_id = self._escape_tantivy_query(workspace_id)
            query = self._index.parse_query(
                query=f'workspace_id:"{escaped_workspace_id}"',
                default_field_names=["workspace_id"],
            )

            # Get all results (use index doc count to avoid truncation)
            top_docs = pinned.search(query=query, limit=doc_limit)

            results: list[str] = []
            seen: set[str] = set()  # Track seen memories to avoid duplicates

            for _, doc_addr in top_docs.hits:
                doc = pinned.doc(doc_addr)
                memory = doc.get_first("content")
                if memory is not None:
                    msg_str = memory
                    # Skip tombstoned memories and duplicates
                    is_deleted_val = doc.get_first("is_deleted")
                    is_deleted = (
                        int(is_deleted_val) if is_deleted_val is not None else 0
                    )
                    if is_deleted == 1:
                        continue
                    if msg_str not in tombstoned_memories and msg_str not in seen:
                        results.append(msg_str)
                        seen.add(msg_str)

            return results

        except Exception as e:
            if self.logger:
                self.logger.warning(
                    "Failed to get all docs from Tantivy",
                    extra={
                        "workspace_id": workspace_id,
                        "error": str(e),
                    },
                )
            raise RuntimeError(
                f"Failed to get all docs from Tantivy: {e}"
            ) from e

    def find_by_exact_match(self, workspace_id: str, content: str) -> list[str]:
        """Find all memories that exactly match the given memory text.

        Uses Python-level string comparison after fetching all docs for the project.
        This works around the en_stem tokenizer's stemming behavior which prevents
        exact phrase matching via Tantivy queries.

        Args:
            workspace_id: Workspace identifier to filter by.
            content: Exact memory text to find.

        Returns:
            List of matching memory strings (may contain duplicates if stored multiple times).
        """
        with self._lease(LeaseMode.SHARED):
            self._refresh_reader()
            all_docs = self._get_all_docs(workspace_id)
            matches = [doc for doc in all_docs if doc == content]
            if self.coordinator is not None:
                with self._searcher_lock:
                    self._searcher = None
            return matches

    def _get_all_docs_all_workspaces(self) -> list[tuple[str, str]]:
        """Get all documents from all workspaces.

        Returns:
            List of (workspace_id, content) tuples for all documents in the index.
        """
        if self._index is None:
            return []

        try:
            doc_limit = self._get_doc_limit()

            # Use match-all query by searching for common patterns
            # Tantivy doesn't have a built-in match-all, so we use the searcher directly
            searcher = self.searcher

            # Use a query that matches everything - workspace_id always exists
            # Search for workspace_id:* doesn't work, so we'll try multiple common project patterns
            # Actually, let's use the searcher's doc method with addresses

            # Alternative approach: search for any document with workspace_id field
            # We'll use a very broad search since all docs have workspace_id
            query = self._index.parse_query(
                query="*",
                default_field_names=["content"],
            )

            top_docs = searcher.search(query=query, limit=doc_limit)

            live: list[tuple[str, str]] = []
            live_counts: dict[tuple[str, str], int] = {}
            tomb_counts: dict[tuple[str, str], int] = {}
            for _, doc_addr in top_docs.hits:
                doc = searcher.doc(doc_addr)
                workspace_id = doc.get_first("workspace_id")
                memory = doc.get_first("content")
                if workspace_id is None or memory is None:
                    continue
                is_deleted_val = doc.get_first("is_deleted")
                is_deleted = int(is_deleted_val) if is_deleted_val is not None else 0
                key = (workspace_id, memory)
                if is_deleted == 1:
                    tomb_counts[key] = tomb_counts.get(key, 0) + 1
                    continue
                live.append(key)
                live_counts[key] = live_counts.get(key, 0) + 1

            kept: list[tuple[str, str]] = []
            kept_live: dict[tuple[str, str], int] = {}
            for item in live:
                surplus = live_counts.get(item, 0) - tomb_counts.get(item, 0)
                if surplus <= 0 or kept_live.get(item, 0) >= surplus:
                    continue
                kept_live[item] = kept_live.get(item, 0) + 1
                kept.append(item)
            return kept

        except Exception as e:
            if self.logger:
                self.logger.warning(
                    "Failed to get all docs from Tantivy (all workspaces)",
                    extra={"error": str(e)},
                )
            return []

    def add(self, workspace_id: str, content: str) -> None:
        """Add a document to the Tantivy index (thread-safe).

        Thread-safe: Uses writer lock since Tantivy IndexWriter is NOT thread-safe.

        Args:
            workspace_id: Workspace identifier for filtering.
            content: Memory content to index.
        """
        self._require_open()
        with self._lease(LeaseMode.EXCLUSIVE):
            with self._writer_lock:
                doc = tantivy.Document()
                doc.add_text("workspace_id", workspace_id)
                doc.add_text("content", content)

                # Add soft-delete fields with default values
                doc.add_unsigned("is_deleted", 0)  # 0 = active
                doc.add_integer("deleted_at", 0)  # 0 = not deleted

                _ = self.writer.add_document(doc)
                if self.coordinator is not None:
                    self._finalize_writer(relinquish=True)

    def add_batch(self, workspace_id: str, contents: list[str]) -> None:
        """Add multiple documents under a single writer lock."""
        if not contents:
            return
        self._require_open()
        with self._lease(LeaseMode.EXCLUSIVE):
            with self._writer_lock:
                for content in contents:
                    doc = tantivy.Document()
                    doc.add_text("workspace_id", workspace_id)
                    doc.add_text("content", content)
                    doc.add_unsigned("is_deleted", 0)
                    doc.add_integer("deleted_at", 0)
                    _ = self.writer.add_document(doc)
                if self.coordinator is not None:
                    self._finalize_writer(relinquish=True)

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
        with self._lease(LeaseMode.EXCLUSIVE):
            with self._writer_lock:
                if self._writer:
                    relinquish = self.coordinator is not None
                    self._finalize_writer(relinquish=relinquish)
                    if self.logger:
                        self.logger.debug(
                            "Tantivy index committed (writer reusable)"
                            if not relinquish
                            else "Tantivy index committed (writer relinquished)",
                            extra={"workspace_id": self.config.workspace_id},
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
        with self._lease(LeaseMode.EXCLUSIVE):
            with self._writer_lock:
                if self._writer:
                    self._finalize_writer(relinquish=True)
                    if self.logger:
                        self.logger.debug(
                            "Tantivy index flushed (writer invalidated)",
                            extra={"workspace_id": self.config.workspace_id},
                        )

    def _invalidate_tombstone_cache(self, workspace_id: str | None = None) -> None:
        """Invalidate tombstone cache for a workspace or all workspaces.

        Thread-safe cache invalidation. Call this after any operation that
        modifies tombstones (soft_delete, commit, compact).

        Args:
            workspace_id: Specific project to invalidate. If None, clears entire cache.
        """
        with self._tombstone_cache_lock:
            if workspace_id is None:
                self._tombstone_cache.clear()
            elif workspace_id in self._tombstone_cache:
                del self._tombstone_cache[workspace_id]

    def _get_tombstoned_memories(
        self,
        workspace_id: str,
        searcher: tantivy.Searcher | None = None,
    ) -> set[str]:
        """Get set of memories that have tombstones for a workspace.

        Uses bounded in-memory caching with LRU eviction for O(1) lookup.
        Cache is populated by querying is_deleted=1 documents directly.
        When cache size exceeds tombstone_cache_max_size, oldest entries are evicted.

        Per-project tombstone sets are also bounded to prevent memory exhaustion
        in projects with many deletions (max 10000 tombstones per project).

        Args:
            workspace_id: Workspace identifier to filter by.
            searcher: Optional pinned searcher so a concurrent commit cannot
                swap the reader mid-scan.

        Returns:
            Set of memory strings that have tombstones.
        """
        use_cache = searcher is None or searcher is self._searcher
        if use_cache:
            with self._tombstone_cache_lock:
                if workspace_id in self._tombstone_cache:
                    # Move to end (most recently used)
                    self._tombstone_cache.move_to_end(workspace_id)
                    return set(self._tombstone_cache[workspace_id])

        if self._index is None:
            return set()

        try:
            doc_limit = self._get_doc_limit()
            pinned = searcher if searcher is not None else self.searcher

            # A text is dead only when tombstones outnumber live copies.
            # That survives process restart (delete + re-add stays visible).
            escaped_workspace_id = self._escape_tantivy_query(workspace_id)
            query = self._index.parse_query(
                query=f'workspace_id:"{escaped_workspace_id}"',
                default_field_names=["workspace_id"],
            )

            top_docs = pinned.search(query=query, limit=doc_limit)
            live_counts: dict[str, int] = {}
            tomb_counts: dict[str, int] = {}

            for _, doc_addr in top_docs.hits:
                doc = pinned.doc(doc_addr)
                memory = doc.get_first("content")
                if memory is None:
                    continue
                is_deleted_val = doc.get_first("is_deleted")
                is_deleted = int(is_deleted_val) if is_deleted_val is not None else 0
                if is_deleted == 1:
                    tomb_counts[memory] = tomb_counts.get(memory, 0) + 1
                else:
                    live_counts[memory] = live_counts.get(memory, 0) + 1

            tombstoned = {
                memory
                for memory, tombs in tomb_counts.items()
                if tombs >= live_counts.get(memory, 0)
            }

            if searcher is None or searcher is self._searcher:
                with self._tombstone_cache_lock:
                    if len(self._tombstone_cache) >= self.config.tombstone_cache_max_size:
                        _ = self._tombstone_cache.popitem(last=False)
                    self._tombstone_cache[workspace_id] = tombstoned
                    self._tombstone_cache.move_to_end(workspace_id)

            return tombstoned

        except OSError:
            raise
        except ValueError as e:
            if self.logger:
                self.logger.warning(
                    "Failed to get tombstoned memories (query parse error)",
                    extra={"workspace_id": workspace_id, "error": str(e)},
                )
            raise RuntimeError(
                f"Failed to get tombstoned memories: {e}"
            ) from e
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    "Failed to get tombstoned memories",
                    extra={"workspace_id": workspace_id, "error": str(e)},
                )
            raise RuntimeError(
                f"Failed to get tombstoned memories: {e}"
            ) from e

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
        if float(np.max(scores)) == float(np.min(scores)):
            return [(msg, float(score)) for msg, score in results]

        # Use existing JIT-optimized normalization
        normalized = normalize_scores_minmax(scores)

        return list(zip(memories, normalized.tolist(), strict=True))

    def _search_unlocked(
        self, query: str, workspace_id: str, limit: int
    ) -> list[tuple[str, float]]:
        """Search after the caller has acquired a shared lease."""
        if self._index is None:
            if self.logger:
                self.logger.warning(
                    "Tantivy search skipped: index not initialized",
                    extra={"workspace_id": self.config.workspace_id},
                )
            return []

        if not query.strip():
            return []

        if self.coordinator is not None:
            self._refresh_reader()
        pinned = self.searcher
        tombstoned_memories = self._get_tombstoned_memories(
            workspace_id, searcher=pinned
        )
        extra = 2 * len(tombstoned_memories)
        search_limit = limit + extra

        parsed_query = self._build_search_query(query, workspace_id)
        results = self._collect_search_results(
            parsed_query,
            search_limit,
            limit,
            tombstoned_memories,
            searcher=pinned,
        )

        if self.config.normalize_scores and results:
            results = self._normalize_scores(results)

        if self.coordinator is not None:
            with self._searcher_lock:
                self._searcher = None

        return results

    def search(
        self, query: str, workspace_id: str, limit: int
    ) -> list[tuple[str, float]]:
        """Execute full-text search.

        Uses cached tombstone set for O(1) post-filtering after first search.
        Tombstone cache is populated lazily and invalidated on writes.

        Args:
            query: Search query string.
            workspace_id: Filter results by workspace_id.
            limit: Maximum number of results to return.

        Returns:
            List of (memory, score) tuples sorted by relevance.
            Empty list if search fails or no results found.
        """
        try:
            self._require_open()
            with self._lease(LeaseMode.SHARED):
                return self._search_unlocked(query, workspace_id, limit)

        except ValueError as e:
            if self.logger:
                self.logger.warning(
                    "Tantivy query parsing failed",
                    extra={
                        "workspace_id": self.config.workspace_id,
                        "query_length": len(query),
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
                        "workspace_id": self.config.workspace_id,
                        "index_path": self.config.index_path,
                        "error": str(e),
                        "error_type": "FileSystemError",
                    },
                )
            raise SearchError(f"Tantivy file system error during search: {e}") from e

        except SearchError:
            raise
        except Exception as e:
            raise SearchError(f"Tantivy search failed: {e}") from e

    def _build_search_query(self, query: str, workspace_id: str) -> tantivy.Query:
        """Build and parse a Tantivy query with workspace_id filter.

        Falls back to escaped query text if initial parsing fails.
        """
        escaped_workspace_id = self._escape_tantivy_query(workspace_id)
        query_text = query.strip()
        if not query_text:
            raise SearchError("Tantivy query must contain non-whitespace characters")
        escaped_query_text = self._escape_tantivy_query(query_text)
        combined_query = (
            f'({escaped_query_text}) AND workspace_id:"{escaped_workspace_id}"'
        )

        assert self._index is not None
        return self._index.parse_query(
            query=combined_query, default_field_names=["content"]
        )

    def _collect_search_results(
        self,
        parsed_query: tantivy.Query,
        search_limit: int,
        result_limit: int,
        tombstoned_memories: set[str],
        searcher: tantivy.Searcher | None = None,
    ) -> list[tuple[str, float]]:
        """Execute query and collect unique live documents."""
        pinned = searcher if searcher is not None else self.searcher
        top_docs = pinned.search(query=parsed_query, limit=search_limit)
        results: list[tuple[str, float]] = []
        seen: set[str] = set()

        for score, doc_addr in top_docs.hits:
            doc = pinned.doc(doc_addr)
            memory = doc.get_first("content")
            if not isinstance(memory, str):
                continue

            is_deleted_val = doc.get_first("is_deleted")
            is_deleted = int(is_deleted_val) if is_deleted_val is not None else 0
            if is_deleted == 1:
                continue

            if memory in tombstoned_memories or memory in seen:
                continue

            if self.logger:
                self.logger.debug(
                    "Tantivy match",
                    extra={"memory_length": len(memory)},
                )
            seen.add(memory)
            results.append((memory, score))

            if len(results) >= result_limit:
                break

        return results

    def _require_open(self) -> None:
        """Reject operations after close() has released the index."""
        if self._closed:
            raise SearchError("TantivyEngine is closed")

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
        self._closed = True
        with self._lease(LeaseMode.EXCLUSIVE):
            self._close_unlocked()

    def _close_unlocked(self) -> None:
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
                                "workspace_id": self.config.workspace_id,
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
        if self.logger:
            self.logger.info(
                "Tantivy engine closed",
                extra={"workspace_id": self.config.workspace_id},
            )

    def soft_delete(
        self, workspace_id: str, content: str, *, verify_exists: bool = True
    ) -> bool:
        """Mark a document as deleted by adding a tombstone (O(1) operation).

        Thread-safe. This is much faster than rebuilding the entire index (O(n) for rebuild vs O(1)).

        The tombstone approach:
        - Original document remains in index (is_deleted=0, deleted_at=0)
        - Tombstone document added (is_deleted=1, deleted_at=<timestamp>)
        - Search filters out memories that have tombstones
        - Compaction removes both original and tombstone

        Args:
            workspace_id: Workspace identifier for filtering.
            content: Exact memory content to soft-delete.

        Returns:
            True if tombstone was added, False if memory wasn't found.
        """
        with self._lease(LeaseMode.EXCLUSIVE):
            return self._soft_delete_unlocked(
                workspace_id, content, verify_exists=verify_exists
            )

    def _soft_delete_unlocked(
        self, workspace_id: str, content: str, *, verify_exists: bool
    ) -> bool:
        if verify_exists:
            existing = self.find_by_exact_match(workspace_id, content)
            if not existing:
                if self.logger:
                    self.logger.debug(
                        "Soft-delete: memory not found",
                        extra={
                            "workspace_id": workspace_id,
                            "memory_length": len(content) if content else 0,
                        },
                    )
                return False

        live_count, tomb_count = self._count_live_and_tomb(workspace_id, content)
        needed = live_count - tomb_count
        if needed <= 0:
            return False

        self._add_tombstone_docs(workspace_id, content, needed)
        if self.coordinator is not None:
            self._finalize_writer(relinquish=True)

        if self.logger:
            self.logger.debug(
                "Soft-delete: tombstone added",
                extra={
                    "workspace_id": workspace_id,
                    "memory_length": len(content) if content else 0,
                    "tombstones_added": needed,
                },
            )

        return True

    def _count_live_and_tomb(self, workspace_id: str, content: str) -> tuple[int, int]:
        """Count committed live and tombstone copies of one memory."""
        if self._index is None:
            return 0, 0

        escaped_workspace_id = self._escape_tantivy_query(workspace_id)
        query = self._index.parse_query(
            query=f'workspace_id:"{escaped_workspace_id}"',
            default_field_names=["workspace_id"],
        )
        live_count = 0
        tomb_count = 0
        pinned = self.searcher
        for _, doc_addr in pinned.search(
            query=query, limit=self._get_doc_limit()
        ).hits:
            doc = pinned.doc(doc_addr)
            memory = doc.get_first("content")
            if memory != content:
                continue
            is_deleted_val = doc.get_first("is_deleted")
            is_deleted = int(is_deleted_val) if is_deleted_val is not None else 0
            if is_deleted == 1:
                tomb_count += 1
            else:
                live_count += 1
        return live_count, tomb_count

    def _add_tombstone_docs(
        self, workspace_id: str, content: str, count: int
    ) -> None:
        """Plant ``count`` tombstone documents under the writer lock."""
        deleted_at = int(time.time() * 1000)
        with self._writer_lock:
            for _ in range(count):
                doc = tantivy.Document()
                doc.add_text("workspace_id", workspace_id)
                doc.add_text("content", content)
                doc.add_unsigned("is_deleted", 1)
                doc.add_integer("deleted_at", deleted_at)
                _ = self.writer.add_document(doc)

    def delete(
        self, workspace_id: str, content: str, *, verify_exists: bool = True
    ) -> bool:
        """Delete a document from the Tantivy index by exact memory match (thread-safe).

        When soft-delete is enabled (default), uses O(1) tombstone marking.
        Otherwise, falls back to O(n) rebuild approach.

        Soft-delete approach:
        - Adds a tombstone document with is_deleted=1
        - Search automatically filters out tombstoned memories
        - Compaction removes tombstones and originals periodically

        Hard-delete approach (when soft-delete disabled):
        - Gets all documents from all workspaces
        - Filters out the target memory
        - Rewrites remaining documents in place (no directory replacement)

        Thread-safe: Uses writer lock since Tantivy IndexWriter is NOT thread-safe.

        Args:
            workspace_id: Workspace identifier for filtering.
            content: Exact memory content to delete.

        Returns:
            True if document was found and deleted, False otherwise.
        """
        with self._lease(LeaseMode.EXCLUSIVE):
            if self.config.soft_delete_enabled:
                result = self.soft_delete(
                    workspace_id, content, verify_exists=verify_exists
                )
                if result:
                    self.commit()
                return result

            try:
                return self._delete_via_rebuild(workspace_id, content)
            except ValueError as e:
                self._log_delete_error(
                    workspace_id, content, e, "warning", "QueryParseError"
                )
                return False
            except OSError as e:
                self._log_delete_error(
                    workspace_id,
                    content,
                    e,
                    "error",
                    "FileSystemError",
                    include_index_path=True,
                )
                return False
            except Exception as e:
                self._log_delete_error(
                    workspace_id, content, e, "error", type(e).__name__
                )
                return False

    def delete_batch(
        self,
        workspace_id: str,
        contents: list[str],
        *,
        verify_exists: bool = True,
    ) -> int:
        """Delete many memories and commit once.

        Honors ``soft_delete_enabled``: tombstones when true, one rebuild
        per item when false.
        """
        if not contents:
            return 0
        with self._lease(LeaseMode.EXCLUSIVE):
            return self._delete_batch_unlocked(
                workspace_id, contents, verify_exists=verify_exists
            )

    def _delete_batch_unlocked(
        self,
        workspace_id: str,
        contents: list[str],
        *,
        verify_exists: bool,
    ) -> int:
        if self.config.soft_delete_enabled:
            deleted = 0
            for content in contents:
                if self.soft_delete(
                    workspace_id, content, verify_exists=verify_exists
                ):
                    deleted += 1
            if deleted:
                self.commit()
            return deleted
        deleted = 0
        for content in contents:
            if self._delete_via_rebuild(workspace_id, content):
                deleted += 1
        return deleted

    def _delete_via_rebuild(self, workspace_id: str, content: str) -> bool:
        """Delete a document using the full index rebuild approach."""
        if self._index is None:
            return False

        all_docs = self._get_all_docs_all_workspaces()

        target_exists = any(
            pid == workspace_id and msg == content for pid, msg in all_docs
        )
        if not target_exists:
            if self.logger:
                self.logger.debug(
                    "Tantivy delete: document not found",
                    extra={
                        "workspace_id": workspace_id,
                        "memory_length": len(content) if content else 0,
                    },
                )
            return False

        docs_to_keep = [
            (pid, msg)
            for pid, msg in all_docs
            if not (pid == workspace_id and msg == content)
        ]
        deleted_count = len(all_docs) - len(docs_to_keep)

        if self.logger:
            self.logger.debug(
                "Tantivy delete: found document(s) to delete",
                extra={
                    "workspace_id": workspace_id,
                    "memory_length": len(content) if content else 0,
                    "deleted_count": deleted_count,
                    "remaining_count": len(docs_to_keep),
                },
            )

        self._rewrite_index_in_place(docs_to_keep)

        if self.logger:
            self.logger.debug(
                "Tantivy delete: completed successfully",
                extra={
                    "workspace_id": workspace_id,
                    "memory_length": len(content) if content else 0,
                    "deleted_count": deleted_count,
                },
            )

        return True

    def _log_delete_error(
        self,
        workspace_id: str,
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
            "workspace_id": workspace_id,
            "memory_length": len(content) if content else 0,
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
        """Rebuild the index with the specified documents from all workspaces.

        Legacy startup-only directory rebuild. Request-path delete/compact
        must use ``_rewrite_index_in_place`` instead.

        Args:
            docs_to_keep: List of (workspace_id, content) tuples to preserve.
        """
        import shutil

        index_path = self.config.index_path
        backup_path = self._rebuild_backup_path(index_path)

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

        self._restore_rebuild_backup_if_needed(index_path)
        if os.path.exists(index_path) and self._index_is_openable(index_path):
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path)
            _ = shutil.copytree(index_path, backup_path)

        try:
            if os.path.exists(index_path):
                shutil.rmtree(index_path)
            self._initialize_index(restore_backup=False)
            for workspace_id, content in docs_to_keep:
                self.add(workspace_id, content)
            self.commit()
        except Exception:
            if os.path.exists(backup_path):
                if os.path.exists(index_path):
                    shutil.rmtree(index_path)
                _ = shutil.move(backup_path, index_path)
                self._initialize_index(restore_backup=False)
            raise
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)

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
            num_docs = searcher.num_docs()
        except Exception as e:
            if self.logger:
                self.logger.debug(
                    "Failed to read Tantivy num_docs",
                    extra={"error": str(e)},
                )

        if not isinstance(num_docs, int) or num_docs < 0:
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

        with self._lease(LeaseMode.SHARED):
            return self._tombstone_stats_unlocked()

    def _tombstone_stats_unlocked(self) -> dict[str, int]:
        if self._index is None:
            return {
                "total_docs": 0,
                "active_docs": 0,
                "tombstones": 0,
                "unique_active_memories": 0,
            }
        self._refresh_reader()
        try:
            doc_limit = self._get_doc_limit()

            # Get all documents across all workspaces
            query = self._index.parse_query(
                query="*",
                default_field_names=["content"],
            )
            pinned = self.searcher
            top_docs = pinned.search(query=query, limit=doc_limit)

            total_docs = 0
            active_docs = 0
            tombstones = 0
            live_counts: dict[str, int] = {}
            tomb_counts: dict[str, int] = {}

            for _, doc_addr in top_docs.hits:
                doc = pinned.doc(doc_addr)
                total_docs += 1
                memory = doc.get_first("content")

                is_deleted_val = doc.get_first("is_deleted")
                is_deleted = int(is_deleted_val) if is_deleted_val is not None else 0
                if is_deleted == 1:
                    tombstones += 1
                    if memory is not None:
                        tomb_counts[memory] = tomb_counts.get(memory, 0) + 1
                else:
                    active_docs += 1
                    if memory is not None:
                        live_counts[memory] = live_counts.get(memory, 0) + 1

            unique_active = {
                memory
                for memory, live in live_counts.items()
                if tomb_counts.get(memory, 0) < live
            }

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

        with self._lease(LeaseMode.EXCLUSIVE):
            return self._compact_unlocked(force=force, start_time=start_time)

    def _compact_unlocked(self, *, force: bool, start_time: float) -> dict[str, int]:
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

            self._rewrite_index_in_place(docs_to_keep)
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
        pinned = index.searcher()
        top_docs = pinned.search(query=query, limit=self._get_doc_limit())

        all_docs: list[tuple[str, str, int]] = []
        live_counts: dict[tuple[str, str], int] = {}
        tomb_counts: dict[tuple[str, str], int] = {}

        for _, doc_addr in top_docs.hits:
            doc = pinned.doc(doc_addr)
            workspace_id = doc.get_first("workspace_id")
            memory = doc.get_first("content")
            is_deleted_val = doc.get_first("is_deleted")
            is_deleted = int(is_deleted_val) if is_deleted_val is not None else 0

            if workspace_id is not None and memory is not None:
                all_docs.append((workspace_id, memory, is_deleted))
                key = (workspace_id, memory)
                if is_deleted == 1:
                    tomb_counts[key] = tomb_counts.get(key, 0) + 1
                else:
                    live_counts[key] = live_counts.get(key, 0) + 1

        docs_to_keep: list[tuple[str, str]] = []
        removed_originals = 0
        removed_tombstones = 0
        kept_live: dict[tuple[str, str], int] = {}

        for workspace_id, memory, is_deleted in all_docs:
            key = (workspace_id, memory)
            if is_deleted == 1:
                removed_tombstones += 1
                continue
            surplus = live_counts.get(key, 0) - tomb_counts.get(key, 0)
            if surplus <= 0 or kept_live.get(key, 0) >= surplus:
                removed_originals += 1
                continue
            kept_live[key] = kept_live.get(key, 0) + 1
            docs_to_keep.append((workspace_id, memory))

        return docs_to_keep, removed_tombstones, removed_originals

    def _try_garbage_collect(self) -> None:
        """Best-effort garbage collection of old segment files."""
        try:
            if self._writer is not None:
                self._writer.garbage_collect_files()
        except Exception:
            pass  # Files will be cleaned up eventually
