"""SQLite-backed memory storage for USearch engine.

This module provides persistent memory text storage separate from the
USearch vector index. Memories are stored with their workspace_id for filtering
and the SQLite row ID is used as the USearch key.

Uses SQLite with WAL mode for improved concurrent write performance via MVCC.
"""

from dataclasses import dataclass
import os
import sqlite3
import threading
from typing import Any, final

from pydantic import BaseModel, ConfigDict, PrivateAttr

from reflectlog.core.exceptions import StorageError
from reflectlog.core.logging import IStructuredLogger
from reflectlog.core.types import (
    IndexIntentKind,
    ReplacementTransition,
    ReplacementTransitionRequest,
    ReplacementTransitionStatus,
)

TRANSITION_PENDING: ReplacementTransitionStatus = "pending"
TRANSITION_COMPLETED: ReplacementTransitionStatus = "completed"


@dataclass(frozen=True)
class MemoryRecord:
    """A memory record from the database.

    Attributes:
        id: libSQL auto-increment ID (used as USearch key).
        workspace_id: Workspace identifier for filtering.
        content: The memory text content.
        created_at: Timestamp when the memory was created (ISO format string).
    """

    id: int
    workspace_id: str
    content: str
    created_at: str = ""  # Default empty for backward compatibility


@dataclass(frozen=True)
class ArchivedMemoryRecord:
    """An archived (replaced) memory record.

    Attributes:
        id: Archive record ID.
        original_id: Original memory ID before archiving.
        workspace_id: Workspace identifier.
        content: The archived memory text.
        replaced_by: The new memory that replaced this one.
        reason: LLM explanation for why replacement occurred.
        confidence: LLM confidence score (0.0-1.0).
        archived_at: Timestamp when memory was archived.
    """

    id: int
    original_id: int
    workspace_id: str
    content: str
    replaced_by: str
    reason: str
    confidence: float
    archived_at: str


@final
class MemoryStore(BaseModel):
    """SQLite-backed memory storage for USearch engine.

    Provides CRUD operations for memory text storage, using SQLite's
    auto-increment ID as the key for USearch vector index.

    Uses SQLite with WAL mode for better concurrent performance
    via MVCC (Multi-Version Concurrency Control).

    Example:
        ```python
        store = MemoryStore(db_path="indexes/project/memories.db")
        mem_id = store.insert("my-project", "Hello world")
        record = store.get(mem_id)
        all_memories = store.get_all("my-project")
        store.delete(mem_id)
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    db_path: str
    logger: IStructuredLogger | None = None
    timeout: float = 30.0  # Database busy timeout in seconds

    _conn: sqlite3.Connection | None = PrivateAttr(default=None)
    _init_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _conn_lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)

    def __init__(
        self, db_path: str, logger: IStructuredLogger | None = None, **kwargs: Any
    ) -> None:
        """Initialize MemoryStore.

        Args:
            db_path: Path to SQLite database file.
            logger: Optional StructuredLogger instance.
            **kwargs: Additional arguments passed to BaseModel.
        """
        super().__init__(db_path=db_path, logger=logger, **kwargs)

    @property
    def connection(self) -> sqlite3.Connection:
        """Get SQLite connection (lazy initialization).

        Returns:
            SQLite connection with WAL mode enabled and busy timeout.
        """
        if self._conn is None:
            with self._init_lock:
                if self._conn is None:
                    # Ensure directory exists
                    db_dir = os.path.dirname(self.db_path)
                    if db_dir:
                        os.makedirs(db_dir, exist_ok=True)

                    self._conn = sqlite3.connect(
                        self.db_path,
                        check_same_thread=False,
                        timeout=self.timeout,
                    )
                    _ = self._conn.execute("PRAGMA journal_mode=WAL")
                    _ = self._conn.execute("PRAGMA synchronous=NORMAL")
                    # Set busy timeout to handle concurrent access gracefully
                    _ = self._conn.execute(
                        f"PRAGMA busy_timeout = {int(self.timeout * 1000)}"
                    )
                    self._create_schema()
                    self._conn.commit()

                    if self.logger:
                        self.logger.debug(
                            "MemoryStore initialized",
                            extra={
                                "db_path": self.db_path,
                                "timeout_seconds": self.timeout,
                            },
                        )
        # Type narrowing assertion - self._conn is guaranteed non-None after init
        assert self._conn is not None
        return self._conn

    def _create_schema(self) -> None:
        """Create database schema if it doesn't exist."""
        cursor = self.connection.cursor()
        self._create_memories_schema(cursor)
        self._create_archive_schema(cursor)
        self._create_transition_schema(cursor)
        cursor.close()

    def _create_memories_schema(self, cursor: sqlite3.Cursor) -> None:
        """Create the active memories table and lookup indexes."""
        _ = cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace_id TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        _ = cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_workspace_id ON memories(workspace_id)"
        )
        _ = cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_dedup ON "
            "memories(workspace_id, content)"
        )

    def _create_archive_schema(self, cursor: sqlite3.Cursor) -> None:
        """Create archived memory audit table and uniqueness guards."""
        _ = cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS archived_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_id INTEGER NOT NULL,
                workspace_id TEXT NOT NULL,
                content TEXT NOT NULL,
                replaced_by TEXT NOT NULL,
                reason TEXT NOT NULL,
                confidence REAL NOT NULL,
                archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        _ = cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_archived_workspace_id "
            "ON archived_memories(workspace_id)"
        )
        _ = cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_archived_at "
            "ON archived_memories(archived_at)"
        )
        self._dedupe_archive_pairs(cursor)
        _ = cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_archived_original_replaced "
            "ON archived_memories(original_id, replaced_by)"
        )

    def _create_transition_schema(self, cursor: sqlite3.Cursor) -> None:
        """Create durable replacement-transition intent table."""
        _ = cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS replacement_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace_id TEXT NOT NULL,
                old_memory_id INTEGER NOT NULL,
                old_content TEXT NOT NULL,
                new_content TEXT NOT NULL,
                archive_id INTEGER NOT NULL,
                reason TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                kind TEXT NOT NULL DEFAULT 'replace'
            )
        """
        )
        self._ensure_transition_kind_column(cursor)
        self._dedupe_transition_old_ids(cursor)
        _ = cursor.execute("DROP INDEX IF EXISTS idx_transition_identity")
        _ = cursor.execute("DROP INDEX IF EXISTS idx_transition_old_memory")
        _ = cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_transition_old_replace "
            "ON replacement_transitions(workspace_id, old_memory_id) "
            "WHERE kind = 'replace'"
        )
        _ = cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_transition_old_delete "
            "ON replacement_transitions(workspace_id, old_memory_id) "
            "WHERE kind = 'delete'"
        )
        _ = cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_pending_add "
            "ON replacement_transitions(workspace_id, new_content) "
            "WHERE kind = 'add' AND status = 'pending'"
        )
        _ = cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_transition_pending "
            "ON replacement_transitions(status)"
        )

    def _ensure_transition_kind_column(self, cursor: sqlite3.Cursor) -> None:
        """Add kind to databases created before add/delete intents existed."""
        columns = {
            str(row[1])
            for row in cursor.execute("PRAGMA table_info(replacement_transitions)")
        }
        if "kind" in columns:
            return
        _ = cursor.execute(
            "ALTER TABLE replacement_transitions "
            "ADD COLUMN kind TEXT NOT NULL DEFAULT 'replace'"
        )

    def _dedupe_archive_pairs(self, cursor: sqlite3.Cursor) -> None:
        """Keep one archive row per (original_id, replaced_by) before uniquing."""
        _ = cursor.execute(
            """
            DELETE FROM archived_memories
            WHERE id NOT IN (
                SELECT MIN(id) FROM archived_memories
                GROUP BY original_id, replaced_by
            )
            """
        )

    def _dedupe_transition_old_ids(self, cursor: sqlite3.Cursor) -> None:
        """Keep one delete/replace row per old id. Never group add intents.

        Adds all use ``old_memory_id=0``. Deduping them by that key would drop
        every later pending add on the next process start.
        """
        _ = cursor.execute(
            """
            DELETE FROM replacement_transitions
            WHERE kind IN ('delete', 'replace')
              AND id NOT IN (
                SELECT MIN(id) FROM replacement_transitions
                WHERE kind IN ('delete', 'replace')
                GROUP BY workspace_id, old_memory_id, kind
            )
            """
        )

    def insert(self, workspace_id: str, content: str) -> int:
        """Insert a memory and return its ID.

        Args:
            workspace_id: Workspace identifier.
            content: Memory text content.

        Returns:
            The auto-generated libSQL row ID.

        Raises:
            StorageError: If insert fails.
        """
        with self._conn_lock:
            cursor = self.connection.cursor()
            try:
                _ = cursor.execute(
                    "INSERT INTO memories (workspace_id, content) VALUES (?, ?)",
                    (workspace_id, content),
                )
                row_id = cursor.lastrowid
                self.connection.commit()

                if row_id is None:
                    raise StorageError("Insert did not return a row ID")

                if self.logger:
                    self.logger.debug(
                        "Memory inserted",
                        extra={
                            "memory_id": row_id,
                            "workspace_id": workspace_id,
                            "memory_length": len(content),
                        },
                    )
                return row_id

            except (sqlite3.Error, ValueError) as e:
                error_str = str(e).lower()
                if "unique constraint" in error_str or "constraint failed" in error_str:
                    if self.logger:
                        self.logger.debug(
                            "Duplicate memory detected",
                            extra={"workspace_id": workspace_id, "error": str(e)},
                        )
                    raise StorageError(f"Duplicate memory: {e}") from e
                # Other libsql errors
                if self.logger:
                    self.logger.error(
                        "Failed to insert memory",
                        extra={"workspace_id": workspace_id, "error": str(e)},
                    )
                raise StorageError(f"Failed to insert memory: {e}") from e
            finally:
                cursor.close()

    def insert_many(
        self, workspace_id: str, contents: list[str]
    ) -> list[tuple[str, int]]:
        """Insert multiple memories in a single transaction.

        Args:
            workspace_id: Workspace identifier.
            contents: List of memory texts to insert.

        Returns:
            List of (content, id) tuples for successfully inserted memories.
            Duplicate memories are skipped.

        Raises:
            StorageError: If the batch insert fails.
        """
        if not contents:
            return []

        with self._conn_lock:
            cursor = self.connection.cursor()
            inserted: list[tuple[str, int]] = []
            skipped = 0
            try:
                _ = cursor.execute("BEGIN")
                for content in contents:
                    try:
                        _ = cursor.execute(
                            "INSERT INTO memories (workspace_id, content) VALUES (?, ?)",
                            (workspace_id, content),
                        )
                        row_id = cursor.lastrowid
                        if row_id is None:
                            raise StorageError("Insert did not return a row ID")
                        inserted.append((content, int(row_id)))
                    except sqlite3.IntegrityError as e:
                        error_str = str(e).lower()
                        if (
                            "unique constraint" in error_str
                            or "constraint failed" in error_str
                        ):
                            skipped += 1
                            if self.logger:
                                self.logger.debug(
                                    "Duplicate memory skipped during batch insert",
                                    extra={"workspace_id": workspace_id, "error": str(e)},
                                )
                            continue
                        raise
                self.connection.commit()

                if self.logger:
                    self.logger.debug(
                        "Batch insert completed",
                        extra={
                            "workspace_id": workspace_id,
                            "inserted_count": len(inserted),
                            "skipped_duplicates": skipped,
                        },
                    )
                return inserted

            except (sqlite3.Error, StorageError) as e:
                self.connection.rollback()
                if self.logger:
                    self.logger.error(
                        "Failed to insert memories batch",
                        extra={
                            "workspace_id": workspace_id,
                            "memory_count": len(contents),
                            "error": str(e),
                        },
                    )
                raise StorageError(f"Failed to insert memory batch: {e}") from e
            finally:
                cursor.close()

    def get(self, memory_id: int) -> MemoryRecord | None:
        """Get a memory by its ID.

        Args:
            memory_id: The libSQL row ID.

        Returns:
            MemoryRecord if found, None otherwise.

        Raises:
            StorageError: If database operation fails (other than not found).
        """
        with self._conn_lock:
            cursor = self.connection.cursor()
            try:
                _ = cursor.execute(
                    "SELECT id, workspace_id, content, created_at "
                    "FROM memories WHERE id = ?",
                    (memory_id,),
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                return MemoryRecord(
                    id=row[0],
                    workspace_id=row[1],
                    content=row[2],
                    created_at=row[3] if row[3] else "",
                )

            except sqlite3.Error as e:
                if self.logger:
                    self.logger.error(
                        "Failed to get memory",
                        extra={"memory_id": memory_id, "error": str(e)},
                    )
                raise StorageError(f"Failed to retrieve memory: {e}") from e
            finally:
                cursor.close()

    def get_batch(self, memory_ids: list[int]) -> dict[int, MemoryRecord]:
        """Get multiple memories by their IDs in a single query.

        This is significantly more efficient than calling get() multiple times,
        reducing N database round-trips to 1.

        Args:
            memory_ids: List of libSQL row IDs to retrieve.

        Returns:
            Dictionary mapping memory ID to MemoryRecord.
            Missing IDs are not included in the result.

        Raises:
            StorageError: If database operation fails.
        """
        if not memory_ids:
            return {}

        with self._conn_lock:
            cursor = self.connection.cursor()
            try:
                placeholders = ",".join("?" * len(memory_ids))
                _ = cursor.execute(
                    f"SELECT id, workspace_id, content, created_at "
                    f"FROM memories WHERE id IN ({placeholders})",
                    memory_ids,
                )
                rows = cursor.fetchall()

                return {
                    row[0]: MemoryRecord(
                        id=row[0],
                        workspace_id=row[1],
                        content=row[2],
                        created_at=row[3] if row[3] else "",
                    )
                    for row in rows
                }

            except sqlite3.Error as e:
                if self.logger:
                    self.logger.error(
                        "Failed to get memories batch",
                        extra={"memory_ids_count": len(memory_ids), "error": str(e)},
                    )
                raise StorageError(f"Failed to retrieve memory batch: {e}") from e
            finally:
                cursor.close()

    def count(self, workspace_id: str) -> int:
        """Return how many memories exist for a workspace."""
        with self._conn_lock:
            cursor = self.connection.cursor()
            try:
                _ = cursor.execute(
                    "SELECT COUNT(*) FROM memories WHERE workspace_id = ?",
                    (workspace_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    return 0
                return int(row[0])
            except sqlite3.Error as e:
                raise StorageError(f"Failed to count memories: {e}") from e
            finally:
                cursor.close()

    def get_all(
        self,
        workspace_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[str]:
        """Get memories for a workspace, optionally paged in SQL.

        Args:
            workspace_id: Workspace identifier.
            limit: Maximum rows to return. ``None`` returns the rest after offset.
            offset: Rows to skip.

        Returns:
            List of memory strings.

        Raises:
            StorageError: If database operation fails.
        """
        start = max(0, offset)
        with self._conn_lock:
            cursor = self.connection.cursor()
            try:
                if limit is None:
                    _ = cursor.execute(
                        "SELECT content FROM memories WHERE workspace_id = ? "
                        "ORDER BY id LIMIT -1 OFFSET ?",
                        (workspace_id, start),
                    )
                else:
                    _ = cursor.execute(
                        "SELECT content FROM memories WHERE workspace_id = ? "
                        "ORDER BY id LIMIT ? OFFSET ?",
                        (workspace_id, max(0, limit), start),
                    )
                rows = cursor.fetchall()

                return [row[0] for row in rows]

            except sqlite3.Error as e:
                if self.logger:
                    self.logger.error(
                        "Failed to get all memories",
                        extra={"workspace_id": workspace_id, "error": str(e)},
                    )
                raise StorageError(f"Failed to retrieve all memories: {e}") from e
            finally:
                cursor.close()

    def delete(self, memory_id: int) -> bool:
        """Delete a memory by its ID.

        Args:
            memory_id: The libSQL row ID.

        Returns:
            True if memory was deleted, False if not found.

        Raises:
            StorageError: If database operation fails.
        """
        with self._conn_lock:
            cursor = self.connection.cursor()
            try:
                _ = cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                deleted = cursor.rowcount > 0
                self.connection.commit()

                if self.logger:
                    if deleted:
                        self.logger.debug(
                            "Memory deleted",
                            extra={"memory_id": memory_id},
                        )
                    else:
                        self.logger.debug(
                            "Memory not found for deletion",
                            extra={"memory_id": memory_id},
                        )

                return deleted

            except sqlite3.Error as e:
                if self.logger:
                    self.logger.error(
                        "Failed to delete memory",
                        extra={"memory_id": memory_id, "error": str(e)},
                    )
                raise StorageError(f"Failed to delete memory: {e}") from e
            finally:
                cursor.close()

    def delete_batch(self, memory_ids: list[int]) -> int:
        """Delete multiple memories by their IDs in a single transaction.

        This is significantly more efficient than calling delete() multiple times,
        reducing N database round-trips and commits to 1.

        Args:
            memory_ids: List of libSQL row IDs to delete.

        Returns:
            Number of memories actually deleted.

        Raises:
            StorageError: If database operation fails.
        """
        if not memory_ids:
            return 0

        with self._conn_lock:
            cursor = self.connection.cursor()
            try:
                placeholders = ",".join("?" * len(memory_ids))
                _ = cursor.execute(
                    f"DELETE FROM memories WHERE id IN ({placeholders})",
                    memory_ids,
                )
                deleted_count = cursor.rowcount
                self.connection.commit()

                if self.logger:
                    self.logger.debug(
                        "Memories batch deleted",
                        extra={
                            "requested_count": len(memory_ids),
                            "deleted_count": deleted_count,
                        },
                    )

                return deleted_count

            except sqlite3.Error as e:
                if self.logger:
                    self.logger.error(
                        "Failed to delete memories batch",
                        extra={"memory_ids_count": len(memory_ids), "error": str(e)},
                    )
                raise StorageError(f"Failed to delete memory batch: {e}") from e
            finally:
                cursor.close()

    def exists(self, workspace_id: str, content: str) -> bool:
        """Check if a memory exists (for deduplication).

        Args:
            workspace_id: Workspace identifier.
            content: Memory text to check.

        Returns:
            True if the memory exists for this project.

        Raises:
            StorageError: If database operation fails.
        """
        with self._conn_lock:
            cursor = self.connection.cursor()
            try:
                _ = cursor.execute(
                    "SELECT 1 FROM memories "
                    "WHERE workspace_id = ? AND content = ? LIMIT 1",
                    (workspace_id, content),
                )
                exists = cursor.fetchone() is not None
                return exists

            except sqlite3.Error as e:
                if self.logger:
                    self.logger.error(
                        "Failed to check memory existence",
                        extra={"workspace_id": workspace_id, "error": str(e)},
                    )
                raise StorageError(f"Failed to check memory existence: {e}") from e
            finally:
                cursor.close()

    def exists_many(self, workspace_id: str, contents: list[str]) -> set[str]:
        """Return the subset of contents that already exist in this workspace."""
        unique = list(dict.fromkeys(contents))
        if not unique:
            return set()
        found: set[str] = set()
        chunk_size = 400
        with self._conn_lock:
            cursor = self.connection.cursor()
            try:
                for start in range(0, len(unique), chunk_size):
                    chunk = unique[start : start + chunk_size]
                    placeholders = ",".join("?" for _ in chunk)
                    _ = cursor.execute(
                        "SELECT content FROM memories "
                        f"WHERE workspace_id = ? AND content IN ({placeholders})",
                        (workspace_id, *chunk),
                    )
                    for row in cursor.fetchall():
                        found.add(str(row[0]))
                return found
            except sqlite3.Error as e:
                raise StorageError(f"Failed to check memory existence: {e}") from e
            finally:
                cursor.close()

    def get_id_by_content(self, workspace_id: str, content: str) -> int | None:
        """Get the ID of a memory by its content.

        Args:
            workspace_id: Workspace identifier.
            content: Memory text to look up.

        Returns:
            The memory ID if found, None otherwise.

        Raises:
            StorageError: If database operation fails.
        """
        with self._conn_lock:
            cursor = self.connection.cursor()
            try:
                _ = cursor.execute(
                    "SELECT id FROM memories "
                    "WHERE workspace_id = ? AND content = ? LIMIT 1",
                    (workspace_id, content),
                )
                row = cursor.fetchone()
                return row[0] if row else None

            except sqlite3.Error as e:
                if self.logger:
                    self.logger.error(
                        "Failed to get memory ID",
                        extra={"workspace_id": workspace_id, "error": str(e)},
                    )
                raise StorageError(f"Failed to get memory ID: {e}") from e
            finally:
                cursor.close()

    def archive(
        self,
        memory_id: int,
        workspace_id: str,
        content: str,
        replaced_by: str,
        reason: str,
        confidence: float,
    ) -> int | None:
        """Archive a memory before deletion (for recovery).

        Persists all six caller-supplied values. Retrying the same
        ``(original_id, replaced_by)`` pair returns the existing row
        instead of inserting a duplicate.

        Args:
            memory_id: Original memory ID being archived.
            workspace_id: Workspace identifier.
            content: The memory text being archived.
            replaced_by: The new memory that replaces this one.
            reason: LLM explanation for why replacement occurred.
            confidence: LLM confidence score (0.0-1.0).

        Returns:
            Archive record ID if successful, None otherwise.

        Raises:
            StorageError: If database operation fails.
        """
        with self._conn_lock:
            cursor = self.connection.cursor()
            try:
                archive_id = self._insert_archive_row(
                    cursor,
                    memory_id=memory_id,
                    workspace_id=workspace_id,
                    content=content,
                    replaced_by=replaced_by,
                    reason=reason,
                    confidence=confidence,
                )
                self.connection.commit()

                if self.logger:
                    self.logger.debug(
                        "Memory archived",
                        extra={
                            "archive_id": archive_id,
                            "original_id": memory_id,
                            "workspace_id": workspace_id,
                            "confidence": confidence,
                        },
                    )
                return archive_id

            except (sqlite3.Error, StorageError) as e:
                self.connection.rollback()
                if self.logger:
                    self.logger.error(
                        "Failed to archive memory",
                        extra={"memory_id": memory_id, "error": str(e)},
                    )
                raise StorageError(f"Failed to archive memory: {e}") from e
            finally:
                cursor.close()

    def begin_replacement_transition(
        self,
        old_memory_id: int,
        workspace_id: str,
        old_content: str,
        new_content: str,
        reason: str,
        confidence: float,
    ) -> ReplacementTransition:
        """Record archive + pending transition in one SQLite transaction.

        This is the durable intent log owned by the semantic backend. It
        is not a cross-backend transaction: USearch and Tantivy commits
        stay independent and are reconciled from this row on restart.

        Args:
            old_memory_id: Active memory ID being replaced.
            workspace_id: Workspace identifier.
            old_content: Content of the memory being replaced.
            new_content: Replacement content to converge toward.
            reason: LLM explanation for the replacement.
            confidence: LLM confidence score (0.0-1.0).

        Returns:
            The pending (or already recorded) transition.

        Raises:
            StorageError: If the archive or transition insert fails.
        """
        with self._conn_lock:
            cursor = self.connection.cursor()
            try:
                _ = cursor.execute("BEGIN")
                transition = self._record_one_transition(
                    cursor,
                    ReplacementTransitionRequest(
                        old_memory_id=old_memory_id,
                        workspace_id=workspace_id,
                        old_content=old_content,
                        new_content=new_content,
                        reason=reason,
                        confidence=confidence,
                    ),
                )
                self.connection.commit()

                if self.logger:
                    self.logger.debug(
                        "Replacement transition recorded",
                        extra={
                            "transition_id": transition.id,
                            "archive_id": transition.archive_id,
                            "old_memory_id": old_memory_id,
                            "workspace_id": workspace_id,
                            "status": transition.status,
                        },
                    )
                return transition

            except (sqlite3.Error, StorageError) as e:
                self.connection.rollback()
                if self.logger:
                    self.logger.error(
                        "Failed to record replacement transition",
                        extra={
                            "old_memory_id": old_memory_id,
                            "workspace_id": workspace_id,
                            "error": str(e),
                        },
                    )
                raise StorageError(
                    f"Failed to record replacement transition: {e}"
                ) from e
            finally:
                cursor.close()

    def begin_replacement_transitions(
        self, requests: list[ReplacementTransitionRequest]
    ) -> list[ReplacementTransition]:
        """Record many archive + pending rows in one SQLite transaction."""
        if not requests:
            return []

        with self._conn_lock:
            cursor = self.connection.cursor()
            try:
                _ = cursor.execute("BEGIN")
                recorded = [
                    self._record_one_transition(cursor, request) for request in requests
                ]
                self.connection.commit()
                return recorded
            except (sqlite3.Error, StorageError) as e:
                self.connection.rollback()
                if self.logger:
                    self.logger.error(
                        "Failed to record replacement transitions",
                        extra={"request_count": len(requests), "error": str(e)},
                    )
                raise StorageError(
                    f"Failed to record replacement transitions: {e}"
                ) from e
            finally:
                cursor.close()

    def _record_one_transition(
        self, cursor: sqlite3.Cursor, request: ReplacementTransitionRequest
    ) -> ReplacementTransition:
        """Insert one archive + transition row inside an open transaction."""
        archive_id = self._insert_archive_row(
            cursor,
            memory_id=request.old_memory_id,
            workspace_id=request.workspace_id,
            content=request.old_content,
            replaced_by=request.new_content,
            reason=request.reason,
            confidence=request.confidence,
        )
        return self._insert_transition_row(
            cursor,
            workspace_id=request.workspace_id,
            old_memory_id=request.old_memory_id,
            old_content=request.old_content,
            new_content=request.new_content,
            archive_id=archive_id,
            reason=request.reason,
            confidence=request.confidence,
            kind="replace",
        )

    def list_pending_transitions(self) -> list[ReplacementTransition]:
        """Return unfinished replacement transitions in insert order.

        Raises:
            StorageError: If database operation fails.
        """
        with self._conn_lock:
            cursor = self.connection.cursor()
            try:
                _ = cursor.execute(
                    """
                    SELECT id, workspace_id, old_memory_id, old_content,
                           new_content, archive_id, reason, confidence, status,
                           COALESCE(kind, 'replace')
                    FROM replacement_transitions
                    WHERE status = ?
                    ORDER BY id
                    """,
                    (TRANSITION_PENDING,),
                )
                return [self._row_to_transition(row) for row in cursor.fetchall()]

            except sqlite3.Error as e:
                if self.logger:
                    self.logger.error(
                        "Failed to list pending replacement transitions",
                        extra={"error": str(e)},
                    )
                raise StorageError(
                    f"Failed to list pending replacement transitions: {e}"
                ) from e
            finally:
                cursor.close()

    def get_transition_for_old_memory(
        self, workspace_id: str, old_memory_id: int
    ) -> ReplacementTransition | None:
        """Return the exclusive transition for an old memory, if any."""
        with self._conn_lock:
            cursor = self.connection.cursor()
            try:
                return self._existing_transition(cursor, workspace_id, old_memory_id)
            except sqlite3.Error as e:
                if self.logger:
                    self.logger.error(
                        "Failed to look up replacement transition",
                        extra={
                            "workspace_id": workspace_id,
                            "old_memory_id": old_memory_id,
                            "error": str(e),
                        },
                    )
                raise StorageError(
                    f"Failed to look up replacement transition: {e}"
                ) from e
            finally:
                cursor.close()

    def begin_add_intents(
        self, workspace_id: str, contents: list[str]
    ) -> list[ReplacementTransition]:
        """Record pending add intents for ordinary persist."""
        unique = list(dict.fromkeys(contents))
        if not unique:
            return []
        with self._conn_lock:
            cursor = self.connection.cursor()
            try:
                _ = cursor.execute("BEGIN")
                recorded = [
                    self._insert_add_intent_row(cursor, workspace_id, content)
                    for content in unique
                ]
                self.connection.commit()
                return recorded
            except (sqlite3.Error, StorageError) as e:
                self.connection.rollback()
                raise StorageError(f"Failed to record add intents: {e}") from e
            finally:
                cursor.close()

    def begin_delete_intents(
        self, workspace_id: str, items: list[tuple[int, str]]
    ) -> list[ReplacementTransition]:
        """Record pending delete intents keyed by old memory id."""
        if not items:
            return []
        with self._conn_lock:
            cursor = self.connection.cursor()
            try:
                _ = cursor.execute("BEGIN")
                recorded = [
                    self._insert_transition_row(
                        cursor,
                        workspace_id=workspace_id,
                        old_memory_id=memory_id,
                        old_content=content,
                        new_content="",
                        archive_id=0,
                        reason="delete",
                        confidence=1.0,
                        kind="delete",
                    )
                    for memory_id, content in items
                ]
                self.connection.commit()
                return recorded
            except (sqlite3.Error, StorageError) as e:
                self.connection.rollback()
                raise StorageError(f"Failed to record delete intents: {e}") from e
            finally:
                cursor.close()

    def has_later_intent(
        self,
        *,
        workspace_id: str,
        kind: str,
        content: str,
        after_id: int,
    ) -> bool:
        """Return True when a later add/delete/replace row exists for the text.

        ``kind='delete'`` also matches a later replace of the same old text so a
        stale add cannot resurrect content that was replaced.
        """
        if kind == "add":
            query = (
                "SELECT 1 FROM replacement_transitions "
                "WHERE workspace_id = ? AND kind = 'add' "
                "AND new_content = ? AND id > ? LIMIT 1"
            )
            params: tuple[object, ...] = (workspace_id, content, after_id)
        elif kind == "delete":
            query = (
                "SELECT 1 FROM replacement_transitions "
                "WHERE workspace_id = ? AND kind IN ('delete', 'replace') "
                "AND old_content = ? AND id > ? LIMIT 1"
            )
            params = (workspace_id, content, after_id)
        elif kind == "replace":
            query = (
                "SELECT 1 FROM replacement_transitions "
                "WHERE workspace_id = ? AND kind = 'replace' "
                "AND old_content = ? AND id > ? LIMIT 1"
            )
            params = (workspace_id, content, after_id)
        else:
            raise StorageError(f"Unknown intent kind: {kind}")
        with self._conn_lock:
            cursor = self.connection.cursor()
            try:
                _ = cursor.execute(query, params)
                return cursor.fetchone() is not None
            except sqlite3.Error as e:
                raise StorageError(f"Failed to look up later intent: {e}") from e
            finally:
                cursor.close()

    def complete_replacement_transition(self, transition_id: int) -> None:
        """Mark a replacement transition completed (idempotent).

        Args:
            transition_id: Transition row ID.

        Raises:
            StorageError: If database operation fails.
        """
        with self._conn_lock:
            cursor = self.connection.cursor()
            try:
                _ = cursor.execute(
                    """
                    UPDATE replacement_transitions
                    SET status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ? AND status = ?
                    """,
                    (TRANSITION_COMPLETED, transition_id, TRANSITION_PENDING),
                )
                if cursor.rowcount == 0:
                    existing = self._transition_status(cursor, transition_id)
                    if existing != TRANSITION_COMPLETED:
                        raise StorageError(
                            f"Replacement transition {transition_id} was not pending"
                        )
                self.connection.commit()

                if self.logger:
                    self.logger.debug(
                        "Replacement transition completed",
                        extra={"transition_id": transition_id},
                    )

            except sqlite3.Error as e:
                self.connection.rollback()
                if self.logger:
                    self.logger.error(
                        "Failed to complete replacement transition",
                        extra={"transition_id": transition_id, "error": str(e)},
                    )
                raise StorageError(
                    f"Failed to complete replacement transition: {e}"
                ) from e
            finally:
                cursor.close()

    def _insert_archive_row(
        self,
        cursor: sqlite3.Cursor,
        memory_id: int,
        workspace_id: str,
        content: str,
        replaced_by: str,
        reason: str,
        confidence: float,
    ) -> int:
        """Insert an archive row or return the existing unique row ID."""
        existing_id = self._existing_archive_id(cursor, memory_id, replaced_by)
        if existing_id is not None:
            return existing_id

        try:
            _ = cursor.execute(
                """
                INSERT INTO archived_memories
                    (original_id, workspace_id, content, replaced_by, reason,
                     confidence)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (memory_id, workspace_id, content, replaced_by, reason, confidence),
            )
        except sqlite3.IntegrityError:
            existing_id = self._existing_archive_id(cursor, memory_id, replaced_by)
            if existing_id is not None:
                return existing_id
            raise

        archive_id = cursor.lastrowid
        if archive_id is None:
            raise StorageError("Archive insert did not return a row ID")
        return int(archive_id)

    def _existing_archive_id(
        self, cursor: sqlite3.Cursor, memory_id: int, replaced_by: str
    ) -> int | None:
        """Return the archive ID for an existing unique pair, if any."""
        _ = cursor.execute(
            """
            SELECT id FROM archived_memories
            WHERE original_id = ? AND replaced_by = ?
            """,
            (memory_id, replaced_by),
        )
        row = cursor.fetchone()
        return int(row[0]) if row is not None else None

    def _transition_status(
        self, cursor: sqlite3.Cursor, transition_id: int
    ) -> str | None:
        """Return the stored status for a transition id, if any."""
        _ = cursor.execute(
            "SELECT status FROM replacement_transitions WHERE id = ?",
            (transition_id,),
        )
        row = cursor.fetchone()
        return str(row[0]) if row is not None else None

    def _insert_add_intent_row(
        self,
        cursor: sqlite3.Cursor,
        workspace_id: str,
        content: str,
    ) -> ReplacementTransition:
        """Insert an add intent, allocating a new id after a later delete."""
        existing = self._existing_intent(
            cursor,
            workspace_id=workspace_id,
            kind="add",
            old_memory_id=0,
            new_content=content,
        )
        if existing is not None and self.has_later_intent(
            workspace_id=workspace_id,
            kind="delete",
            content=content,
            after_id=existing.id,
        ):
            self._mark_transition_complete(cursor, existing.id)
        return self._insert_transition_row(
            cursor,
            workspace_id=workspace_id,
            old_memory_id=0,
            old_content="",
            new_content=content,
            archive_id=0,
            reason="add",
            confidence=1.0,
            kind="add",
        )

    def _mark_transition_complete(
        self, cursor: sqlite3.Cursor, transition_id: int
    ) -> None:
        """Mark one transition completed on the caller's cursor."""
        _ = cursor.execute(
            """
            UPDATE replacement_transitions
            SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND status = ?
            """,
            (TRANSITION_COMPLETED, transition_id, TRANSITION_PENDING),
        )

    def _insert_transition_row(
        self,
        cursor: sqlite3.Cursor,
        workspace_id: str,
        old_memory_id: int,
        old_content: str,
        new_content: str,
        archive_id: int,
        reason: str,
        confidence: float,
        kind: IndexIntentKind = "replace",
    ) -> ReplacementTransition:
        """Insert a pending transition or return the exclusive existing row."""
        existing = self._existing_intent(
            cursor,
            workspace_id=workspace_id,
            kind=kind,
            old_memory_id=old_memory_id,
            new_content=new_content,
        )
        if existing is not None:
            if kind == "replace" and existing.new_content != new_content:
                raise StorageError(
                    "Old memory already has a replacement transition"
                )
            return existing

        try:
            _ = cursor.execute(
                """
                INSERT INTO replacement_transitions
                    (workspace_id, old_memory_id, old_content, new_content,
                     archive_id, reason, confidence, status, kind)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workspace_id,
                    old_memory_id,
                    old_content,
                    new_content,
                    archive_id,
                    reason,
                    confidence,
                    TRANSITION_PENDING,
                    kind,
                ),
            )
        except sqlite3.IntegrityError:
            existing = self._existing_intent(
                cursor,
                workspace_id=workspace_id,
                kind=kind,
                old_memory_id=old_memory_id,
                new_content=new_content,
            )
            if existing is not None:
                return existing
            raise StorageError(
                "Old memory already has a replacement transition"
            ) from None

        transition_id = cursor.lastrowid
        if transition_id is None:
            raise StorageError("Transition insert did not return a row ID")
        return ReplacementTransition(
            id=int(transition_id),
            workspace_id=workspace_id,
            old_memory_id=old_memory_id,
            old_content=old_content,
            new_content=new_content,
            archive_id=archive_id,
            reason=reason,
            confidence=confidence,
            status=TRANSITION_PENDING,
            kind=kind,
        )

    def _existing_transition(
        self,
        cursor: sqlite3.Cursor,
        workspace_id: str,
        old_memory_id: int,
    ) -> ReplacementTransition | None:
        """Return the exclusive replace/delete transition for an old memory."""
        return self._existing_intent(
            cursor,
            workspace_id=workspace_id,
            kind="replace",
            old_memory_id=old_memory_id,
            new_content="",
        )

    def _existing_intent(
        self,
        cursor: sqlite3.Cursor,
        *,
        workspace_id: str,
        kind: IndexIntentKind,
        old_memory_id: int,
        new_content: str,
    ) -> ReplacementTransition | None:
        """Return a matching pending or exclusive intent, if present."""
        if kind == "add":
            _ = cursor.execute(
                """
                SELECT id, workspace_id, old_memory_id, old_content, new_content,
                       archive_id, reason, confidence, status,
                       COALESCE(kind, 'replace')
                FROM replacement_transitions
                WHERE workspace_id = ? AND kind = 'add' AND new_content = ?
                  AND status = ?
                """,
                (workspace_id, new_content, TRANSITION_PENDING),
            )
        else:
            _ = cursor.execute(
                """
                SELECT id, workspace_id, old_memory_id, old_content, new_content,
                       archive_id, reason, confidence, status,
                       COALESCE(kind, 'replace')
                FROM replacement_transitions
                WHERE workspace_id = ? AND old_memory_id = ? AND kind = ?
                """,
                (workspace_id, old_memory_id, kind),
            )
        row = cursor.fetchone()
        return self._row_to_transition(row) if row is not None else None

    def _row_to_transition(self, row: tuple[object, ...]) -> ReplacementTransition:
        """Map a replacement_transitions SELECT row to a dataclass."""
        status_value = str(row[8])
        transition_status: ReplacementTransitionStatus = (
            TRANSITION_COMPLETED
            if status_value == TRANSITION_COMPLETED
            else TRANSITION_PENDING
        )
        kind_value = str(row[9]) if len(row) > 9 else "replace"
        kind: IndexIntentKind = (
            "add"
            if kind_value == "add"
            else "delete"
            if kind_value == "delete"
            else "replace"
        )
        return ReplacementTransition(
            id=int(str(row[0])),
            workspace_id=str(row[1]),
            old_memory_id=int(str(row[2])),
            old_content=str(row[3]),
            new_content=str(row[4]),
            archive_id=int(str(row[5])),
            reason=str(row[6]),
            confidence=float(str(row[7])),
            status=transition_status,
            kind=kind,
        )

    def get_archived(
        self, workspace_id: str, limit: int = 100
    ) -> list[ArchivedMemoryRecord]:
        """Get archived memories for a workspace.

        Args:
            workspace_id: Workspace identifier.
            limit: Maximum number of records to return.

        Returns:
            List of ArchivedMemoryRecord objects.

        Raises:
            StorageError: If database operation fails.
        """
        with self._conn_lock:
            cursor = self.connection.cursor()
            try:
                _ = cursor.execute(
                    """
                    SELECT id, original_id, workspace_id, content, replaced_by,
                           reason, confidence, archived_at
                    FROM archived_memories
                    WHERE workspace_id = ?
                    ORDER BY archived_at DESC
                    LIMIT ?
                    """,
                    (workspace_id, limit),
                )
                rows = cursor.fetchall()

                return [
                    ArchivedMemoryRecord(
                        id=row[0],
                        original_id=row[1],
                        workspace_id=row[2],
                        content=row[3],
                        replaced_by=row[4],
                        reason=row[5],
                        confidence=row[6],
                        archived_at=row[7],
                    )
                    for row in rows
                ]

            except sqlite3.Error as e:
                if self.logger:
                    self.logger.error(
                        "Failed to get archived memories",
                        extra={"workspace_id": workspace_id, "error": str(e)},
                    )
                raise StorageError(f"Failed to get archived memories: {e}") from e
            finally:
                cursor.close()

    def restore_from_archive(self, archive_id: int) -> int | None:
        """Restore a memory from the archive.

        Args:
            archive_id: The archive record ID.

        Returns:
            New memory ID if successful, None otherwise.

        Raises:
            StorageError: If database operation fails.
        """
        with self._conn_lock:
            cursor = self.connection.cursor()
            try:
                # Get the archived record
                _ = cursor.execute(
                    """
                    SELECT workspace_id, content FROM archived_memories WHERE id = ?
                    """,
                    (archive_id,),
                )
                row = cursor.fetchone()

                if row is None:
                    if self.logger:
                        self.logger.warning(
                            "Archive record not found",
                            extra={"archive_id": archive_id},
                        )
                    return None

                workspace_id, content = row[0], row[1]

                # Insert back into memories table
                _ = cursor.execute(
                    "INSERT INTO memories (workspace_id, content) VALUES (?, ?)",
                    (workspace_id, content),
                )
                new_id = cursor.lastrowid

                _ = cursor.execute(
                    "DELETE FROM archived_memories WHERE id = ?",
                    (archive_id,),
                )

                self.connection.commit()

                if self.logger:
                    self.logger.info(
                        "Memory restored from archive",
                        extra={
                            "archive_id": archive_id,
                            "new_memory_id": new_id,
                            "workspace_id": workspace_id,
                        },
                    )
                return new_id

            except sqlite3.Error as e:
                if self.logger:
                    self.logger.error(
                        "Failed to restore from archive",
                        extra={"archive_id": archive_id, "error": str(e)},
                    )
                raise StorageError(f"Failed to restore from archive: {e}") from e
            finally:
                cursor.close()

    def cleanup_expired_archive(self, ttl_days: int) -> int:
        """Remove archived memories older than TTL.

        Args:
            ttl_days: Number of days to retain archived memories.
                      If 0, no cleanup is performed (permanent archive).

        Returns:
            Number of records deleted.

        Raises:
            StorageError: If database operation fails.
        """
        if ttl_days <= 0:
            return 0

        with self._conn_lock:
            cursor = self.connection.cursor()
            try:
                _ = cursor.execute(
                    """
                    DELETE FROM archived_memories
                    WHERE archived_at < datetime('now', '-' || ? || ' days')
                    """,
                    (ttl_days,),
                )
                deleted_count = cursor.rowcount
                self.connection.commit()

                if self.logger and deleted_count > 0:
                    self.logger.info(
                        "Expired archive records cleaned up",
                        extra={
                            "deleted_count": deleted_count,
                            "ttl_days": ttl_days,
                        },
                    )
                return deleted_count

            except sqlite3.Error as e:
                if self.logger:
                    self.logger.error(
                        "Failed to cleanup expired archive",
                        extra={"ttl_days": ttl_days, "error": str(e)},
                    )
                raise StorageError(f"Failed to cleanup expired archive: {e}") from e
            finally:
                cursor.close()

    def close(self) -> None:
        """Close the database connection."""
        with self._conn_lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
                if self.logger:
                    self.logger.debug(
                        "MemoryStore closed",
                        extra={"db_path": self.db_path},
                    )

    def is_ready(self) -> bool:
        """Return True if the SQLite connection has already been opened."""
        return self._conn is not None

    def ensure_initialized(self) -> None:
        """Ensure the store is fully initialized.

        Forces lazy initialization to complete.
        """
        _ = self.connection
