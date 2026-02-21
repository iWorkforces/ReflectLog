"""SQLite-backed memory storage for USearch engine.

This module provides persistent memory text storage separate from the
USearch vector index. Memories are stored with their project_id for filtering
and the SQLite row ID is used as the USearch key.

Uses SQLite with WAL mode for improved concurrent write performance via MVCC.
"""

from dataclasses import dataclass
import os
import sqlite3
import threading
from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr

from reflectlog.application.exceptions import StorageError


@dataclass(frozen=True)
class MemoryRecord:
    """A memory record from the database.

    Attributes:
        id: libSQL auto-increment ID (used as USearch key).
        project_id: Project identifier for filtering.
        content: The memory text content.
        created_at: Timestamp when the memory was created (ISO format string).
    """

    id: int
    project_id: str
    content: str
    created_at: str = ""  # Default empty for backward compatibility


@dataclass(frozen=True)
class ArchivedMemoryRecord:
    """An archived (replaced) memory record.

    Attributes:
        id: Archive record ID.
        original_id: Original memory ID before archiving.
        project_id: Project identifier.
        content: The archived memory text.
        replaced_by: The new memory that replaced this one.
        reason: LLM explanation for why replacement occurred.
        confidence: LLM confidence score (0.0-1.0).
        archived_at: Timestamp when memory was archived.
    """

    id: int
    original_id: int
    project_id: str
    content: str
    replaced_by: str
    reason: str
    confidence: float
    archived_at: str


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
    logger: Any = None
    timeout: float = 30.0  # Database busy timeout in seconds

    _conn: sqlite3.Connection | None = PrivateAttr(default=None)
    _init_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _conn_lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)

    def __init__(self, db_path: str, logger: Any = None, **kwargs: Any) -> None:
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
        # Main memories table
        _ = cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        _ = cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_project_id ON memories(project_id)"
        )
        _ = cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_dedup ON "
            "memories(project_id, content)"
        )

        # Archived memories table (for smart replacement recovery)
        _ = cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS archived_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_id INTEGER NOT NULL,
                project_id TEXT NOT NULL,
                content TEXT NOT NULL,
                replaced_by TEXT NOT NULL,
                reason TEXT NOT NULL,
                confidence REAL NOT NULL,
                archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        _ = cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_archived_project_id "
            "ON archived_memories(project_id)"
        )
        _ = cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_archived_at "
            "ON archived_memories(archived_at)"
        )
        cursor.close()

    def insert(self, project_id: str, content: str) -> int:
        """Insert a memory and return its ID.

        Args:
            project_id: Project identifier.
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
                    "INSERT INTO memories (project_id, content) VALUES (?, ?)",
                    (project_id, content),
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
                            "project_id": project_id,
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
                            extra={"project_id": project_id, "error": str(e)},
                        )
                    raise StorageError(f"Duplicate memory: {e}") from e
                # Other libsql errors
                if self.logger:
                    self.logger.error(
                        "Failed to insert memory",
                        extra={"project_id": project_id, "error": str(e)},
                    )
                raise StorageError(f"Failed to insert memory: {e}") from e
            finally:
                cursor.close()

    def insert_many(
        self, project_id: str, contents: list[str]
    ) -> list[tuple[str, int]]:
        """Insert multiple memories in a single transaction.

        Args:
            project_id: Project identifier.
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
                            "INSERT INTO memories (project_id, content) VALUES (?, ?)",
                            (project_id, content),
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
                                    extra={"project_id": project_id, "error": str(e)},
                                )
                            continue
                        raise
                self.connection.commit()

                if self.logger:
                    self.logger.debug(
                        "Batch insert completed",
                        extra={
                            "project_id": project_id,
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
                            "project_id": project_id,
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
                    "SELECT id, project_id, content, created_at "
                    "FROM memories WHERE id = ?",
                    (memory_id,),
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                return MemoryRecord(
                    id=row[0],
                    project_id=row[1],
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
                    f"SELECT id, project_id, content, created_at "
                    f"FROM memories WHERE id IN ({placeholders})",
                    memory_ids,
                )
                rows = cursor.fetchall()

                return {
                    row[0]: MemoryRecord(
                        id=row[0],
                        project_id=row[1],
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

    def get_all(self, project_id: str) -> list[str]:
        """Get all memories for a project.

        Args:
            project_id: Project identifier.

        Returns:
            List of memory strings.

        Raises:
            StorageError: If database operation fails.
        """
        with self._conn_lock:
            cursor = self.connection.cursor()
            try:
                _ = cursor.execute(
                    "SELECT content FROM memories WHERE project_id = ? ORDER BY id",
                    (project_id,),
                )
                rows = cursor.fetchall()

                return [row[0] for row in rows]

            except sqlite3.Error as e:
                if self.logger:
                    self.logger.error(
                        "Failed to get all memories",
                        extra={"project_id": project_id, "error": str(e)},
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

    def exists(self, project_id: str, content: str) -> bool:
        """Check if a memory exists (for deduplication).

        Args:
            project_id: Project identifier.
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
                    "WHERE project_id = ? AND content = ? LIMIT 1",
                    (project_id, content),
                )
                exists = cursor.fetchone() is not None
                return exists

            except sqlite3.Error as e:
                if self.logger:
                    self.logger.error(
                        "Failed to check memory existence",
                        extra={"project_id": project_id, "error": str(e)},
                    )
                raise StorageError(f"Failed to check memory existence: {e}") from e
            finally:
                cursor.close()

    def get_id_by_content(self, project_id: str, content: str) -> int | None:
        """Get the ID of a memory by its content.

        Args:
            project_id: Project identifier.
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
                    "WHERE project_id = ? AND content = ? LIMIT 1",
                    (project_id, content),
                )
                row = cursor.fetchone()
                return row[0] if row else None

            except sqlite3.Error as e:
                if self.logger:
                    self.logger.error(
                        "Failed to get memory ID",
                        extra={"project_id": project_id, "error": str(e)},
                    )
                raise StorageError(f"Failed to get memory ID: {e}") from e
            finally:
                cursor.close()

    def archive(
        self,
        memory_id: int,
        project_id: str,
        content: str,
        replaced_by: str,
        reason: str,
        confidence: float,
    ) -> int | None:
        """Archive a memory before deletion (for recovery).

        Args:
            memory_id: Original memory ID being archived.
            project_id: Project identifier.
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
                _ = cursor.execute(
                    """
                    INSERT INTO archived_memories
                        (original_id, project_id, content, replaced_by, reason,
                         confidence)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (memory_id, project_id, content, replaced_by, reason, confidence),
                )
                archive_id = cursor.lastrowid
                self.connection.commit()

                if self.logger:
                    self.logger.debug(
                        "Memory archived",
                        extra={
                            "archive_id": archive_id,
                            "original_id": memory_id,
                            "project_id": project_id,
                            "confidence": confidence,
                        },
                    )
                return archive_id

            except sqlite3.Error as e:
                if self.logger:
                    self.logger.error(
                        "Failed to archive memory",
                        extra={"memory_id": memory_id, "error": str(e)},
                    )
                raise StorageError(f"Failed to archive memory: {e}") from e
            finally:
                cursor.close()

    def get_archived(
        self, project_id: str, limit: int = 100
    ) -> list[ArchivedMemoryRecord]:
        """Get archived memories for a project.

        Args:
            project_id: Project identifier.
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
                    SELECT id, original_id, project_id, content, replaced_by,
                           reason, confidence, archived_at
                    FROM archived_memories
                    WHERE project_id = ?
                    ORDER BY archived_at DESC
                    LIMIT ?
                    """,
                    (project_id, limit),
                )
                rows = cursor.fetchall()

                return [
                    ArchivedMemoryRecord(
                        id=row[0],
                        original_id=row[1],
                        project_id=row[2],
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
                        extra={"project_id": project_id, "error": str(e)},
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
                    SELECT project_id, content FROM archived_memories WHERE id = ?
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

                project_id, content = row[0], row[1]

                # Insert back into memories table
                _ = cursor.execute(
                    "INSERT INTO memories (project_id, content) VALUES (?, ?)",
                    (project_id, content),
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
                            "project_id": project_id,
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

    def ensure_initialized(self) -> None:
        """Ensure the store is fully initialized.

        Forces lazy initialization to complete.
        """
        _ = self.connection
