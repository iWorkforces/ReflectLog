"""libSQL-backed message storage for USearch engine.

This module provides persistent message text storage separate from the
USearch vector index. Messages are stored with their project_id for filtering
and the SQLite row ID is used as the USearch key.

Uses libSQL (a high-performance SQLite fork) for improved concurrent write
performance via MVCC.
"""

import os
import threading
from dataclasses import dataclass
from typing import Any, Optional

import libsql
from pydantic import BaseModel, ConfigDict, PrivateAttr


@dataclass(frozen=True)
class MessageRecord:
    """A message record from the database.

    Attributes:
        id: libSQL auto-increment ID (used as USearch key).
        project_id: Project identifier for filtering.
        message: The message text content.
    """

    id: int
    project_id: str
    message: str


@dataclass(frozen=True)
class ArchivedMessageRecord:
    """An archived (replaced) message record.

    Attributes:
        id: Archive record ID.
        original_id: Original message ID before archiving.
        project_id: Project identifier.
        message: The archived message text.
        replaced_by: The new message that replaced this one.
        reason: LLM explanation for why replacement occurred.
        confidence: LLM confidence score (0.0-1.0).
        archived_at: Timestamp when message was archived.
    """

    id: int
    original_id: int
    project_id: str
    message: str
    replaced_by: str
    reason: str
    confidence: float
    archived_at: str


class MessageStore(BaseModel):
    """libSQL-backed message storage for USearch engine.

    Provides CRUD operations for message text storage, using libSQL's
    auto-increment ID as the key for USearch vector index.

    Uses libSQL (SQLite fork) with WAL mode for better concurrent performance
    via MVCC (Multi-Version Concurrency Control).

    Example:
        ```python
        store = MessageStore(db_path="indexes/project/messages.db")
        msg_id = store.insert("my-project", "Hello world")
        record = store.get(msg_id)
        all_msgs = store.get_all("my-project")
        store.delete(msg_id)
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    db_path: str
    logger: Any = None

    _conn: Optional[libsql.Connection] = PrivateAttr(default=None)
    _init_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def __init__(self, db_path: str, logger: Any = None, **kwargs: Any) -> None:
        """Initialize MessageStore.

        Args:
            db_path: Path to SQLite database file.
            logger: Optional StructuredLogger instance.
            **kwargs: Additional arguments passed to BaseModel.
        """
        super().__init__(db_path=db_path, logger=logger, **kwargs)

    @property
    def connection(self) -> libsql.Connection:
        """Get libSQL connection (lazy initialization).

        Returns:
            libSQL connection with WAL mode enabled.
        """
        if self._conn is None:
            with self._init_lock:
                if self._conn is None:
                    # Ensure directory exists
                    db_dir = os.path.dirname(self.db_path)
                    if db_dir:
                        os.makedirs(db_dir, exist_ok=True)

                    self._conn = libsql.connect(self.db_path)
                    self._conn.execute("PRAGMA journal_mode=WAL")
                    self._conn.execute("PRAGMA synchronous=NORMAL")
                    self._create_schema()
                    self._conn.commit()

                    if self.logger:
                        self.logger.debug(
                            "MessageStore initialized",
                            extra={"db_path": self.db_path},
                        )
        # Type narrowing assertion - self._conn is guaranteed non-None after init
        assert self._conn is not None
        return self._conn

    def _create_schema(self) -> None:
        """Create database schema if it doesn't exist."""
        cursor = self.connection.cursor()
        # Main messages table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_project_id ON messages(project_id)"
        )
        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_dedup ON messages(project_id, message)"
        )

        # Archived messages table (for smart replacement recovery)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS archived_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_id INTEGER NOT NULL,
                project_id TEXT NOT NULL,
                message TEXT NOT NULL,
                replaced_by TEXT NOT NULL,
                reason TEXT NOT NULL,
                confidence REAL NOT NULL,
                archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_archived_project_id ON archived_messages(project_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_archived_at ON archived_messages(archived_at)"
        )
        cursor.close()

    def insert(self, project_id: str, message: str) -> int:
        """Insert a message and return its ID.

        Args:
            project_id: Project identifier.
            message: Message text content.

        Returns:
            The auto-generated libSQL row ID.

        Raises:
            RuntimeError: If insert fails.
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "INSERT INTO messages (project_id, message) VALUES (?, ?)",
                (project_id, message),
            )
            row_id = cursor.lastrowid
            self.connection.commit()
            cursor.close()

            if row_id is None:
                raise RuntimeError("Insert did not return a row ID")

            if self.logger:
                self.logger.debug(
                    "Message inserted",
                    extra={
                        "message_id": row_id,
                        "project_id": project_id,
                        "message_length": len(message),
                    },
                )
            return row_id

        except Exception as e:
            # Check for unique constraint violation (duplicate message)
            error_str = str(e).lower()
            if "unique constraint" in error_str or "constraint failed" in error_str:
                if self.logger:
                    self.logger.debug(
                        "Duplicate message detected",
                        extra={"project_id": project_id, "error": str(e)},
                    )
                raise RuntimeError(f"Duplicate message: {e}") from e
            # Other errors
            if self.logger:
                self.logger.error(
                    "Failed to insert message",
                    extra={"project_id": project_id, "error": str(e)},
                )
            raise RuntimeError(f"Failed to insert message: {e}") from e

    def get(self, message_id: int) -> Optional[MessageRecord]:
        """Get a message by its ID.

        Args:
            message_id: The libSQL row ID.

        Returns:
            MessageRecord if found, None otherwise.
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT id, project_id, message FROM messages WHERE id = ?",
                (message_id,),
            )
            row = cursor.fetchone()
            cursor.close()

            if row is None:
                return None

            return MessageRecord(id=row[0], project_id=row[1], message=row[2])

        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Failed to get message",
                    extra={"message_id": message_id, "error": str(e)},
                )
            return None

    def get_batch(self, message_ids: list[int]) -> dict[int, MessageRecord]:
        """Get multiple messages by their IDs in a single query.

        This is significantly more efficient than calling get() multiple times,
        reducing N database round-trips to 1.

        Args:
            message_ids: List of libSQL row IDs to retrieve.

        Returns:
            Dictionary mapping message ID to MessageRecord.
            Missing IDs are not included in the result.
        """
        if not message_ids:
            return {}

        try:
            cursor = self.connection.cursor()
            placeholders = ",".join("?" * len(message_ids))
            cursor.execute(
                f"SELECT id, project_id, message FROM messages WHERE id IN ({placeholders})",
                message_ids,
            )
            rows = cursor.fetchall()
            cursor.close()

            return {
                row[0]: MessageRecord(id=row[0], project_id=row[1], message=row[2])
                for row in rows
            }

        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Failed to get messages batch",
                    extra={"message_ids_count": len(message_ids), "error": str(e)},
                )
            return {}

    def get_all(self, project_id: str) -> list[str]:
        """Get all messages for a project.

        Args:
            project_id: Project identifier.

        Returns:
            List of message strings.
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT message FROM messages WHERE project_id = ? ORDER BY id",
                (project_id,),
            )
            rows = cursor.fetchall()
            cursor.close()

            return [row[0] for row in rows]

        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Failed to get all messages",
                    extra={"project_id": project_id, "error": str(e)},
                )
            return []

    def delete(self, message_id: int) -> bool:
        """Delete a message by its ID.

        Args:
            message_id: The libSQL row ID.

        Returns:
            True if message was deleted, False if not found.
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("DELETE FROM messages WHERE id = ?", (message_id,))
            deleted = cursor.rowcount > 0
            self.connection.commit()
            cursor.close()

            if self.logger:
                if deleted:
                    self.logger.debug(
                        "Message deleted",
                        extra={"message_id": message_id},
                    )
                else:
                    self.logger.debug(
                        "Message not found for deletion",
                        extra={"message_id": message_id},
                    )

            return deleted

        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Failed to delete message",
                    extra={"message_id": message_id, "error": str(e)},
                )
            return False

    def delete_batch(self, message_ids: list[int]) -> int:
        """Delete multiple messages by their IDs in a single transaction.

        This is significantly more efficient than calling delete() multiple times,
        reducing N database round-trips and commits to 1.

        Args:
            message_ids: List of libSQL row IDs to delete.

        Returns:
            Number of messages actually deleted.
        """
        if not message_ids:
            return 0

        try:
            cursor = self.connection.cursor()
            placeholders = ",".join("?" * len(message_ids))
            cursor.execute(
                f"DELETE FROM messages WHERE id IN ({placeholders})",
                message_ids,
            )
            deleted_count = cursor.rowcount
            self.connection.commit()
            cursor.close()

            if self.logger:
                self.logger.debug(
                    "Messages batch deleted",
                    extra={
                        "requested_count": len(message_ids),
                        "deleted_count": deleted_count,
                    },
                )

            return deleted_count

        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Failed to delete messages batch",
                    extra={"message_ids_count": len(message_ids), "error": str(e)},
                )
            return 0

    def exists(self, project_id: str, message: str) -> bool:
        """Check if a message exists (for deduplication).

        Args:
            project_id: Project identifier.
            message: Message text to check.

        Returns:
            True if the message exists for this project.
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT 1 FROM messages WHERE project_id = ? AND message = ? LIMIT 1",
                (project_id, message),
            )
            exists = cursor.fetchone() is not None
            cursor.close()
            return exists

        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Failed to check message existence",
                    extra={"project_id": project_id, "error": str(e)},
                )
            return False

    def get_id_by_message(self, project_id: str, message: str) -> Optional[int]:
        """Get the ID of a message by its content.

        Args:
            project_id: Project identifier.
            message: Message text to look up.

        Returns:
            The message ID if found, None otherwise.
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT id FROM messages WHERE project_id = ? AND message = ? LIMIT 1",
                (project_id, message),
            )
            row = cursor.fetchone()
            cursor.close()
            return row[0] if row else None

        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Failed to get message ID",
                    extra={"project_id": project_id, "error": str(e)},
                )
            return None

    def archive(
        self,
        message_id: int,
        project_id: str,
        message: str,
        replaced_by: str,
        reason: str,
        confidence: float,
    ) -> Optional[int]:
        """Archive a message before deletion (for recovery).

        Args:
            message_id: Original message ID being archived.
            project_id: Project identifier.
            message: The message text being archived.
            replaced_by: The new message that replaces this one.
            reason: LLM explanation for why replacement occurred.
            confidence: LLM confidence score (0.0-1.0).

        Returns:
            Archive record ID if successful, None otherwise.
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                INSERT INTO archived_messages
                    (original_id, project_id, message, replaced_by, reason, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (message_id, project_id, message, replaced_by, reason, confidence),
            )
            archive_id = cursor.lastrowid
            self.connection.commit()
            cursor.close()

            if self.logger:
                self.logger.debug(
                    "Message archived",
                    extra={
                        "archive_id": archive_id,
                        "original_id": message_id,
                        "project_id": project_id,
                        "confidence": confidence,
                    },
                )
            return archive_id

        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Failed to archive message",
                    extra={"message_id": message_id, "error": str(e)},
                )
            return None

    def get_archived(
        self, project_id: str, limit: int = 100
    ) -> list[ArchivedMessageRecord]:
        """Get archived messages for a project.

        Args:
            project_id: Project identifier.
            limit: Maximum number of records to return.

        Returns:
            List of ArchivedMessageRecord objects.
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                SELECT id, original_id, project_id, message, replaced_by,
                       reason, confidence, archived_at
                FROM archived_messages
                WHERE project_id = ?
                ORDER BY archived_at DESC
                LIMIT ?
                """,
                (project_id, limit),
            )
            rows = cursor.fetchall()
            cursor.close()

            return [
                ArchivedMessageRecord(
                    id=row[0],
                    original_id=row[1],
                    project_id=row[2],
                    message=row[3],
                    replaced_by=row[4],
                    reason=row[5],
                    confidence=row[6],
                    archived_at=row[7],
                )
                for row in rows
            ]

        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Failed to get archived messages",
                    extra={"project_id": project_id, "error": str(e)},
                )
            return []

    def restore_from_archive(self, archive_id: int) -> Optional[int]:
        """Restore a message from the archive.

        Args:
            archive_id: The archive record ID.

        Returns:
            New message ID if successful, None otherwise.
        """
        try:
            cursor = self.connection.cursor()

            # Get the archived record
            cursor.execute(
                """
                SELECT project_id, message FROM archived_messages WHERE id = ?
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
                cursor.close()
                return None

            project_id, message = row[0], row[1]

            # Insert back into messages table
            cursor.execute(
                "INSERT INTO messages (project_id, message) VALUES (?, ?)",
                (project_id, message),
            )
            new_id = cursor.lastrowid

            # Delete from archive
            cursor.execute(
                "DELETE FROM archived_messages WHERE id = ?",
                (archive_id,),
            )

            self.connection.commit()
            cursor.close()

            if self.logger:
                self.logger.info(
                    "Message restored from archive",
                    extra={
                        "archive_id": archive_id,
                        "new_message_id": new_id,
                        "project_id": project_id,
                    },
                )
            return new_id

        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Failed to restore from archive",
                    extra={"archive_id": archive_id, "error": str(e)},
                )
            return None

    def cleanup_expired_archive(self, ttl_days: int) -> int:
        """Remove archived messages older than TTL.

        Args:
            ttl_days: Number of days to retain archived messages.
                      If 0, no cleanup is performed (permanent archive).

        Returns:
            Number of records deleted.
        """
        if ttl_days <= 0:
            return 0

        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                DELETE FROM archived_messages
                WHERE archived_at < datetime('now', '-' || ? || ' days')
                """,
                (ttl_days,),
            )
            deleted_count = cursor.rowcount
            self.connection.commit()
            cursor.close()

            if self.logger and deleted_count > 0:
                self.logger.info(
                    "Expired archive records cleaned up",
                    extra={
                        "deleted_count": deleted_count,
                        "ttl_days": ttl_days,
                    },
                )
            return deleted_count

        except Exception as e:
            if self.logger:
                self.logger.error(
                    "Failed to cleanup expired archive",
                    extra={"ttl_days": ttl_days, "error": str(e)},
                )
            return 0

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            if self.logger:
                self.logger.debug(
                    "MessageStore closed",
                    extra={"db_path": self.db_path},
                )

    def ensure_initialized(self) -> None:
        """Ensure the store is fully initialized.

        Forces lazy initialization to complete.
        """
        _ = self.connection
