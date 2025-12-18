"""Type stubs for libsql package.

libSQL is a high-performance SQLite fork by Turso with MVCC support.
"""

from typing import Any, Iterator, Sequence

class Cursor:
    """libSQL database cursor."""

    @property
    def lastrowid(self) -> int | None:
        """The row ID of the last inserted row."""
        ...

    @property
    def rowcount(self) -> int:
        """The number of rows affected by the last operation."""
        ...

    def execute(self, sql: str, parameters: Sequence[Any] = ...) -> Cursor:
        """Execute a SQL statement."""
        ...

    def executemany(
        self, sql: str, seq_of_parameters: Sequence[Sequence[Any]]
    ) -> Cursor:
        """Execute a SQL statement against all parameter sequences."""
        ...

    def fetchone(self) -> tuple[Any, ...] | None:
        """Fetch the next row of a query result set."""
        ...

    def fetchall(self) -> list[tuple[Any, ...]]:
        """Fetch all remaining rows of a query result set."""
        ...

    def fetchmany(self, size: int = ...) -> list[tuple[Any, ...]]:
        """Fetch the next set of rows of a query result set."""
        ...

    def close(self) -> None:
        """Close the cursor."""
        ...

    def __iter__(self) -> Iterator[tuple[Any, ...]]:
        """Iterate over the result set."""
        ...

    def __next__(self) -> tuple[Any, ...]:
        """Return the next row."""
        ...

class Connection:
    """libSQL database connection."""

    def cursor(self) -> Cursor:
        """Create a new cursor object."""
        ...

    def execute(self, sql: str, parameters: Sequence[Any] = ...) -> Cursor:
        """Execute a SQL statement and return a cursor."""
        ...

    def executemany(
        self, sql: str, seq_of_parameters: Sequence[Sequence[Any]]
    ) -> Cursor:
        """Execute a SQL statement against all parameter sequences."""
        ...

    def commit(self) -> None:
        """Commit the current transaction."""
        ...

    def rollback(self) -> None:
        """Roll back the current transaction."""
        ...

    def close(self) -> None:
        """Close the database connection."""
        ...

    def sync(self) -> None:
        """Sync with remote database (for embedded replicas)."""
        ...

    def __enter__(self) -> "Connection":
        """Context manager entry."""
        ...

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        ...

class Error(Exception):
    """Base exception for libSQL errors."""

    ...

# Module-level constants
LEGACY_TRANSACTION_CONTROL: bool
paramstyle: str
sqlite_version_info: tuple[int, int, int]

def connect(
    database: str,
    *,
    sync_url: str | None = ...,
    auth_token: str | None = ...,
    sync_interval: int | None = ...,
    encryption_key: str | None = ...,
) -> Connection:
    """Connect to a libSQL database.

    Args:
        database: Database file path or URL.
        sync_url: URL for syncing with remote database (embedded replicas).
        auth_token: Authentication token for remote database.
        sync_interval: Auto-sync interval in seconds.
        encryption_key: Encryption key for encrypted databases.

    Returns:
        A Connection object.
    """
    ...
