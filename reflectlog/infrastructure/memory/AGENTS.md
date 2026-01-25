# Agent Guidelines for reflectlog/infrastructure/memory/

This directory is a placeholder for memory storage implementations.

## Directory Structure

```
memory/
└── __init__.py            # Placeholder - re-exports from parent
```

## Current Status

This directory is currently empty. Memory storage implementations are located in the parent `infrastructure/` directory and re-exported here for backward compatibility when this directory is populated in the future.

## Re-Exports

The parent `infrastructure/` directory provides:
- `MessageStore`: libSQL-based message storage
- `SmartReplacer`: LLM-based memory replacement detection

## Future Expansion

When implementing new memory storage backends, add them to this directory:
1. Create backend-specific files (e.g., `redis.py`, `postgres.py`, `s3.py`)
2. Implement `IMemoryStore` protocol from `core/memory.py`
3. Re-export in `__init__.py`
4. Update this documentation

## Key Patterns

### Storage Backend Protocol

All memory storage backends must implement `IMemoryStore` protocol:

```python
@runtime_checkable
class IMemoryStore(Protocol):
    """Protocol for memory storage operations."""

    def add(self, message: str) -> str:
        """Add a message to storage, return ID."""
        ...

    def search(self, query: str, limit: int) -> list[tuple[str, float, str]]:
        """Search for messages matching query."""
        ...

    def get_all(self) -> list[str]:
        """Get all stored messages."""
        ...

    def delete(self, memory_id: str) -> bool:
        """Delete a message by ID."""
        ...
```

### Soft-Delete Pattern

Implement soft-deletion for data recovery:

```python
def delete(self, memory_id: str) -> bool:
    """Soft-delete a message."""
    self._conn.execute(
        "UPDATE messages SET is_deleted = 1, deleted_at = ? WHERE id = ?",
        (datetime.now(timezone.utc).isoformat(), memory_id)
    )
```

## Dependencies

### Internal Dependencies

- `core/memory.py`: `IMemoryStore` protocol
- `application/config/`: Configuration dataclasses
- `application/exceptions.py`: `MemoryStorageError` exception

### External Dependencies

- `libsql`: SQLite-compatible database
- `redis`: Redis client (when implemented)
- `sqlalchemy`: SQL ORM (when implemented)
