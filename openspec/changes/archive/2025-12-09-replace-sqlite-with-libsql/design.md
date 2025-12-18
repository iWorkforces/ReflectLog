# Design: Replace SQLite with libSQL

## Context

`MessageStore` provides SQLite-backed message text storage for the USearch semantic search engine. Messages are stored with auto-increment IDs that serve as USearch vector keys. The current implementation uses Python's built-in `sqlite3` module with WAL mode.

**Stakeholders**: OpenMemoriesMCP users, AI agents using MCP tools

**Constraints**:
- Must maintain backward compatibility with existing database files
- Must preserve all current MessageStore operations (insert, get, get_all, delete, exists)
- Must not break USearch key relationship (SQLite ID = USearch key)
- Must maintain thread-safe operation

## Goals / Non-Goals

### Goals
- Improve concurrent write performance via MVCC
- Reduce connection opening latency
- Maintain API compatibility with existing code
- Minimal migration effort

### Non-Goals
- Adding cloud sync capabilities (Turso remote features)
- Changing the database schema
- Adding new MessageStore operations
- Async API conversion (stay synchronous)

## Decisions

### Decision 1: Use libSQL's synchronous API

**What**: Use `libsql.connect()` for local file mode, not the async client.

**Why**:
- MessageStore is currently synchronous
- Async conversion would require changes to USearchEngine and MemoryManager
- Local file mode matches current SQLite usage pattern

**Alternatives considered**:
- Async libSQL client: Would require broader code changes, rejected for scope creep
- libsql-experimental: Deprecated, use main `libsql` package

### Decision 2: Keep WAL mode configuration

**What**: Retain `PRAGMA journal_mode=WAL` and `PRAGMA synchronous=NORMAL`.

**Why**:
- WAL mode provides good read concurrency
- libSQL supports SQLite PRAGMA statements
- No reason to change what works

### Decision 3: Remove check_same_thread parameter

**What**: Remove `check_same_thread=False` from connection creation.

**Why**:
- libSQL handles multi-threading natively with MVCC
- Parameter not supported by libSQL connect API
- Thread safety is built-in

### Decision 4: No schema changes

**What**: Keep existing schema unchanged:
```sql
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**Why**:
- libSQL is SQLite-compatible
- Existing indexes and constraints work as-is
- No need to migrate data

## Risks / Trade-offs

### Risk 1: libSQL API differences
**Risk**: libSQL Python API may have subtle differences from sqlite3.
**Mitigation**:
- Run comprehensive test suite
- Test cursor operations explicitly
- Document any API differences found

### Risk 2: Performance regression
**Risk**: libSQL could be slower for some operations.
**Mitigation**:
- Benchmark before/after for key operations (insert, get, get_all)
- Rollback plan documented in tasks.md
- Keep sqlite3 code commented for quick revert

### Risk 3: Dependency stability
**Risk**: libSQL Python package is newer than sqlite3 (built-in).
**Mitigation**:
- Pin version in pyproject.toml (`libsql>=0.5.0`)
- Monitor Turso releases for breaking changes
- Built-in sqlite3 remains available as fallback

## Migration Plan

### Phase 1: Code Changes (1 PR)
1. Add libsql dependency
2. Update MessageStore implementation
3. Update tests
4. Update documentation

### Phase 2: Verification
1. Run full test suite
2. Manual smoke test with MCP server
3. Verify database compatibility (existing .db files work)

### Rollback
1. Revert to sqlite3 import
2. Remove libsql dependency
3. Existing database files remain compatible (SQLite format)

## Open Questions

1. **Connection pooling**: Should we implement connection pooling with libSQL?
   - Current answer: No, keep simple single-connection model
   - Future consideration if concurrency issues arise

2. **Type stubs**: Does libSQL have official type stubs?
   - If not, may need to add to `stubs/` directory
   - Check `types-libsql` or `libsql-stubs` packages

## Appendix: API Comparison

### sqlite3 (current)
```python
import sqlite3
conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
cursor = conn.cursor()
cursor.execute("INSERT INTO messages ...")
row_id = cursor.lastrowid
cursor.close()
conn.close()
```

### libSQL (proposed)
```python
import libsql
conn = libsql.connect(path)
cursor = conn.cursor()
cursor.execute("INSERT INTO messages ...")
conn.commit()  # May need explicit commit
row_id = cursor.lastrowid
cursor.close()
conn.close()
```

Key differences to verify:
- Auto-commit behavior
- Transaction handling
- Cursor lifecycle
