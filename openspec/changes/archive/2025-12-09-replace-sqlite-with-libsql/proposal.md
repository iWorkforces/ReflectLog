# Change: Replace SQLite with libSQL for MessageStore

## Why

The current `MessageStore` uses Python's built-in `sqlite3` module. While SQLite is reliable, it has a single-writer bottleneck that limits concurrent write performance. libSQL is a modern fork of SQLite by Turso that provides:
- MVCC (Multi-Version Concurrency Control) for concurrent writes
- 575x faster connection opening (Turso benchmark)
- Drop-in SQLite compatibility with minimal code changes
- Active development with performance improvements

## What Changes

- **Replace `sqlite3` with `libsql`** in `MessageStore` (`openmemories/infrastructure/message_store.py`)
- **Add `libsql` dependency** to `pyproject.toml`
- **Update connection handling** to use libSQL's synchronous API
- **Remove `check_same_thread=False`** (libSQL handles this natively)
- **Update type stubs** for libSQL if needed

## Impact

- **Affected specs**: message-storage (new capability)
- **Affected code**:
  - `openmemories/infrastructure/message_store.py` - Primary change
  - `pyproject.toml` - Add libsql dependency
  - `tests/` - Update test fixtures

## Research Summary

### Alternatives Evaluated

| Database | Type | Read Perf | Write Perf | SQL | Migration Effort | Verdict |
|----------|------|-----------|------------|-----|------------------|---------|
| SQLite (current) | Row-based | Good | Good (WAL) | Full | N/A | Baseline |
| **libSQL** | Row-based | Good | Better (MVCC) | Full | **Low** | **Selected** |
| LMDB | Key-Value | Excellent | Good | None | High | Rejected |
| DuckDB | Columnar | Excellent (OLAP) | Poor (OLTP) | Full | Medium | Rejected |
| RocksDB | LSM-tree | Good | Excellent | None | High | Rejected |

### Why libSQL

1. **Drop-in replacement**: Same SQL syntax, similar connection API
2. **Concurrent writes**: MVCC eliminates single-writer bottleneck
3. **Faster connections**: 575x faster connection opening
4. **All current features work**: Auto-increment IDs, indexes, unique constraints
5. **Minimal code changes**: Change import from `sqlite3` to `libsql`

### Why NOT Other Alternatives

- **LMDB**: Key-value only, requires reimplementing SQL features (indexes, deduplication, range queries)
- **DuckDB**: Optimized for OLAP analytics, 2-500x slower for transactional writes
- **RocksDB**: Key-value only, complex setup, overkill for simple OLTP workload

## Migration Path

libSQL provides a synchronous Python API that closely mirrors sqlite3:

```python
# Before (current)
import sqlite3
conn = sqlite3.connect(db_path, check_same_thread=False)

# After (libSQL)
import libsql
conn = libsql.connect(db_path)
```

The MessageStore's existing SQL queries, schema, and business logic remain unchanged.

## References

- [libSQL GitHub](https://github.com/tursodatabase/libsql)
- [Turso Python SDK](https://docs.turso.tech/sdk/python/quickstart)
- [libSQL vs SQLite Performance](https://turso.tech/blog/how-turso-made-connections-faster)
