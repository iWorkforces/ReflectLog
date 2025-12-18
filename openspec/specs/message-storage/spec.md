# Message Storage Capability

## Purpose

Provide persistent message text storage for USearch vector keys using libSQL, with support for concurrent read/write operations via MVCC.
## Requirements
### Requirement: libSQL-based Message Storage

The MessageStore component SHALL use libSQL as the underlying database engine for storing message text records.

#### Scenario: Database connection with libSQL
- **WHEN** MessageStore is initialized with a database path
- **THEN** a libSQL connection SHALL be established to the specified file
- **AND** the connection SHALL support multi-threaded access via MVCC

#### Scenario: Auto-increment ID generation
- **WHEN** a message is inserted into the store
- **THEN** libSQL SHALL generate an auto-increment integer ID
- **AND** this ID SHALL be returned for use as USearch vector key

#### Scenario: WAL mode configuration
- **WHEN** the database connection is established
- **THEN** WAL journal mode SHALL be enabled via PRAGMA
- **AND** synchronous mode SHALL be set to NORMAL for performance

#### Scenario: Schema compatibility
- **WHEN** MessageStore creates its schema
- **THEN** the following table structure SHALL be created:
  - `id`: INTEGER PRIMARY KEY AUTOINCREMENT
  - `project_id`: TEXT NOT NULL (indexed)
  - `message`: TEXT NOT NULL
  - `created_at`: TIMESTAMP DEFAULT CURRENT_TIMESTAMP
- **AND** a unique constraint SHALL exist on (project_id, message) for deduplication

#### Scenario: Auto-migration from legacy schema
- **WHEN** MessageStore opens a database with legacy `user_id` column
- **THEN** the column SHALL be automatically renamed to `project_id`
- **AND** all indices referencing `user_id` SHALL be recreated with `project_id`
- **AND** all existing data SHALL be preserved

### Requirement: Concurrent Write Support

The MessageStore SHALL support concurrent write operations without blocking readers.

#### Scenario: Multiple concurrent inserts
- **WHEN** multiple threads attempt to insert messages simultaneously
- **THEN** libSQL's MVCC SHALL allow concurrent writes
- **AND** no thread SHALL block waiting for another writer
- **AND** all inserts SHALL complete successfully with unique IDs

#### Scenario: Read during write
- **WHEN** a read operation occurs while a write is in progress
- **THEN** the read SHALL NOT be blocked
- **AND** the read SHALL return consistent data (snapshot isolation)

### Requirement: SQLite Database File Compatibility

The MessageStore SHALL maintain compatibility with existing SQLite database files.

#### Scenario: Open existing database
- **WHEN** MessageStore opens an existing SQLite database file created by sqlite3
- **THEN** libSQL SHALL read the file successfully
- **AND** all existing records SHALL be accessible

#### Scenario: Database file format
- **WHEN** MessageStore creates or modifies a database
- **THEN** the file SHALL use standard SQLite file format
- **AND** the file SHALL be readable by both libSQL and sqlite3

### Requirement: Tantivy Soft-Delete Schema

The Tantivy full-text search engine SHALL support soft-delete via tombstone records instead of index rebuild.

#### Scenario: Soft-delete schema fields

- **WHEN** `TANTIVY_SOFT_DELETE_ENABLED=true` (default)
- **THEN** the Tantivy schema SHALL include an `is_deleted` field (raw tokenizer, values "0" or "1")
- **AND** the schema SHALL include a `deleted_at` field (raw tokenizer, ISO timestamp or empty string)

#### Scenario: Soft-delete operation

- **WHEN** a document is deleted from Tantivy
- **AND** `TANTIVY_SOFT_DELETE_ENABLED=true`
- **THEN** the document SHALL be marked with `is_deleted="1"` and `deleted_at` timestamp
- **AND** the operation SHALL complete in O(1) time
- **AND** no index rebuild SHALL occur

#### Scenario: Hard-delete fallback

- **WHEN** `TANTIVY_SOFT_DELETE_ENABLED=false` is configured
- **THEN** the original O(n) index rebuild behavior SHALL be used
- **AND** schema SHALL NOT include soft-delete fields

#### Scenario: Search filters tombstones

- **WHEN** a search query is executed
- **AND** `TANTIVY_SOFT_DELETE_ENABLED=true`
- **THEN** the query SHALL automatically filter out documents with `is_deleted="1"`
- **AND** tombstoned documents SHALL NOT appear in search results

#### Scenario: get_all filters tombstones

- **WHEN** `get_all_messages()` is called on Tantivy
- **AND** `TANTIVY_SOFT_DELETE_ENABLED=true`
- **THEN** documents with `is_deleted="1"` SHALL be excluded
- **AND** only active documents SHALL be returned

### Requirement: Tantivy Compaction Service

The system SHALL provide a compaction service to periodically remove tombstoned documents and reclaim space.

#### Scenario: Tombstone ratio threshold

- **WHEN** the ratio of tombstoned documents exceeds `TANTIVY_COMPACTION_THRESHOLD_RATIO` (default: 0.2)
- **THEN** compaction SHALL be triggered
- **AND** all tombstoned documents SHALL be permanently removed
- **AND** the index SHALL be rebuilt without tombstones

#### Scenario: Tombstone count threshold

- **WHEN** the count of tombstoned documents exceeds `TANTIVY_COMPACTION_MAX_TOMBSTONES` (default: 10000)
- **THEN** compaction SHALL be triggered regardless of ratio
- **AND** this prevents unbounded tombstone growth

#### Scenario: Manual compaction trigger

- **WHEN** compaction is explicitly requested
- **THEN** compaction SHALL run regardless of threshold values
- **AND** SHALL return statistics about documents removed

#### Scenario: Compaction statistics

- **WHEN** compaction completes
- **THEN** logs SHALL include tombstone count before compaction
- **AND** logs SHALL include total documents after compaction
- **AND** logs SHALL include duration of compaction operation

### Requirement: Tantivy Schema Migration

The system SHALL support migration of existing Tantivy indexes to the new soft-delete schema.

#### Scenario: Automatic migration detection

- **WHEN** the server starts with an existing Tantivy index
- **AND** the index uses the old schema (no soft-delete fields)
- **AND** `TANTIVY_SOFT_DELETE_ENABLED=true`
- **THEN** migration SHALL be automatically detected as needed
- **AND** a warning log SHALL indicate migration is required

#### Scenario: Migration execution

- **WHEN** migration is triggered (automatic or manual)
- **THEN** all existing documents SHALL be extracted
- **AND** index SHALL be rebuilt with new schema
- **AND** all documents SHALL be re-added with `is_deleted="0"` and empty `deleted_at`
- **AND** this is a one-time operation per index

#### Scenario: Migration failure handling

- **WHEN** migration fails during execution
- **THEN** the original index SHALL be preserved
- **AND** error logs SHALL indicate the failure reason
- **AND** the system SHALL fall back to hard-delete mode

### Requirement: Eager Connection Initialization

The storage engines SHALL support eager initialization to reduce first-request latency.

#### Scenario: Eager initialization enabled

- **WHEN** `EAGER_INITIALIZATION=true` (default)
- **THEN** database connections SHALL be established during server startup
- **AND** first request SHALL NOT incur connection initialization overhead

#### Scenario: Lazy initialization fallback

- **WHEN** `EAGER_INITIALIZATION=false` is configured
- **THEN** connections SHALL be initialized on first use (current behavior)
- **AND** first request latency SHALL include connection setup

### Requirement: MessageStore Existence Check

The MessageStore SHALL provide an efficient method to check if a message exists without retrieving it.

#### Scenario: Existence check

- **WHEN** `exists(project_id, message)` is called
- **THEN** the system SHALL return boolean indicating existence
- **AND** SHALL use indexed lookup (O(log n) complexity)
- **AND** SHALL NOT retrieve or return the message content

#### Scenario: Non-existent message

- **WHEN** `exists()` is called for a message not in storage
- **THEN** `False` SHALL be returned
- **AND** lookup SHALL complete in O(log n) time

## Rationale

### Why libSQL over sqlite3

libSQL is chosen over Python's built-in sqlite3 for the following reasons:

1. **Concurrent writes**: libSQL implements MVCC, eliminating SQLite's single-writer bottleneck
2. **Connection performance**: 575x faster connection opening (Turso benchmark)
3. **Drop-in compatibility**: Same SQL syntax, similar Python API
4. **Active development**: Modern fork with ongoing performance improvements

### Alternatives Rejected

- **LMDB**: Key-value only, requires manual SQL-like feature implementation
- **DuckDB**: OLAP-optimized, 2-500x slower for transactional writes
- **RocksDB**: Key-value only, complex setup, overkill for simple OLTP
