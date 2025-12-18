# Message Storage Capability

## ADDED Requirements

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
  - `user_id`: TEXT NOT NULL (indexed)
  - `message`: TEXT NOT NULL
  - `created_at`: TIMESTAMP DEFAULT CURRENT_TIMESTAMP
- **AND** a unique constraint SHALL exist on (user_id, message) for deduplication

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

## ADDED Rationale

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
