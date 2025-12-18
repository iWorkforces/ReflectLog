## MODIFIED Requirements

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
