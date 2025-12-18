# add-tool Specification

## Purpose
TBD - created by archiving change add-smart-memory-replacement. Update Purpose after archive.
## Requirements
### Requirement: Smart Memory Replacement

The add tool SHALL detect when a new memory semantically replaces an existing memory and automatically remove the old one before storing the new one.

#### Scenario: Replacement detected and executed

- **WHEN** a new memory is added that updates or contradicts an existing memory
- **AND** the LLM confidence score meets the threshold (default: 0.7)
- **THEN** the old memory SHALL be deleted before adding the new one
- **AND** detailed logs SHALL show old memory preview, new memory preview, confidence score, and reason

#### Scenario: No replacement needed

- **WHEN** a new memory is added that does not semantically replace any existing memory
- **THEN** the memory SHALL be added normally without any deletion

#### Scenario: Smart replacement disabled

- **WHEN** `ENABLE_SMART_REPLACE=false` is configured
- **THEN** no replacement detection SHALL occur
- **AND** memories SHALL be added normally with existing deduplication still applied

#### Scenario: LLM failure graceful degradation

- **WHEN** the LLM call for replacement detection fails
- **THEN** a warning SHALL be logged with the error details
- **AND** the memory SHALL be added normally without replacement

### Requirement: Smart Replacement Configuration

The smart memory replacement feature SHALL be configurable via environment variables.

#### Scenario: Default configuration

- **WHEN** no smart replacement environment variables are set
- **THEN** `ENABLE_SMART_REPLACE` SHALL default to `true`
- **AND** `SMART_REPLACE_THRESHOLD` SHALL default to `0.7`

#### Scenario: Disable smart replacement

- **WHEN** `ENABLE_SMART_REPLACE=false` is set
- **THEN** the SmartReplacer component SHALL NOT be initialized
- **AND** no LLM calls for replacement detection SHALL be made

#### Scenario: Custom threshold

- **WHEN** `SMART_REPLACE_THRESHOLD` is set to a value between 0.0 and 1.0
- **THEN** the system SHALL use that threshold for replacement decisions
- **AND** only replacements with confidence >= threshold SHALL be executed

### Requirement: Smart Replacement Logging

The smart memory replacement feature SHALL provide detailed logging for transparency and debugging.

#### Scenario: Replacement executed logging

- **WHEN** a replacement is detected and executed
- **THEN** the log SHALL include the old memory preview (truncated to 80 chars)
- **AND** the log SHALL include the new memory preview (truncated to 80 chars)
- **AND** the log SHALL include the LLM confidence score
- **AND** the log SHALL include the LLM reason for the decision

#### Scenario: No replacement logging

- **WHEN** a replacement check is performed but no replacement is needed
- **THEN** the system SHALL log at DEBUG level with confidence score and reason

### Requirement: Parallel Smart Replacement Detection

The add tool SHALL detect replacement candidates in parallel to reduce latency for multi-candidate scenarios.

#### Scenario: Multiple candidates checked in parallel

- **WHEN** a new memory is added with 3 or more similar existing memories above similarity threshold
- **THEN** replacement detection LLM calls SHALL execute in parallel using semaphore-controlled concurrency
- **AND** concurrency SHALL be limited by `RERANK_MAX_CONCURRENCY` setting (default: 10)
- **AND** results SHALL be collected and processed in original candidate order

#### Scenario: Single candidate sequential behavior

- **WHEN** a new memory has only 1 similar existing memory
- **THEN** replacement detection SHALL proceed without parallel overhead
- **AND** behavior SHALL be identical to pre-optimization implementation

#### Scenario: Parallel check failure handling

- **WHEN** one or more parallel replacement checks fail
- **THEN** failures SHALL be logged as warnings
- **AND** successful checks SHALL still be processed
- **AND** the add operation SHALL NOT fail due to individual check failures

### Requirement: Phased Parallel Add Processing

The add tool SHALL process multiple messages using a three-phase parallel architecture.

#### Scenario: Phase 1 - Batch embedding generation

- **WHEN** multiple messages are added in a single call
- **THEN** all message embeddings SHALL be generated in a single batched API call
- **AND** batch size SHALL be configurable via `EMBEDDING_BATCH_SIZE` (default: 512)
- **AND** concurrent batches SHALL be limited by `EMBEDDING_MAX_CONCURRENT_BATCHES` (default: 4)

#### Scenario: Phase 2 - Parallel replacement detection

- **WHEN** embeddings are generated for all messages
- **THEN** replacement detection SHALL run in parallel for all messages
- **AND** pre-computed embeddings SHALL be reused (no re-embedding)
- **AND** similarity search SHALL use pre-computed embeddings directly

#### Scenario: Phase 3 - Sequential storage

- **WHEN** replacement detection completes for all messages
- **THEN** storage operations SHALL execute sequentially
- **AND** this ensures SQLite/libSQL write consistency
- **AND** batch commit SHALL occur after all messages are stored

#### Scenario: Single message optimization bypass

- **WHEN** only a single message is added
- **THEN** the phased parallel architecture MAY be bypassed for simplicity
- **AND** behavior SHALL be functionally identical to the parallel path

### Requirement: Async Embedding Batching

The add tool SHALL use batched async embedding requests to reduce API call overhead.

#### Scenario: Batch embedding request

- **WHEN** embedding multiple texts asynchronously
- **THEN** texts SHALL be grouped into batches of `EMBEDDING_BATCH_SIZE` (default: 512)
- **AND** batch requests SHALL execute with limited concurrency via `EMBEDDING_MAX_CONCURRENT_BATCHES` (default: 4)
- **AND** result order SHALL match input text order

#### Scenario: Partial batch handling

- **WHEN** the number of texts is not evenly divisible by batch size
- **THEN** the final batch SHALL contain remaining texts
- **AND** no empty batches SHALL be created

