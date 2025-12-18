## ADDED Requirements

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
