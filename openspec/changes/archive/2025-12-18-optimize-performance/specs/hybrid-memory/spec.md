## ADDED Requirements

### Requirement: Adaptive Overfetch Multiplier

The hybrid search pipeline SHALL use an adaptive overfetch multiplier based on index size to reduce unnecessary processing overhead.

#### Scenario: Small index optimization

- **WHEN** the index contains fewer than 100 documents
- **AND** `OVERFETCH_ADAPTIVE=true` (default)
- **THEN** the overfetch multiplier SHALL be 1.5x the requested limit
- **AND** this reduces processing overhead for small collections

#### Scenario: Medium index overfetch

- **WHEN** the index contains 100-1000 documents
- **AND** `OVERFETCH_ADAPTIVE=true`
- **THEN** the overfetch multiplier SHALL be 2.0x the requested limit

#### Scenario: Large index overfetch

- **WHEN** the index contains 1000-10000 documents
- **AND** `OVERFETCH_ADAPTIVE=true`
- **THEN** the overfetch multiplier SHALL be 2.5x the requested limit

#### Scenario: Very large index default

- **WHEN** the index contains more than 10000 documents
- **AND** `OVERFETCH_ADAPTIVE=true`
- **THEN** the overfetch multiplier SHALL be 3.0x the requested limit (current default)

#### Scenario: Adaptive overfetch disabled

- **WHEN** `OVERFETCH_ADAPTIVE=false` is configured
- **THEN** the fixed multiplier of 3.0x SHALL be used regardless of index size

### Requirement: RRF Duplicate Score Averaging

The RRF fusion engine SHALL average scores when the same document appears multiple times within a single result set.

#### Scenario: Duplicate document score aggregation

- **WHEN** the same document appears multiple times in an engine's result set
- **THEN** the scores SHALL be averaged instead of keeping the first occurrence
- **AND** this preserves the signal from all occurrences

#### Scenario: Cross-engine duplicate handling

- **WHEN** the same document appears in both USearch and Tantivy results
- **THEN** RRF fusion SHALL combine rankings as normal (existing behavior)
- **AND** within-engine duplicate averaging SHALL occur before cross-engine fusion

### Requirement: Query Embedding Cache

The hybrid search pipeline SHALL cache query embeddings to reduce API call overhead for repeated searches.

#### Scenario: Cache hit for repeated query

- **WHEN** a search query has been executed recently
- **AND** `EMBEDDING_CACHE_ENABLED=true` (default)
- **THEN** the cached embedding SHALL be returned without API call
- **AND** cache lookup SHALL use MD5 hash of query text as key

#### Scenario: Cache miss behavior

- **WHEN** a search query is not in cache
- **THEN** embedding SHALL be generated via API call
- **AND** result SHALL be stored in cache for future use

#### Scenario: LRU eviction

- **WHEN** cache reaches `EMBEDDING_CACHE_SIZE` limit (default: 100)
- **AND** a new entry needs to be cached
- **THEN** the least recently used entry SHALL be evicted
- **AND** the new entry SHALL be added to cache

#### Scenario: Cache disabled

- **WHEN** `EMBEDDING_CACHE_ENABLED=false` is configured
- **THEN** embeddings SHALL always be generated via API call
- **AND** no caching overhead SHALL be incurred

### Requirement: Direct Message Lookup for Removal

The remove tool SHALL use indexed database lookup instead of memory-scanning for exact matches.

#### Scenario: Efficient exact match lookup

- **WHEN** `search_for_removal()` is called with a query
- **THEN** the system SHALL use `get_id_by_message()` for O(log n) indexed lookup
- **AND** SHALL NOT load all messages into memory

#### Scenario: Message not found

- **WHEN** the query does not match any stored message
- **THEN** an empty result list SHALL be returned
- **AND** no memory allocation for full message list SHALL occur

### Requirement: Optimized Duplicate Detection Fallback

The duplicate detection system SHALL use MessageStore as fallback instead of semantic search when Tantivy is unavailable.

#### Scenario: Tantivy failure fallback

- **WHEN** Tantivy exact match check fails
- **THEN** MessageStore direct lookup SHALL be used as fallback
- **AND** semantic search (embedding API call) SHALL NOT be used
- **AND** fallback latency SHALL be <10ms instead of 200-400ms

#### Scenario: Fallback chain exhausted

- **WHEN** both Tantivy and MessageStore lookups fail
- **THEN** the system SHALL log a warning
- **AND** SHALL proceed without deduplication (allow the add)
- **AND** SHALL NOT block the operation

### Requirement: Increased Default Reranking Concurrency

The LLM reranking system SHALL use higher default concurrency to improve throughput.

#### Scenario: Default concurrency value

- **WHEN** `RERANK_MAX_CONCURRENCY` is not explicitly configured
- **THEN** the default value SHALL be 10 (increased from 5)
- **AND** this allows more parallel LLM calls for faster reranking

#### Scenario: Custom concurrency

- **WHEN** `RERANK_MAX_CONCURRENCY` is explicitly set
- **THEN** the configured value SHALL be used
- **AND** values from 1 to 20 SHALL be supported
