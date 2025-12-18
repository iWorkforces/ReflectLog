## 1. Sprint 1: High Impact, Low Risk Optimizations ✅ COMPLETED

- [x] 1.1 Add new config options to `settings.py`:
  - `EMBEDDING_BATCH_SIZE` (default: 512) → Added in settings.py
  - `EMBEDDING_MAX_CONCURRENT_BATCHES` (default: 4) → Added in settings.py
  - `EMBEDDING_CACHE_ENABLED` (default: true) → Added in settings.py
  - `EMBEDDING_CACHE_SIZE` (default: 100) → Added in settings.py
  - `RERANK_MAX_CONCURRENCY` increase default to 10 → Already configurable

- [x] 1.2 Implement parallel smart replacement in `manager.py:398-455`:
  - Created `_check_for_replacement_parallel()` method using `anyio.create_task_group()`
  - Semaphore-based rate limiting via `RERANK_MAX_CONCURRENCY`
  - Parallel LLM calls for checking multiple candidates
  - 14 unit tests in `TestParallelSmartReplacement` class

- [x] 1.3 Implement async embedding batching in `qwen3_embedding.py:204-240`:
  - Modified `aembed_documents()` to batch texts with configurable batch size
  - Added concurrent batch processing with semaphore limiting
  - Preserves text order in results via indexed collection
  - 5 unit tests in `TestAsyncEmbedDocuments` class

- [x] 1.4 Implement direct message lookup in `manager.py:1250-1277`:
  - Modified `search_for_removal()` to use `get_id_by_message()` for O(log n) lookup
  - Returns single result with indexed database lookup
  - Eliminated O(n) `get_all()` call
  - 4 unit tests for direct lookup behavior

- [x] 1.5 Add unit tests for parallel operations:
  - Test parallel smart replacement with mocked LLM (14 tests)
  - Test async embedding batching with mocked API (5 tests)
  - Test direct message lookup (4 tests)
  - Total: 23 unit tests passing

## 2. Sprint 2: High Impact, Medium Risk Optimizations ✅ COMPLETED

- [x] 2.1 Implement optimized duplicate detection fallback in `manager.py:1371-1419`:
  - Added `MessageStore.exists()` and `MessageStore.get_id_by_message()` methods
  - Replaced semantic search fallback with fast database lookup
  - Eliminated 200-400ms API call on Tantivy failure
  - O(log n) indexed lookup instead of O(n) `get_all()` scan

- [x] 2.2 Implement phased parallel add processing in `manager.py:553-732`:
  - Created three-phase architecture for parallel message processing:
  - Phase 1: Parallel duplicate detection (batch + concurrent checks)
  - Phase 2: Parallel smart replacement LLM calls with semaphore limiting
  - Phase 3: Sequential storage with batch commit for data consistency
  - 8 unit tests in `TestParallelMessageAddition` class

- [x] 2.3 Add integration tests for parallel add:
  - `TestPhasedParallelAdd` class with 4 integration tests:
    - `test_parallel_add_multiple_messages` - bulk add verification
    - `test_parallel_add_batch_deduplication` - input-level dedup
    - `test_parallel_add_storage_deduplication` - storage-level dedup
    - `test_parallel_add_preserves_order` - message ordering
  - All 4 tests passing

- [x] 2.4 Update CLAUDE.md documentation:
  - Performance optimization section already documented (lines 519-594)
  - Covers: phased parallel add, async embedding batching, query cache
  - Covers: direct database lookup, parallel smart replacement
  - All config options documented

## 3. Sprint 3: Tantivy Soft-Delete Implementation ✅ COMPLETED

- [x] 3.1 Add soft-delete config options to `settings.py`:
  - `TANTIVY_SOFT_DELETE_ENABLED` (default: true) → Added as `soft_delete_enabled` in TantivyConfig
  - `TANTIVY_COMPACTION_THRESHOLD_RATIO` (default: 0.2) → Added as `compaction_threshold_ratio`
  - `TANTIVY_COMPACTION_MAX_TOMBSTONES` (default: 10000) → Added as `compaction_max_tombstones`
  - `TANTIVY_TOMBSTONE_TTL_DAYS` (default: 7) → Added as `tombstone_ttl_days`

- [x] 3.2 Add soft-delete schema fields to `tantivy_engine.py`:
  - Added `is_deleted` field (u64, fast=True for columnar access)
  - Added `deleted_at` field (i64, fast=True for columnar access)
  - Updated `_build_schema()` method with V2 schema versioning
  - Schema migration handled via `migrate_to_v2()` method

- [x] 3.3 Implement tombstone-based delete in `tantivy_engine.py:687-743`:
  - Created `soft_delete()` method (O(1) tombstone marking)
  - Marks document with `is_deleted=1` and `deleted_at=<timestamp_ms>`
  - Updated `delete()` to use soft-delete when enabled and schema supports it
  - Fallback to hard delete (rebuild) when soft-delete disabled or V1 schema

- [x] 3.4 Modify search to filter deleted documents in `tantivy_engine.py`:
  - Added `_get_tombstoned_messages()` helper (lines 489-531)
  - Updated `search()` to filter tombstoned messages (lines 533-589)
  - Updated `_get_all_docs()` to exclude tombstoned messages (lines 267-337)
  - Added `get_tombstone_stats()` method for compaction statistics (lines 1048-1133)

- [x] 3.5 Implement `TantivyCompactionService`:
  - Implemented as methods in TantivyEngine (not separate class):
  - `needs_compaction()` - threshold checks (lines 1135-1175)
  - `compact(force=False)` - rebuild index without tombstones (lines 1177-1260)
  - Configurable via `compaction_threshold_ratio` and `compaction_max_tombstones`

- [x] 3.6 Add migration script for existing indexes:
  - Implemented `migrate_to_v2()` method (lines 953-1046)
  - Exports all V1 documents, rebuilds with V2 schema, re-imports
  - Returns migration statistics (migrated, from_version, to_version, doc_count)
  - Safe to call on V2 indexes (no-op)

- [x] 3.7 Add unit tests for soft-delete and compaction:
  - Added 22 new tests in `tests/unit/infrastructure/test_tantivy_engine.py`:
    - `TestTantivySoftDelete` (7 tests): tombstone creation, filtering, edge cases
    - `TestTantivyCompaction` (7 tests): stats, thresholds, compact operation
    - `TestTantivySchemaMigration` (3 tests): V2 detection, version property
    - `TestTantivyTombstoneHelpers` (4 tests): helper methods, edge cases
  - All 60 TantivyEngine tests passing

## 4. Sprint 4: Lower Priority Optimizations

- [x] 4.1 Implement adaptive overfetch multiplier in `manager.py`:
  - Added `_calculate_adaptive_overfetch(limit)` method (lines 932-987)
  - Logarithmic interpolation: small index (≤100) → 3.0x, large (≥10k) → 1.5x
  - Added config options in `settings.py`:
    - `OVERFETCH_ADAPTIVE` (default: true)
    - `OVERFETCH_MIN_MULTIPLIER` (default: 1.5)
    - `OVERFETCH_MAX_MULTIPLIER` (default: 3.0)
  - Updated search method to use adaptive calculation (line 1042)

- [x] 4.2 Implement RRF score averaging in `ranx_fusion.py`:
  - Modified `_convert_to_run()` to aggregate duplicate scores (lines 126-140)
  - Uses sum/count dictionaries to compute averages
  - Existing test `test_duplicate_in_same_list` verifies behavior
  - All 35 fusion tests passing

- [x] 4.3 Implement query embedding LRU cache:
  - Created `CachedEmbeddings` wrapper class in `infrastructure/cached_embeddings.py`
  - Uses MD5 hash of query text as cache key
  - Thread-safe LRU eviction using `OrderedDict` with `threading.Lock`
  - Only caches `embed_query()`, not `embed_documents()` (documents cached once on ingestion)
  - Both sync and async support (`aembed_query()`)
  - Cache stats via `get_cache_stats()` method
  - Exported from `infrastructure/__init__.py`
  - Integrated with `MemoryManager` (lines 117-126 in manager.py)
  - Config options already in `settings.py`:
    - `EMBEDDING_CACHE_ENABLED` (default: true)
    - `EMBEDDING_CACHE_SIZE` (default: 100)

- [x] 4.4 Implement eager connection initialization:
  - Added `eager_initialization` config option in `settings.py` (default: false)
  - Added `EAGER_INITIALIZATION` environment variable parsing
  - Created `_eager_initialize_engines()` method in `manager.py` (lines 228-259)
  - Calls `ensure_initialized()` on USearchEngine and TantivyEngine during init
  - Logs initialization timing for monitoring

- [ ] 4.5 Add unit tests for lower priority changes:
  - Test adaptive overfetch calculation
  - Test RRF score averaging
  - Test embedding cache hit/miss/eviction
  - Test eager initialization

## 5. Final Validation

- [x] 5.1 Run full test suite with coverage:
  - `./start-unittest.sh --coverage`
  - TantivyEngine: 60/60 tests passing
  - Some pre-existing failures in other modules (unrelated to soft-delete)

- [x] 5.2 Run type checking:
  - `./start-type-check.sh --concise`
  - Passes (only pre-existing warning about SmartReplacer)

- [x] 5.3 Run linting:
  - `./start-lint.sh --all`
  - All checks passed, no issues

- [ ] 5.4 Performance benchmark:
  - Measure delete latency before/after soft-delete
  - Measure add throughput before/after parallel processing
  - Measure embedding API call reduction
  - Document results

- [ ] 5.5 Update OpenSpec specs with implementation status
