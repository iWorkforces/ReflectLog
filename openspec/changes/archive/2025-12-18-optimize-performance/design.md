## Context

OpenMemoriesMCP combines USearch (semantic) and Tantivy (full-text) for hybrid search. The current implementation has performance bottlenecks that become severe at moderate scale (10k+ documents):

- Tantivy delete is O(n) due to index rebuild limitation
- Smart replacement makes sequential LLM calls
- Async embedding doesn't batch requests
- Add operations process messages one-at-a-time

The user has requested aggressive optimizations with breaking changes acceptable.

## Goals / Non-Goals

**Goals**:
- Reduce Tantivy delete latency from O(n) to O(1)
- Parallelize LLM calls for smart replacement (3-6x speedup)
- Batch async embeddings (100x reduction in API calls)
- Enable phased parallel add processing (5-8x speedup)
- Maintain backward compatibility via feature flags

**Non-Goals**:
- Change MCP tool API contracts
- Replace USearch or Tantivy with different engines
- Add new MCP tools (except optional `compact()` for Tantivy)

## Decisions

### Decision 1: Tombstone-Based Soft Delete for Tantivy

**What**: Instead of rebuilding the index on delete, mark documents as deleted with tombstone records.

**Why**: Tantivy-py's `delete_documents()` consumes the IndexWriter, making commit impossible. Soft-delete avoids this limitation.

**Schema Change**:
```
Field: is_deleted (raw tokenizer) - "0" or "1"
Field: deleted_at (raw tokenizer) - timestamp or ""
```

**Alternatives considered**:
- Switch to Elasticsearch (rejected: adds operational complexity)
- Accept O(n) performance (rejected: unacceptable at scale)
- Batch deletes with deferred rebuild (rejected: still O(n) per batch)

### Decision 2: Background Compaction Service

**What**: New `TantivyCompactionService` monitors tombstone ratio and triggers compaction.

**Trigger conditions** (any):
- Tombstone ratio > 20% of total documents
- Tombstone count > 10,000
- Manual trigger via optional `compact()` tool

**Why**: Prevents unbounded tombstone growth while keeping delete O(1).

### Decision 3: Parallel Smart Replacement with Semaphore

**What**: Use `anyio.create_task_group()` with semaphore to check replacement candidates in parallel.

**Why**: Reuses existing pattern from LLMReranker (proven, tested).

**Concurrency**: Reuse `RERANK_MAX_CONCURRENCY` setting (shared API rate limit concern).

### Decision 4: Three-Phase Parallel Add Processing

**What**: Restructure `add_messages_async()` into three phases:
1. **Phase 1**: Batch embed all messages in parallel
2. **Phase 2**: Check replacements in parallel using pre-computed embeddings
3. **Phase 3**: Sequential storage (required by SQLite/libSQL)

**Why**: Phase 3 must be sequential due to database write serialization, but phases 1 and 2 can be fully parallel.

### Decision 5: Async Embedding Batching

**What**: Modify `aembed_documents()` to batch texts (512 per request) like the sync version.

**Why**: OpenRouter API supports up to 2048 items per request. Current implementation sends 1 text per request.

**Configuration**:
```
EMBEDDING_BATCH_SIZE=512
EMBEDDING_MAX_CONCURRENT_BATCHES=4
```

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|------------|
| Soft-delete increases index size | Medium | Compaction service with configurable thresholds |
| Parallel LLM calls may hit rate limits | Low | Reuse existing semaphore/concurrency config |
| Migration breaks existing indexes | Medium | Feature flag + migration script |
| Phased add increases code complexity | Medium | Thorough testing, maintain fallback path |
| Tombstone queries add search overhead | Low | Simple AND filter, negligible cost |

## Migration Plan

### Phase 1: Non-Breaking Optimizations (Deploy First)
1. Parallel smart replacement
2. Async embedding batching
3. Direct message lookup
4. Optimized fallback

### Phase 2: Tantivy Soft-Delete
1. Deploy with `TANTIVY_SOFT_DELETE_ENABLED=false`
2. Run migration script to add new fields to existing indexes
3. Enable soft delete via config change
4. Monitor tombstone growth
5. Enable compaction service

### Rollback Strategy
- Soft-delete can be disabled via config flag
- Old indexes work with both modes
- Compaction recreates clean index if needed

## Open Questions

1. **Compaction scheduling**: Should compaction run automatically in background thread, or only via explicit trigger?
   - Recommendation: Automatic with configurable thresholds, plus manual trigger option

2. **Migration tooling**: Should we provide CLI command for migration, or automatic on-startup detection?
   - Recommendation: Automatic detection + optional CLI for explicit control

3. **Embedding cache TTL**: Should cached embeddings expire, or use LRU eviction only?
   - Recommendation: LRU eviction only (simpler, embeddings don't change)
