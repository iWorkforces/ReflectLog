# Change: Optimize Performance Across Search and Add Operations

## Why

OpenMemoriesMCP has several critical performance bottlenecks that significantly impact user experience:

1. **Tantivy delete rebuilds entire index** - O(n) operation taking ~50 seconds for 100k documents
2. **Sequential LLM calls** in smart replacement - 3 candidates = 3-6 seconds sequentially
3. **Async embedding sends 1 text per API request** instead of batching (100x more API calls than needed)
4. **Sequential message processing** in add operations causes 5-8x slowdown for bulk adds

These bottlenecks create unacceptable latency for production workloads with moderate data volumes.

## What Changes

### HIGH PRIORITY - Critical Bottlenecks

- **Tantivy Soft-Delete** - Replace O(n) index rebuild with O(1) tombstone marking + background compaction
- **Parallel Smart Replacement** - Check replacement candidates in parallel using semaphore (like LLMReranker)
- **Async Embedding Batching** - Send 512 texts per API request instead of 1
- **Phased Parallel Add** - Three-phase parallel processing: embed → detect replacements → store

### MEDIUM PRIORITY - Search Pipeline

- **Adaptive Overfetch** - Adjust multiplier based on index size (15-20% latency reduction)
- **RRF Score Averaging** - Average duplicate scores instead of keeping first
- **Direct Message Lookup** - Use indexed database lookup instead of O(n) memory scan for removal
- **Optimized Fallback** - Use MessageStore instead of semantic search when Tantivy fails

### LOWER PRIORITY - Configuration & Caching

- Increase default reranking concurrency from 5 to 10
- Add query embedding LRU cache
- Implement batch database operations
- Add eager connection initialization option

### **BREAKING** Schema Changes

- **Tantivy Index Schema**: New fields `is_deleted` and `deleted_at` for soft-delete
- Requires migration script for existing indexes
- Backward compatible via `TANTIVY_SOFT_DELETE_ENABLED` config flag

## Impact

- **Affected specs**: `add-tool`, `hybrid-memory`, `message-storage`
- **Affected code**:
  - `openmemories/infrastructure/tantivy_engine.py` - Soft-delete, compaction
  - `openmemories/application/memory/manager.py` - Parallel processing
  - `openmemories/infrastructure/qwen3_embedding.py` - Async batching
  - `openmemories/application/config/settings.py` - New config options
  - `openmemories/application/memory/fusion/ranx_fusion.py` - Score averaging
- **New components**: `TantivyCompactionService` for background compaction
- **Migration required**: Existing Tantivy indexes need schema migration
