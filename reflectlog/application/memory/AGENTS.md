# Agent Guidelines for reflectlog/application/memory/

This directory contains the hybrid memory management system that powers ReflectLogMCP. It combines semantic vector search (USearch) with full-text search (Tantivy) using Reciprocal Rank Fusion (RRF) for intelligent result ranking.

## Directory Structure

```
memory/
├── __init__.py              # Package exports (MemoryManager, protocols, utilities)
├── manager.py               # MemoryManager (main hybrid search engine)
├── engine_factory.py        # EngineFactory for search engine initialization
├── search_pipeline.py       # SearchPipeline with pluggable stages
├── add_pipeline.py          # AddPipeline with pluggable phases
├── search_strategies.py     # Original search strategies (legacy)
├── add_phases.py            # Original add phases (legacy)
├── match_utils.py           # Escape Tantivy queries, exact match detection
├── protocols.py             # Local protocols (ISemanticSearchEngine, etc.)
├── fusion/                  # Hybrid ranking (RRF fusion)
│   ├── __init__.py          # Fusion exports and factory function
│   ├── base.py              # FusionEngine protocol
│   └── ranx_fusion.py       # RanxFusionEngine implementation
└── reranking/               # Score normalization utilities
    ├── __init__.py          # Reranking exports
    └── normalization.py     # Min-max batch normalization for unified threshold semantics
```

## Core Responsibilities

### MemoryManager

The `MemoryManager` class orchestrates all memory operations:

- **add_messages_async()**: Stores messages with semantic embeddings using 3-phase pipeline
- **get_all()**: Retrieves all stored messages from USearchEngine
- **search()**: Performs hybrid search with RRF fusion and optional reranking
- **delete_by_message()**: Removes messages by exact match

### EngineFactory

The `EngineFactory` class handles search engine initialization:

```python
class EngineFactory:
    '''Factory for creating and configuring search engines.'''

    def __init__(self, config: Config, logger: StructuredLogger):
        self.config = config
        self.logger = logger

    def create_usearch_engine(self) -> USearchEngine:
        '''Create and configure USearch engine with embedder.'''
        ...

    def create_tantivy_engine(self) -> TantivyEngine:
        '''Create and configure Tantivy engine.'''
        ...
```

### Search Pipeline

The search operation follows a 4-step pipeline:

```
Query → [Step 1: Parallel Search] → [Step 2: RRF Fusion] → [Step 3: Fusion Filter] → [Step 4: Rerank] → Results
         USearch + Tantivy           RanxFusionEngine       threshold >= 0.8        LLM/CrossEncoder
              ↓                                                                              ↓
         timestamp_map ──────────────────────────────────────────────────────────→ Recency Decay
```

### Add Pipeline

The add operation uses a 3-phase architecture for optimal performance:

```
Messages → [Phase 1: Parallel Dedup] → [Phase 2: Parallel Replace] → [Phase 3: Sequential Store]
            Batch + Storage check       LLM candidate checks          SQLite writes
```

## Core Components

### USearchEngine (Semantic Search)

- **Backend**: USearch HNSW index + libSQL `MessageStore` for text
- **Configuration**: `USearchConfig` (cosine metric, 4096 dims)
- **Embeddings**: `LangchainQwenEmbeddings` (Qwen 4096 dims default)
- **Storage**: `indexes/{project_id}/usearch/` (vectors.usearch + messages.db)
- **Source of truth** for `get_all()`

### TantivyEngine (Full-text Search)

- **Schema**: `project_id`, `message`, `is_deleted`, `deleted_at`
- **Storage**: `indexes/{project_id}/tantivy/`
- **Optimized** for exact phrase matching and keyword search
- **O(1) soft-delete** via tombstone marking

### RanxFusionEngine (Hybrid Ranking)

- **Implementation**: `fusion/ranx_fusion.py`
- **Formula**: `RRF_score(doc) = sum over rankings of: 1 / (k + rank(doc))`
- **Configurable** `k` parameter via `FUSION_RRF_K` (default: 60)
- **Normalizes** scores to 0-1 range

### LLMReranker (AI Relevance Scoring)

- **Purpose**: Post-fusion relevance scoring using LLM
- **Provider**: Uses `IRerankerProvider` protocol (OpenAI or Anthropic)
- **Temporal-Aware**: When `ENABLE_RECENCY_BOOST=true`, includes memory age
- **Graceful fallback** to fusion score on LLM errors

### CrossEncoderReranker (Local Reranking)

- **Purpose**: Fast local reranking using FlagReranker model
- **Default model**: `BAAI/bge-reranker-v2-m3` (multilingual)
- **No API costs**, runs locally on CPU/GPU/MPS
- **Built-in** score normalization (sigmoid to 0-1 range)

## Key Patterns

### Protocol-Based Design

Define protocols for swappable implementations:

```python
class ISemanticSearchEngine(Protocol):
    def search(self, query: str, limit: int) -> list[tuple[str, float, str]]:
        '''Search for similar documents.'''
        ...

    def add(self, message: str) -> None:
        '''Add a document to the index.'''
        ...
```

### Lazy Initialization

For expensive resources, use thread-safe lazy initialization:

```python
@property
def llm_reranker(self) -> LLMReranker | None:
    '''Get LLM reranker with lazy initialization.'''
    if self._llm_reranker is not None or self.config.reranker_engine != "llm":
        return self._llm_reranker
    with self._reranker_lock:
        if self._llm_reranker is not None:
            return self._llm_reranker
        reranker_config = LLMRerankerConfig.from_app_config(self.config)
        self._llm_reranker = LLMReranker(config=reranker_config, logger=self.logger)
        return self._llm_reranker
```

### Pipeline Architecture

Extract complex logic into focused pipeline classes:

```python
class SearchPipeline:
    def __init__(self, memory_manager: MemoryManager):
        self.memory = memory_manager

    def execute(self, context: SearchContext) -> SearchResult:
        # Step 1: Parallel Search
        usearch_results, tantivy_results = self._parallel_search(context)

        # Step 2: RRF Fusion
        fused = self._fuse_results(usearch_results, tantivy_results)

        # Step 3: Filter and Rerank
        return self._filter_and_rerank(fused, context)
```

## Search Operations

### Hybrid Search

```python
def search(self, query: str, limit: int) -> list[str]:
    # Step 1: Parallel search
    usearch_results = self._search_usearch(query, limit=overfetch)
    tantivy_results = self._search_tantivy(query, limit=overfetch)

    # Build timestamp map for recency decay
    timestamp_map = self._build_timestamp_map(usearch_results)

    # Step 2: RRF fusion
    fused = self._fuse_hybrid_results(usearch_results, tantivy_results)

    # Step 3: Fusion threshold filter
    filtered = [r for r in fused if r.score >= self.config.fusion_threshold]

    # Step 4: Reranking
    if self.config.reranker_engine != "none":
        reranked = self._rerank(query, filtered, timestamp_map)
        return [r[0] for r in reranked]

    return [r[0] for r in filtered]
```

### Adaptive Overfetch

Dynamically adjust overfetch based on index size:

```python
def _calculate_adaptive_overfetch(self, index_size: int) -> int:
    if index_size <= 100:
        return int(limit * self.config.overfetch_max_multiplier)  # 3.0x
    elif index_size >= 10000:
        return int(limit * self.config.overfetch_min_multiplier)  # 1.5x
    else:
        # Logarithmic interpolation
        pass
```

## Add Operations

### Phased Parallel Add

```python
async def add_messages_async(self, messages: list[str]) -> AddResult:
    # Phase 1: Parallel duplicate detection
    phase1 = DuplicateDetectionPhase(self.memory)
    phase1_result = await phase1.execute(messages)

    # Phase 2: Parallel smart replacement
    phase2 = SmartReplacementPhase(self.memory)
    phase2_result = await phase2.execute(phase1_result.new_messages)

    # Phase 3: Sequential storage
    phase3 = StoragePhase(self.memory, self.tantivy_engine)
    phase3_result = await phase3.execute(phase2_result.to_store)

    return AddResult(
        stored_count=phase3_result.count,
        skipped_count=phase1_result.duplicates,
        replaced_count=phase2_result.replacements,
        replacements=phase2_result.replacement_info,
    )
```

### Smart Replacement Detection

```python
async def _detect_replacements(self, messages: list[str]) -> list[ReplacementInfo]:
    # Find similar memories
    candidates = self._find_similar_memories(messages)

    # Parallel LLM checks
    async with anyio.create_task_group() as tg:
        for candidate in candidates:
            tg.start_soon(self._check_replacement, candidate)

    return [r for r in results if r.confidence >= self.config.smart_replace_threshold]
```

## Delete Operations

### Delete by Message

```python
def delete_by_message(self, message: str) -> bool:
    '''Delete a memory by its exact message content.'''
    with self._write_lock:
        # Look up message ID
        memory_id = self._get_id_by_message(message)
        if memory_id is None:
            return False

        # Delete from both engines
        self._semantic_engine.delete_by_id(memory_id)
        self._tantivy_engine.delete(memory_id)

        return True
```

## Configuration

### Search Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_HYBRID_SEARCH` | true | Enable Tantivy full-text search |
| `ENABLE_RRF_FUSION` | true | Enable RRF fusion |
| `FUSION_RRF_K` | 60 | RRF constant |
| `FUSION_RANKING_THRESHOLD` | 0.8 | Min RRF score after fusion |
| `RERANKER_ENGINE` | llm | Reranking engine |
| `SEARCH_SCORE_THRESHOLD` | 0.5 | Min LLM relevance score |

### Recency Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_RECENCY_BOOST` | true | Include memory age in reranking |
| `RECENCY_DECAY_RATE` | 0.01 | Exponential decay per hour |
| `RERANKER_BATCH_NORMALIZE` | true | Enable batch normalization |

### Smart Replacement Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_SMART_REPLACE` | true | Enable smart replacement |
| `SMART_REPLACE_THRESHOLD` | 0.7 | Min LLM confidence |
| `SMART_REPLACE_MIN_SIMILARITY` | 0.9 | Min embedding similarity |
| `SMART_REPLACE_CANDIDATE_LIMIT` | 3 | Max candidates to check |

## Performance Considerations

### Add Operation Optimizations

- **Phased Parallel Add**: 5-8x speedup for bulk adds
- **Parallel Duplicate Detection**: Concurrent checks with semaphore limiting
- **Parallel Smart Replacement**: LLM calls run in parallel
- **Batch Commits**: All Phase 3 writes committed together

### Search Optimizations

- **Adaptive Overfetch**: 1.5-3x multiplier based on index size
- **Query Embedding Cache**: Reduces API calls for repeated queries
- **Parallel Hybrid Search**: USearch + Tantivy execute concurrently
- **RRF Score Averaging**: Improved fusion quality

### Delete Optimizations

- **Tantivy Soft-Delete**: O(1) tombstone marking vs O(n) rebuild
- **Direct Message Lookup**: O(log n) indexed lookup

## Thread Safety

### Lock Hierarchy

Follow the lock hierarchy to prevent deadlocks:

1. **`_write_lock`**: Acquired first for write operations
2. **`_lock`**: Acquired second for read operations
3. **`_reranker_lock`**: For reranker initialization
4. **`_smart_replacer_lock`**: For SmartReplacer initialization

```python
# Correct lock order
with self._write_lock:
    with self._lock:
        # Critical section
        pass
```

### USearch Thread Safety

USearch is not thread-safe. Serialize all write operations:

```python
def add(self, message: str) -> None:
    with self._write_lock:
        # All USearch operations must be under write lock
        self._engine.add(message)
```

## Error Handling

### Storage Failures

```python
try:
    self._semantic_engine.add(message)
    self._tantivy_engine.add(message)
except (USearchError, TantivyError) as e:
    raise MemoryStorageError(f"Failed to store message: {message[:100]}") from e
```

### Search Failures

```python
try:
    usearch_results = self._search_usearch(query)
    tantivy_results = self._search_tantivy(query)
except USearchError:
    # Fall back to Tantivy-only search
    return tantivy_results
```

## Testing Guidelines

### Unit Tests

- Mock both USearch and Tantivy dependencies
- Test RRF fusion algorithm correctness
- Verify deduplication logic
- Test error handling paths

### Integration Tests

- Real USearch and Tantivy indices
- Cross-engine consistency checks
- Performance benchmarks
- Persistence validation

### Key Test Cases

```python
def test_rrf_fusion_algorithm():
    '''RRF should rank by reciprocal rank.'''
    usearch_results = [("A", 0.9), ("B", 0.8)]
    tantivy_results = [("B", 0.7), ("C", 0.6)]
    fused = manager._fuse_hybrid_results(usearch_results, tantivy_results)

    # B appears in both, should rank highest
    assert fused[0][0] == "B"

def test_smart_replacement():
    '''New memory should replace similar old memory.'''
    manager.add_messages(["I prefer Python"])
    result = manager.add_messages(["I prefer JavaScript"])

    assert result.replaced_count == 1
    assert "Python" not in manager.get_all()
```

## Dependencies

### Internal Dependencies

- `infrastructure/usearch_engine.py`: USearch wrapper
- `infrastructure/tantivy_engine.py`: Tantivy wrapper
- `infrastructure/llm_reranker.py`: LLM-based reranking
- `infrastructure/cross_encoder_reranker.py`: Local reranking
- `infrastructure/smart_replacer.py`: Smart replacement detection
- `application/utils/`: Logging and utilities

### External Dependencies

- `usearch`: HNSW vector search library
- `tantivy`: Full-text search engine
- `ranx`: RRF fusion ranking library
- `langchain`: Embeddings interface
- `pydantic`: Configuration validation

## Important Notes

### Source of Truth

- USearchEngine is source of truth for `get_all()`
- Both engines must be kept in sync
- Use transactions where supported

### Recency Decay

The recency decay formula: `score * exp(-rate * hours_old)`

| Rate | Half-life | Use Case |
|------|-----------|----------|
| 0.001 | ~693 hours | Long-term preferences |
| 0.01 | ~69 hours | General memories |
| 0.05 | ~14 hours | Fast-changing context |
| 0.1 | ~7 hours | Session-specific data |

### Score Normalization

Different rerankers produce different score ranges:

| Reranker | Typical Range | Normalization |
|----------|---------------|---------------|
| LLMReranker | 0.7-0.9 | Batch min-max to 0-1 |
| CrossEncoderReranker | 0.001-0.17 | Sigmoid then min-max |

This enables unified threshold semantics across rerankers.
