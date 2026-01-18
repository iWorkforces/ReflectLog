# reflectlog/application/memory/

This directory contains the hybrid memory management system that powers ReflectLogMCP.

## Structure

```
memory/
├── __init__.py          # Package exports
├── manager.py           # MemoryManager (USearch + Tantivy hybrid engine)
├── protocols.py         # Search engine protocols (ISemanticSearchEngine)
├── search_strategies.py # SearchPipeline, SearchContext, SearchResult
├── add_phases.py        # AddPipeline, DuplicateDetectionPhase, SmartReplacementPhase, StoragePhase
├── fusion/              # Hybrid ranking
│   ├── __init__.py      # Fusion exports
│   ├── base.py          # FusionEngine protocol
│   └── ranx_fusion.py   # RanxFusionEngine (RRF implementation)
└── reranking/           # Score normalization utilities
    ├── __init__.py      # Reranking exports
    └── normalization.py # Min-max batch normalization for unified threshold semantics
```

## Purpose

The `memory/` module implements the core hybrid storage engine:

- **Dual Storage**: USearch/libSQL for semantic search + Tantivy for full-text search
- **RRF Fusion**: Reciprocal Rank Fusion via ranx library for intelligent result ranking
- **LLM Reranking**: Optional AI-powered relevance scoring for post-fusion refinement
- **Temporal-Aware Scoring**: Recency decay for handling contradictory memories (newer memories prioritized)
- **Deduplication**: Exact match checking before storage (via Tantivy or MessageStore)
- **Smart Replacement**: LLM-based detection when new memories update/replace existing ones
- **Parallel Processing**: 3-phase add operation with concurrent duplicate detection and smart replacement
- **Embedding Caching**: LRU cache for query embeddings to reduce API calls
- **Archiving**: Soft delete with recovery capability for replaced memories
- **Persistence**: Both engines persist data across server restarts
- **Consistent API**: Unified interface for all memory operations

### Search Pipeline (3-4 Steps)

When `ENABLE_RRF_FUSION=true` (default):
```
Query → [Step 1: Parallel Search] → [Step 2: RRF Fusion] → [Step 3: Fusion Filter] → [Step 4: Rerank] → Results
         USearch + Tantivy           RanxFusionEngine       threshold >= 0.8        LLM/CrossEncoder
              ↓                                                                          ↓
         timestamp_map ───────────────────────────────────────────────────────────→ Recency Decay
```

When `ENABLE_RRF_FUSION=false`:
```
Query → [Step 1: Parallel Search] → [Step 2: Concatenate] → [Step 3: Rerank] → Results
         USearch + Tantivy           Semantic priority       LLM/CrossEncoder
              ↓                                                    ↓
         timestamp_map ─────────────────────────────────────→ Recency Decay
```

**Timestamp Map Flow**: USearch returns `(message, score, created_at)` tuples. The `created_at` values are collected into `timestamp_map: Dict[str, str]` and passed to rerankers for recency decay calculation.

| Step | Component | Purpose | Configurable |
|------|-----------|---------|--------------|
| 1 | USearchEngine + TantivyEngine | Parallel semantic + full-text search | `ENABLE_HYBRID_SEARCH` |
| 2 | RanxFusionEngine or Concatenation | Combine results using RRF or concatenation | `ENABLE_RRF_FUSION`, `FUSION_RRF_K` |
| 3 | Fusion threshold (RRF only) | Filter low-confidence matches | `FUSION_RANKING_THRESHOLD` |
| 3/4 | Reranker | AI/CrossEncoder relevance scoring | `RERANKER_ENGINE`, `SEARCH_SCORE_THRESHOLD` |

**Note**: When RRF fusion is disabled, Step 3 (fusion threshold) is skipped because scores from different engines aren't comparable. Reranking becomes Step 3 instead of Step 4.

## MemoryManager Architecture

### Core Components

1. **USearchEngine** (Semantic Search)
   - Infrastructure: `reflectlog.infrastructure.USearchEngine`
   - Backend: USearch HNSW index + libSQL `MessageStore` for text
   - Configuration: `USearchConfig` (cosine metric, 4096 dims)
   - Embeddings: `LangchainQwenEmbeddings` (Qwen 4096 dims default)
   - Storage: `indexes/{project_id}/usearch/` (vectors.usearch + messages.db)
   - **Source of truth for `get_all()`**

2. **TantivyEngine** (Full-text Search)
   - Infrastructure: `reflectlog.infrastructure.TantivyEngine`
   - Schema: `project_id`, `message`, `is_deleted`, `deleted_at` (V2 schema)
   - Storage: `indexes/{project_id}/tantivy/`
   - Optimized for exact phrase matching and keyword search
   - **O(1) soft-delete** via tombstone marking (when enabled)

3. **CachedEmbeddings** (Query Embedding Cache)
   - Infrastructure: `reflectlog.infrastructure.CachedEmbeddings`
   - Wraps base embedder with LRU cache for `embed_query()` calls
   - MD5 hash of query text as cache key
   - Thread-safe with hit/miss statistics
   - Configurable via `EMBEDDING_CACHE_ENABLED` and `EMBEDDING_CACHE_SIZE`

4. **RanxFusionEngine** (Hybrid Ranking)
   - Implementation: `reflectlog/application/memory/fusion/ranx_fusion.py`
   - Formula: `RRF_score(doc) = sum over rankings of: 1 / (k + rank(doc))`
   - Configurable `k` parameter via `FUSION_RRF_K` (default: 60)
   - Normalizes scores to 0-1 range

5. **LLMReranker** (AI Relevance Scoring)
   - Infrastructure: `reflectlog.infrastructure.LLMReranker`
   - Purpose: Post-fusion relevance scoring using LLM
   - Uses `SCORING_PROMPT` or `SCORING_PROMPT_WITH_AGE` from `config/prompts.py`
   - **Temporal-Aware**: When `ENABLE_RECENCY_BOOST=true`, includes memory age in prompt
   - Parallel scoring with concurrency control (default: 10)
   - Graceful fallback to fusion score on LLM errors
   - **Provider Abstraction**: Uses `IRerankerProvider` protocol (OpenAI or Anthropic)

6. **CrossEncoderReranker** (Local Reranking)
   - Infrastructure: `reflectlog.infrastructure.CrossEncoderReranker`
   - Purpose: Fast local reranking using FlagReranker model
   - Default model: `BAAI/bge-reranker-v2-m3` (multilingual)
   - **Temporal-Aware**: Supports recency decay via `timestamp_map` parameter
   - No API costs, runs locally on CPU/GPU/MPS
   - Built-in score normalization (sigmoid to 0-1 range)

7. **Score Normalization** (`reranking/normalization.py`)
   - Purpose: Batch min-max normalization for unified threshold semantics
   - Transforms diverse reranker score ranges to [0, 1]
   - LLMReranker scores (0.7-0.9) and CrossEncoder scores (0.001-0.17) become comparable
   - Safety net: `apply_threshold_with_safety_net()` guarantees min results
   - **Recency Decay**: `apply_recency_decay()` multiplies scores by `exp(-rate * hours_old)`

### Infrastructure Layer Abstraction

**MemoryManager** uses dedicated engine classes from the infrastructure layer with lazy initialization:

```python
# In MemoryManager.__init__():
from reflectlog.infrastructure import (
    USearchConfig, USearchEngine, TantivyConfig, TantivyEngine,
)

# Semantic engine (USearch/libSQL)
usearch_config = USearchConfig.from_app_config(config)
embedder = LangchainQwenEmbeddings(embedder_config)
self._semantic_engine = USearchEngine(config=usearch_config, embedder=embedder, logger=self.logger)

# Full-text engine (Tantivy)
if config.enable_hybrid_search:
    tantivy_config = TantivyConfig(project_id=config.project_id, index_path=tantivy_index_path)
    self._tantivy_engine = TantivyEngine(tantivy_config, logger=self.logger)

# Rerankers and SmartReplacer - lazily initialized via properties
# These are created on first use to avoid startup overhead
self._llm_reranker: LLMReranker | None = None
self._cross_encoder_reranker: CrossEncoderReranker | None = None
self._smart_replacer: SmartReplacer | None = None

# Properties provide thread-safe lazy loading
@property
def llm_reranker(self) -> LLMReranker | None:
    """Get LLM reranker (lazy initialization with thread-safety)."""
    if self._llm_reranker is not None or self.config.reranker_engine != "llm":
        return self._llm_reranker
    with self._reranker_lock:
        if self._llm_reranker is not None:
            return self._llm_reranker
        reranker_config = LLMRerankerConfig.from_app_config(self.config)
        self._llm_reranker = LLMReranker(config=reranker_config, logger=self.logger)
        return self._llm_reranker

def get_reranker(self) -> LLMReranker | CrossEncoderReranker | None:
    """Get the appropriate reranker based on configuration."""
    if self.config.reranker_engine == "llm":
        return self.llm_reranker
    elif self.config.reranker_engine == "cross_encoder":
        return self.cross_encoder_reranker
    return None
```

This architecture provides:
- **Separation of concerns**: Application logic separated from infrastructure integration
- **Testability**: Engines can be easily mocked in unit tests
- **Consistency**: Both engines follow the same Pydantic BaseModel pattern
- **Error handling**: Centralized exception wrapping at the infrastructure layer

### Pipeline Architecture (Phase 9 Refactoring)

**SearchPipeline** (`search_strategies.py`):
- Extracts the 4-step search logic from manager.py into a focused, testable module
- Implements: Parallel Search → RRF Fusion/Concatenation → Fusion Filter → Reranking
- Classes: `SearchContext` (input parameters), `SearchResult` (output with metadata)
- Functions: `calculate_adaptive_overfetch()` (see `match_utils.escape_tantivy_query()` for query escaping)

**AddPipeline** (`add_phases.py`):
- Extracts the 3-phase add logic from manager.py into modular phase classes
- Classes:
  - `DuplicateDetectionPhase`: Parallel duplicate checking (batch + storage)
  - `SmartReplacementPhase`: Parallel LLM replacement detection
  - `StoragePhase`: Sequential database writes
  - `AddPipeline`: Orchestrates all three phases
- Dataclasses: `Phase1Result`, `Phase2Result`, `Phase3Result`, `ReplacementInfo`, `AddResult`

**Benefits**:
- **Reduced manager.py**: From ~1,010 lines to ~776 lines (~230 lines extracted)
- **Better testability**: Each pipeline/phase can be tested independently
- **Clearer separation of concerns**: Manager orchestrates, pipelines execute
- **Reusability**: Pipelines can be reused or extended without modifying manager.py

### Lazy Initialization Architecture

**Performance Optimization**: Rerankers and SmartReplacer use lazy loading via thread-safe properties to reduce server startup time.

**Components with Lazy Loading**:

1. **LLMReranker** (`llm_reranker` property)
   - Initialized on first search when `RERANKER_ENGINE=llm`
   - Thread-safe double-checked locking with `_reranker_lock`
   - Returns `None` if not configured

2. **CrossEncoderReranker** (`cross_encoder_reranker` property)
   - Initialized on first search when `RERANKER_ENGINE=cross_encoder`
   - Thread-safe double-checked locking with `_reranker_lock`
   - Returns `None` if not configured

3. **SmartReplacer** (`smart_replacer` property)
   - Initialized on first add operation when `ENABLE_SMART_REPLACE=true`
   - Thread-safe double-checked locking with `_smart_replacer_lock`
   - Returns `None` if smart replacement disabled

**Unified Access via `get_reranker()`**:
```python
# In MemoryManager:
def get_reranker(self) -> LLMReranker | CrossEncoderReranker | None:
    """Get the appropriate reranker based on configuration."""
    if self.config.reranker_engine == "llm":
        return self.llm_reranker  # Triggers lazy init if needed
    elif self.config.reranker_engine == "cross_encoder":
        return self.cross_encoder_reranker  # Triggers lazy init if needed
    return None
```

**Pipeline Integration**:
- `SearchPipeline` receives `memory_manager` parameter
- Calls `memory_manager.get_reranker()` during search execution
- `SmartReplacementPhase` receives `memory_manager` parameter
- Calls `memory_manager.smart_replacer` during add execution

**Benefits**:
- **Startup Time**: Reduces startup by 500-2000ms (LLM) or avoids unnecessary model loading
- **Memory**: Only loads components when actually needed
- **Thread-Safe**: Double-checked locking prevents race conditions

### Phased Parallel Add Processing

The `add_messages_async()` method uses a 3-phase architecture for optimal performance:

```
Messages → [Phase 1: Parallel Dedup] → [Phase 2: Parallel Replace] → [Phase 3: Sequential Store]
            Batch + Storage check       LLM candidate checks          SQLite writes
```

| Phase | Operation | Parallelism | Purpose |
|-------|-----------|-------------|---------|
| 1 | Duplicate Detection | Full parallel | Check batch duplicates + storage duplicates concurrently |
| 2 | Smart Replacement | Parallel (semaphore-limited) | LLM checks for replacement candidates |
| 3 | Database Writes | Sequential | SQLite constraint (single-threaded writes) |

**Performance**: Provides 5-8x speedup over sequential processing for multiple messages by maximizing I/O parallelism in phases 1-2 while maintaining data consistency in phase 3.

**Concurrency Control**: Uses `ADD_MAX_CONCURRENCY` (default: 4) semaphore for duplicate detection parallelism.

### Smart Replacement (SmartReplacer)

**Infrastructure**: `reflectlog.infrastructure.SmartReplacer`

The smart replacement system detects when a new memory semantically updates or contradicts an existing one:

```
New Memory → [Similarity Search] → [Parallel LLM Check] → [Archive + Replace]
                 USearch              SmartReplacer           MessageStore
```

**Flow**:
1. **Similarity Pre-filter**: Find candidates with embedding similarity >= `SMART_REPLACE_MIN_SIMILARITY`
2. **Parallel LLM Detection**: Check all candidates concurrently via `anyio.create_task_group()`
3. **Confidence Check**: Only replace if LLM confidence >= `SMART_REPLACE_THRESHOLD`
4. **Archive**: Soft-delete old memory to `archived_messages` table
5. **Store**: Add new memory normally

**Parallel Optimization**: LLM replacement checks run in parallel with semaphore-based rate limiting, reducing latency from 3-6s to 1-2s for multiple candidates.

**Result Dataclasses**:

```python
@dataclass
class ReplacementInfo:
    """Details about a single replacement operation."""
    old_memory: str       # The memory that was replaced
    new_memory: str       # The new memory that replaced it
    confidence: float     # LLM confidence score (0.0-1.0)
    reason: str           # LLM explanation for replacement

@dataclass
class AddResult:
    """Result of add_messages_async() operation."""
    stored_count: int                    # Number of new messages stored
    skipped_count: int                   # Number of duplicates skipped
    replaced_count: int                  # Number of memories replaced
    replacements: List[ReplacementInfo]  # Details of each replacement
```

### Key Methods

#### `add_messages_async(messages: List[str], dry_run: bool = False) -> AddResult`
- Stores messages in both USearchEngine and TantivyEngine
- Deduplication via TantivyEngine exact match check (when enabled)
- Smart replacement detection (when `ENABLE_SMART_REPLACE=true`)
- **dry_run**: If True, performs replacement detection but doesn't modify storage
- Returns `AddResult` with counts and replacement details

#### `add_messages(messages: List[str]) -> int` (sync wrapper)
- Synchronous wrapper around `add_messages_async()`
- Returns count of actually stored messages (for backward compatibility)

#### `get_all() -> List[str]`
- Returns all messages from USearchEngine (source of truth)
- Uses `_semantic_engine.get_all()` which returns clean list of strings
- Defensive: returns fresh list each time

#### `search(query: str, limit: int) -> List[str]`

- **Step 1**: Executes both semantic (USearchEngine) and full-text (TantivyEngine) search in parallel
  - USearch returns `(message, score, created_at)` tuples
  - Builds `timestamp_map: Dict[str, str]` from `created_at` values
- **Step 2**: Combines results using RRF fusion or concatenation (based on `ENABLE_RRF_FUSION`)
- **Step 3**: When RRF enabled: applies fusion threshold filtering (`FUSION_RANKING_THRESHOLD`, default: 0.8)
- **Step 3/4**: Reranks with LLMReranker/CrossEncoder (when `RERANKER_ENGINE` is `llm` or `cross_encoder`)
  - Passes `timestamp_map` to reranker for recency decay
  - Reranker applies `normalize_reranker_scores()` then `apply_recency_decay()`
- **Processing**: Returns list of message strings

**Note**: When `ENABLE_RRF_FUSION=false`, Step 2 concatenates results with semantic priority, Step 3 (fusion threshold) is skipped, and reranking becomes Step 3.

**Recency Decay Flow** (in rerankers):
```
Reranker Score → [Batch Normalize] → [Apply Decay] → [Re-sort] → [Threshold Filter] → Results
                  0-1 range          score * exp(-rate * hours)   by decayed score
```

#### `search_for_removal(message: str) -> List[dict]`
- Specialized search for `remove` tool
- Returns candidates with `id`, `memory`, and `score` fields
- Uses USearchEngine (source of truth) with Python-level exact string matching
- Guarantees finding messages regardless of Tantivy index state

#### `delete_by_id(memory_id: str) -> None`
- Deletes a specific memory by its ID
- Removes from both USearchEngine (libSQL) and TantivyEngine

#### `delete_by_message(message: str) -> bool`
- Deletes a memory by its exact message content (thread-safe)
- Looks up message ID via `get_id_by_message()`, then deletes from both engines
- Returns `True` if found and deleted, `False` if not found
- **Note:** TantivyEngine.delete() commits internally, so no explicit commit needed
- Raises `InconsistentStateError` if Tantivy deletion fails after USearch deletion

#### `_fuse_hybrid_results(usearch_results, tantivy_results) -> List[Tuple[str, float]]`
- Industry-standard RRF algorithm via ranx library
- Handles duplicate removal and score normalization
- Returns messages sorted by RRF score descending

### Adaptive Overfetch

The `_calculate_adaptive_overfetch()` method dynamically adjusts how many documents to fetch based on index size:

```
Index Size → [Logarithmic Interpolation] → Overfetch Multiplier
  ≤ 100           max (3.0x default)         Better diversity
  ≥ 10,000        min (1.5x default)         Sufficient for fusion
```

**Configuration**:
- `OVERFETCH_ADAPTIVE`: Enable adaptive calculation (default: true)
- `OVERFETCH_MIN_MULTIPLIER`: Multiplier for large indexes (default: 1.5)
- `OVERFETCH_MAX_MULTIPLIER`: Multiplier for small indexes (default: 3.0)

**Impact**: ~15-20% latency reduction for large indexes by fetching fewer documents while maintaining fusion quality.

### Eager Initialization

The `_eager_initialize_engines()` method provides granular control over which components are pre-warmed during `MemoryManager.__init__()`:

**Default Behavior** (Rerankers/SmartReplacer are lazy by default):
- **USearch**: Loads HNSW index and opens libSQL connection (when `EAGER_INITIALIZE_SEARCH_ENGINES=true`)
- **Tantivy**: Opens index, creates writer and searcher (when `EAGER_INITIALIZE_SEARCH_ENGINES=true`)
- **Rerankers**: Lazy loaded on first search (can be overridden with `EAGER_INITIALIZE_RERANKER=true`)
- **SmartReplacer**: Lazy loaded on first add (can be overridden with `EAGER_INITIALIZE_SMART_REPLACER=true`)

**Configuration**:
- `EAGER_INITIALIZATION`: General flag (default: true) - applies to search engines only
- `EAGER_INITIALIZE_SEARCH_ENGINES`: Pre-warm USearch/Tantivy (default: falls back to EAGER_INITIALIZATION)
- `EAGER_INITIALIZE_RERANKER`: Pre-load reranker on startup (default: false - lazy loading)
- `EAGER_INITIALIZE_SMART_REPLACER`: Pre-load SmartReplacer on startup (default: false - lazy loading)

**Impact**:
- **Search engines**: Reduces first-request latency by moving initialization to server startup
- **Rerankers/SmartReplacer**: Lazy loading reduces server startup time by 500-2000ms

### Configuration

Key environment variables (via `Config`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_HYBRID_SEARCH` | true | Enable Tantivy full-text search |
| `ENABLE_RRF_FUSION` | true | Enable RRF fusion (false = concatenate results) |
| `FUSION_RRF_K` | 60 | RRF constant (lower = more weight to top ranks) |
| `FUSION_RANKING_THRESHOLD` | 0.8 | Min RRF score to keep after fusion (Step 3, RRF only) |
| `RERANKER_ENGINE` | llm | Reranking engine: `llm`, `cross_encoder`, or `none` (Step 3/4) |
| `SEARCH_SCORE_THRESHOLD` | 0.5 | Min LLM relevance score to keep (Step 3/4) |
| `RERANK_MAX_CONCURRENCY` | 10 | Max parallel LLM calls for reranking |
| `LLM_MODEL` | `x-ai/grok-4.1-fast` | LLM model for reranking |
| `LLM_PROVIDER` | `anthropic` | LLM provider: `openai` or `anthropic` |
| `TANTIVY_INDEX_PATH_TEMPLATE` | `indexes/{project_id}/tantivy` | Tantivy index path |
| `USEARCH_INDEX_PATH_TEMPLATE` | `indexes/{project_id}/usearch` | USearch index path |
| `DEDUPLICATE_MESSAGES` | true | Skip exact duplicates on add |

**Recency Decay Configuration**:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_RECENCY_BOOST` | true | Include memory age in reranking context |
| `RECENCY_DECAY_RATE` | 0.01 | Exponential decay per hour: `exp(-rate * hours_old)` |
| `RERANKER_BATCH_NORMALIZE` | true | Enable batch min-max normalization before decay |
| `RERANKER_MIN_RESULTS` | 0 | Safety net: min results to return (0 = disabled) |

**Decay Rate Examples**:

| Rate | Half-life (hours) | Use Case |
|------|-------------------|----------|
| 0.001 | ~693 | Long-term preferences |
| 0.01 (default) | ~69 | General memories |
| 0.05 | ~14 | Fast-changing context |
| 0.1 | ~7 | Session-specific data |

**Embedding Cache Configuration**:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_CACHE_ENABLED` | true | Enable LRU cache for query embeddings |
| `EMBEDDING_CACHE_SIZE` | 100 | Max cached embeddings |
| `EMBEDDING_BATCH_SIZE` | 512 | Texts per API request for async batching |
| `EMBEDDING_MAX_CONCURRENT_BATCHES` | 4 | Max parallel batch requests |

**Overfetch Configuration**:

| Variable | Default | Description |
|----------|---------|-------------|
| `OVERFETCH_ADAPTIVE` | true | Enable adaptive overfetch based on index size |
| `OVERFETCH_MIN_MULTIPLIER` | 1.5 | Multiplier for large indexes (≥10k docs) |
| `OVERFETCH_MAX_MULTIPLIER` | 3.0 | Multiplier for small indexes (≤100 docs) |

**Concurrency Configuration**:

| Variable | Default | Description |
|----------|---------|-------------|
| `ADD_MAX_CONCURRENCY` | 4 | Max concurrent message additions in Phase 1 |

**Eager Initialization Configuration**:

| Variable | Default | Description |
|----------|---------|-------------|
| `EAGER_INITIALIZATION` | true | Pre-warm search engines during server startup |
| `EAGER_INITIALIZE_SEARCH_ENGINES` | null | Pre-warm USearch/Tantivy (null = use EAGER_INITIALIZATION) |
| `EAGER_INITIALIZE_RERANKER` | null | Pre-load reranker on startup (null = false, lazy loading) |
| `EAGER_INITIALIZE_SMART_REPLACER` | null | Pre-load SmartReplacer on startup (null = false, lazy loading) |

**Tantivy Soft-Delete Configuration**:

| Variable | Default | Description |
|----------|---------|-------------|
| `TANTIVY_SOFT_DELETE_ENABLED` | true | O(1) tombstone marking vs O(n) rebuild |
| `TANTIVY_COMPACTION_THRESHOLD_RATIO` | 0.2 | Compact when tombstones > 20% of docs |
| `TANTIVY_COMPACTION_MAX_TOMBSTONES` | 10000 | Force compaction above this count |
| `TANTIVY_TOMBSTONE_TTL_DAYS` | 7 | Days before tombstones eligible for removal |

**Smart Replacement Configuration**:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_SMART_REPLACE` | true | Enable smart memory replacement detection |
| `SMART_REPLACE_THRESHOLD` | 0.7 | Min LLM confidence to trigger replacement (0.0-1.0) |
| `SMART_REPLACE_MIN_SIMILARITY` | 0.9 | Min embedding similarity to trigger LLM check |
| `SMART_REPLACE_CANDIDATE_LIMIT` | 3 | Max candidates to check for replacement |
| `SMART_REPLACE_ARCHIVE_TTL_DAYS` | 30 | Days to keep archived memories (0 = permanent) |
| `SMART_REPLACE_MAX_RETRIES` | 3 | Max LLM call retries with exponential backoff |
| `SMART_REPLACE_RETRY_DELAY` | 1.0 | Base delay in seconds for exponential backoff |

### Error Handling

- **Storage failures**: Wrapped in `RuntimeError` with context
- **Search failures**: Graceful fallback between engines
- **TantivyEngine issues**: Non-blocking, logs warnings
- **USearchEngine issues**: Raises `RuntimeError`, blocks operation (semantic backend is critical)

### Logging Strategy

Structured logging with context:

```python
self.logger.info(
    "Hybrid search completed",
    extra={
        "project_id": self.config.project_id,
        "usearch_count": len(usearch_results),
        "tantivy_count": len(tantivy_results),
        "fused_count": len(fused_results),
        "query": query[:100],
    }
)
```

## Usage Examples

### Basic Storage and Retrieval

```python
from reflectlog.application.memory import MemoryManager
from reflectlog.application.config import config
from reflectlog.application.utils import create_logger

logger = create_logger(__name__, config.project_id, config.log_level)
manager = MemoryManager(config, logger)

# Add messages (deduplicated)
stored = manager.add_messages([
    "Python is a programming language",
    "JavaScript is used for web development"
])

# Retrieve all
all_messages = manager.get_all()

# Search with hybrid ranking
results = manager.search("Python programming", limit=5)
```

### Hybrid Search Details

```python
# Semantic search finds conceptually similar
usearch_results = manager._search_usearch("AI", limit=20)

# Full-text search finds exact phrases
tantivy_results = manager._search_tantivy("machine learning", limit=20)

# RRF fusion combines both via RanxFusionEngine
fused = manager._fuse_hybrid_results(usearch_results, tantivy_results)
```

### Deduplication Logic

```python
# Tantivy exact match check before storage
if manager._has_exact_match("existing message"):
    print("Message already exists, skipping")
else:
    manager.add_messages(["existing message"])
```

## Testing Strategy

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

### Test Scenarios

```python
def test_rrf_fusion_algorithm():
    """RRF should rank by reciprocal rank."""
    usearch_results = [("A", 0.9), ("B", 0.8)]
    tantivy_results = [("B", 0.7), ("C", 0.6)]
    fused = manager._fuse_hybrid_results(usearch_results, tantivy_results)

    # B appears in both, should rank highest
    assert fused[0][0] == "B"

def test_deduplication_logic():
    """Exact matches should be skipped."""
    manager.add_messages(["test message"])
    stored = manager.add_messages(["test message"])  # Duplicate
    assert stored == 0  # No new messages stored
```

## Performance Considerations

### Add Operation Optimizations

- **Phased Parallel Add**: 3-phase architecture provides 5-8x speedup for bulk adds
- **Parallel Duplicate Detection**: Concurrent checks with semaphore limiting
- **Parallel Smart Replacement**: LLM calls run in parallel (3-6s → 1-2s)
- **Commit Strategy**: Batch commits after all Phase 3 writes

### Search Optimizations

- **Adaptive Overfetch**: 1.5-3x multiplier based on index size (~15-20% latency reduction for large indexes)
- **Query Embedding Cache**: LRU cache reduces API calls for repeated queries
- **Parallel Hybrid Search**: USearch + Tantivy execute concurrently
- **RRF Score Averaging**: Improved fusion quality for duplicate documents

### Delete Optimizations

- **Tantivy Soft-Delete**: O(1) tombstone marking vs O(n) index rebuild
- **Direct Message Lookup**: O(log n) indexed lookup instead of O(n) scan

### General Optimizations

- **Eager Initialization**: Pre-warm engines at startup for faster first request
- **Concurrency Control**: `RERANK_MAX_CONCURRENCY=10` limits parallel LLM calls
- **Embedding Batching**: 512 texts per API request for async document embedding
- **Disable Reranking**: Set `RERANKER_ENGINE=none` for faster but less accurate results

### Memory & Latency

- **Memory Usage**: Both engines load indices into memory
- **Search Latency**: Hybrid search ~2x single-engine latency
- **Reranking Latency**: LLM reranking adds ~100-500ms per candidate (parallelized)

## Embedding Configuration

### OpenRouter (default)

```python
embedder_config = {
    "provider": "openai",
    "config": {
        "model": "openai/text-embedding-3-large",
        "api_key": config.openrouter_api_key,
        "base_url": "https://openrouter.ai/api/v1",
    }
}
```

### Langchain/Qwen

```python
embedder_config = {
    "provider": "langchain",
    "config": {
        "class": "reflectlog.infrastructure.qwen3_embedding.LangchainQwenEmbeddings",
        "config": {
            "model": "qwen3-embedding",
            "embedding_dims": 4096,
        }
    }
}
```

## Dependencies

- `usearch`: HNSW vector search library
- `tantivy`: Full-text search engine
- `ranx`: RRF fusion ranking library
- `langchain`: Embeddings interface
- `pydantic`: Configuration validation

## Migration Notes

The current architecture uses USearchEngine as the primary semantic backend:
- **Semantic**: USearch HNSW + libSQL MessageStore
- **Full-text**: Tantivy with English stemming
- **Fusion**: RRF via ranx library
- **Reranking**: LLM-based relevance scoring via LLMReranker
- **Benefit**: Better precision via full-text matching + semantic understanding + AI refinement


<claude-mem-context>
# Recent Activity

<!-- This section is auto-generated by claude-mem. Edit content outside the tags. -->

### Dec 26, 2025

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #577 | 11:57 AM | ✅ | Documentation Directory Structure Update | ~202 |
| #498 | 11:39 AM | 🟣 | Memory Module Enhancement | ~179 |
| #483 | 11:34 AM | 🔵 | Memory Manager Methods Identified | ~167 |
</claude-mem-context>
