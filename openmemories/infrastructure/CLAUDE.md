# ccmemories/infrastructure/

This directory contains infrastructure components and external integrations for CCMemoriesMCP.

## Structure

```
infrastructure/
├── __init__.py              # Package exports
├── cached_embeddings.py     # CachedEmbeddings LRU cache wrapper
├── cross_encoder_reranker.py # CrossEncoderReranker for local reranking
├── llm_reranker.py          # LLMReranker for AI relevance scoring
├── message_store.py         # MessageStore libSQL storage (for USearch)
├── qwen3_embedding.py       # LangchainQwenEmbeddings implementation
├── smart_replacer.py        # SmartReplacer for LLM-based memory replacement
├── tantivy_engine.py        # TantivyEngine full-text search wrapper (with soft-delete)
└── usearch_engine.py        # USearchEngine vector search wrapper (USearch/libSQL)
```

## Purpose

The `infrastructure/` module provides:
- External service integrations (embedding providers, search engines)
- Semantic vector search via **USearch/libSQL** (`USearchEngine`): HNSW-based search with libSQL message storage
- Full-text search (Tantivy)
- Async operation support via `anyio`
- HTTP/2 performance optimizations
- Compatibility layers for different providers
- Retry logic and error handling
- Persistent index storage (vector and full-text)

## Exports

```python
from ccmemories.infrastructure import (
    CachedEmbeddings,         # LRU caching wrapper for query embeddings
    CrossEncoderConfig,       # CrossEncoder reranker configuration
    CrossEncoderReranker,     # Local cross-encoder based reranking
    LangchainQwenEmbeddings,  # OpenRouter embedding provider
    LLMReranker,              # LLM-based relevance scoring for search results
    LLMRerankerConfig,        # LLMReranker configuration dataclass
    MessageRecord,            # Message record dataclass for MessageStore
    MessageStore,             # libSQL message storage (for USearch)
    RelevanceScore,           # Relevance score dataclass for LLMReranker
    ReplacementDecision,      # Replacement decision from SmartReplacer
    SmartReplacer,            # LLM-based memory replacement detection
    SmartReplacerConfig,      # SmartReplacer configuration dataclass
    TantivyConfig,            # Tantivy configuration dataclass (includes soft-delete options)
    TantivyEngine,            # Full-text search engine with soft-delete support
    USearchConfig,            # USearch engine configuration dataclass
    USearchEngine,            # USearch vector search engine (primary semantic backend)
)
```

---

## USearchEngine (`usearch_engine.py`)

### Overview

Primary semantic search engine using [USearch](https://github.com/unum-cloud/usearch) for HNSW vector similarity search and libSQL for message text storage:
- Clean infrastructure-layer abstraction implementing `ISemanticSearchEngine`
- Persistent USearch index storage on disk
- libSQL-based message text storage with WAL mode and MVCC
- Thread-safe lazy initialization
- Vector similarity search with cosine distance
- Compatible with any `langchain_core.embeddings.Embeddings` provider
- Source of truth for `get_all()` operations

### Configuration

```python
@dataclass(frozen=True)
class USearchConfig:
    project_id: str           # Unique project identifier for filtering
    index_path: str           # Path to the USearch index file (.usearch)
    db_path: str              # Path to the libSQL message database
    embedding_dims: int       # Embedding vector dimensions
    metric: str = "cos"       # Distance metric (cos, l2, ip)
    connectivity: int = 16    # HNSW M parameter
    expansion_add: int = 128  # HNSW efConstruction parameter
    expansion_search: int = 64 # HNSW ef search parameter
```

**Factory Method:**
```python
config = USearchConfig.from_app_config(app_config)
```

This factory method creates a `USearchConfig` from the application's `Config` object, extracting all necessary fields and constructing paths for both USearch index and libSQL database.

### Class Definition

```python
class USearchEngine(BaseModel, ISemanticSearchEngine):
    """USearch-based semantic search engine."""

    config: USearchConfig
    embedder: Embeddings      # Any Langchain-compatible embedder
    logger: Any = None        # StructuredLogger

    _index: Optional[Index] = PrivateAttr(default=None)
    _message_store: Optional[MessageStore] = PrivateAttr(default=None)
```

### Key Methods

#### `add(project_id: str, message: str, infer: bool) -> None`
Add a message to the semantic index:
```python
engine.add(
    project_id="my-project",
    message="Python is great for data science",
    infer=False  # infer not supported, logs warning if True
)
```

**Note:** The `infer` parameter is ignored (logs warning) as USearch doesn't support LLM inference.

#### `search(query: str, project_id: str, limit: int, rerank: bool, score_threshold: float) -> List[Tuple[str, float]]`
Execute semantic vector search:
```python
results = engine.search(
    query="data science with Python",
    project_id="my-project",
    limit=5,
    rerank=False,        # Passed through (not used by USearch itself)
    score_threshold=0.5  # Passed through (not used by USearch itself)
)
# Returns: [("Python is great for data science", 0.92), ...]
```

**Returns**: List of (message, score) tuples sorted by relevance (highest first).

**Note:** `rerank` and `score_threshold` are passed through for API compatibility but not applied within USearchEngine. The MemoryManager handles reranking.

#### `get_all(project_id: str) -> List[str]`
Retrieve all messages for a user/project:
```python
messages = engine.get_all(project_id="my-project")
# Returns: ["Message 1", "Message 2", "Message 3"]
```

**Returns**: List of message strings (no scores).

#### `delete(memory_id: str) -> None`
Delete a message by its libSQL row ID:
```python
engine.delete(memory_id="123")
```

**Raises:** `RuntimeError` if memory_id is not a valid integer string.

#### `commit() -> None`
Save the USearch index to disk:
```python
engine.commit()  # Persists USearch index to disk
```

#### `ensure_initialized() -> None`
Force initialization of index and message store (thread-safe):
```python
engine.ensure_initialized()  # Useful before parallel operations
```

#### `close() -> None`
Close resources and cleanup:
```python
engine.close()  # Closes libSQL connection
```

### Lazy Initialization

Both USearch index and libSQL MessageStore are lazily initialized on first access:

```python
@property
def index(self) -> Index:
    """Get USearch index (thread-safe lazy initialization)."""
    if self._index is not None:
        return self._index
    with self._init_lock:
        if self._index is None:
            if os.path.exists(self.config.index_path):
                self._index = Index.restore(self.config.index_path)
            else:
                self._index = Index(
                    ndim=self.config.embedding_dims,
                    metric=self.config.metric,
                    dtype="f32",
                    connectivity=self.config.connectivity,
                )
    return self._index
```

### Message Storage

USearch stores only vector embeddings (keys → vectors). Message text is stored separately in libSQL via `MessageStore`:

```python
# Insert: get libSQL ID, use as USearch key
msg_id = self.message_store.insert(project_id, message)
vector = self.embedder.embed_query(message)
self.index.add(msg_id, vector)

# Search: find USearch keys, look up messages in libSQL
matches = self.index.search(query_vector, limit)
for match in matches:
    record = self.message_store.get(match.key)
    results.append((record.message, 1.0 - match.distance))
```

### Usage Example

```python
from ccmemories.infrastructure import USearchConfig, USearchEngine
from ccmemories.infrastructure.qwen3_embedding import LangchainQwenEmbeddings

# Create configuration
config = USearchConfig(
    project_id="my-project",
    index_path="indexes/my-project/usearch/vectors.usearch",
    db_path="indexes/my-project/usearch/messages.db",
    embedding_dims=128,
)

# Initialize embedder
embedder = LangchainQwenEmbeddings({
    "model": "qwen3-embedding",
    "embedding_dims": 128,
})

# Initialize engine
engine = USearchEngine(config=config, embedder=embedder, logger=logger)

try:
    # Add messages
    engine.add("my-project", "Python is great for data science", infer=False)
    engine.add("my-project", "JavaScript powers the web", infer=False)
    engine.commit()

    # Search
    results = engine.search(
        query="data science",
        project_id="my-project",
        limit=5,
        rerank=False,
        score_threshold=0.0
    )

    for message, score in results:
        print(f"{score:.2f}: {message}")

    # Get all messages
    all_messages = engine.get_all(project_id="my-project")

    # Delete by ID
    engine.delete(memory_id="1")

finally:
    engine.close()
```

### Integration with MemoryManager

`MemoryManager` uses USearchEngine as the primary semantic backend:

```python
# In MemoryManager.__init__():
usearch_config = USearchConfig.from_app_config(config)
embedder = LangchainQwenEmbeddings(...)
self._semantic_engine = USearchEngine(
    config=usearch_config,
    embedder=embedder,
    logger=self.logger
)
```

### Design Rationale

**Why USearchEngine?**

1. **Lightweight**: Minimal dependencies (usearch, libsql)
2. **Fast startup**: Simple initialization with lazy loading
3. **Predictable behavior**: No LLM inference side effects
4. **Testing friendly**: Easier to mock with simple embedder
5. **Clean interface**: Implements `ISemanticSearchEngine` protocol
6. **Persistent storage**: Index survives server restarts

**Storage Architecture:**
- **USearch Index**: Stores only (key → vector) mappings in `.usearch` file
- **libSQL Database**: Stores (id, project_id, message) records with WAL mode and MVCC
- **Key relationship**: libSQL auto-increment ID = USearch key

---

## LLMReranker (`llm_reranker.py`)

### Overview

LLM-based document reranker for post-fusion relevance scoring:
- Uses OpenRouter API for LLM inference (compatible with any OpenAI-compatible endpoint)
- Parallel scoring with concurrency control via `anyio.Semaphore`
- HTTP/2 enabled via `DefaultAioHttpClient` for better performance
- Graceful fallback: returns fusion score if LLM call fails
- JSON output format for deterministic score parsing

### Configuration

```python
@dataclass(frozen=True)
class LLMRerankerConfig:
    api_key: str                    # OpenRouter API key for authentication
    base_url: str                   # OpenRouter API base URL
    model: str                      # LLM model identifier (e.g., 'x-ai/grok-4.1-fast')
    score_threshold: float = 0.5   # Minimum relevance score to keep results (0.0-1.0)
    max_concurrency: int = 5       # Maximum parallel LLM calls
    timeout: float = 30.0          # HTTP request timeout in seconds
    min_results: int = 0           # Safety net: min results to return (0 = disabled)
    batch_normalize: bool = True   # Enable batch min-max normalization
```

**Factory Method:**
```python
config = LLMRerankerConfig.from_app_config(app_config)
```

This factory method creates an `LLMRerankerConfig` from the application's `Config` object, extracting `openrouter_api_key`, `openrouter_base_url`, `llm_model`, `search_score_threshold`, and `rerank_max_concurrency`.

### Class Definition

```python
class LLMReranker(BaseModel):
    """LLM-based document reranker using OpenRouter API."""

    config: LLMRerankerConfig
    logger: Any = None  # StructuredLogger

    _client: AsyncOpenAI | None = PrivateAttr(default=None)
```

### Key Methods

#### `rerank(query: str, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]`
Rerank candidates by LLM relevance scores:
```python
# candidates from RRF fusion: [(message, fusion_score), ...]
reranked = await reranker.rerank(
    query="Python tutorials",
    candidates=[("Python basics guide", 0.8), ("JavaScript intro", 0.6)],
)
# Returns: [("Python basics guide", 0.95), ...] - sorted by LLM score descending
```

**Behavior:**
- Scores each candidate document against the query using the LLM
- Uses `SCORING_PROMPT` template from `config/prompts.py`
- Filters by `score_threshold` (default: 0.5)
- Returns results sorted by relevance score descending
- Falls back to fusion score if LLM scoring fails for a document

#### `_score_single(query: str, document: str, fallback_score: float) -> Tuple[str, float]`
Score a single document's relevance (internal method):
```python
# Called internally by rerank() with concurrency control
result = await reranker._score_single(
    query="Python",
    document="Python is a programming language",
    fallback_score=0.7,  # fusion score used if LLM fails
)
# Returns: ("Python is a programming language", 0.95)
```

### Scoring Prompt

The LLMReranker uses `SCORING_PROMPT` from `config/prompts.py`:

```python
SCORING_PROMPT = """You are a relevance scoring system. Score how relevant a document is to a query.

⚠️ CRITICAL OUTPUT REQUIREMENTS ⚠️
Your output MUST be a SINGLE NUMBER between 0.0 and 1.0 (inclusive).
...

Query: "{query}"
Document: "{document}"
"""
```

The prompt:
- Requests JSON output format (`{"score": 0.85}`)
- Uses `temperature=0` for deterministic scoring
- Limits response to 50 tokens (only need short JSON)
- Clamps scores to valid 0.0-1.0 range

### Concurrency Control

Parallel scoring uses `anyio.Semaphore` for concurrency limiting:

```python
async def rerank(self, query: str, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    semaphore = anyio.Semaphore(self.config.max_concurrency)

    async def score_with_semaphore(idx: int, doc: str, fallback: float) -> None:
        async with semaphore:
            results[idx] = await self._score_single(query, doc, fallback)

    async with anyio.create_task_group() as tg:
        for idx, (document, fusion_score) in enumerate(candidates):
            tg.start_soon(score_with_semaphore, idx, document, fusion_score)
```

### Error Handling

The reranker implements graceful fallback:

```python
try:
    response = await self._client.chat.completions.create(...)
    result = json.loads(content)
    score = float(result.get("score", fallback_score))
except json.JSONDecodeError as e:
    # Invalid JSON from LLM, use fusion score as fallback
    return (document, fallback_score)
except Exception as e:
    # Any other error, use fusion score as fallback
    return (document, fallback_score)
```

### Usage Example

```python
from ccmemories.infrastructure import LLMReranker, LLMRerankerConfig

# Create configuration
config = LLMRerankerConfig(
    api_key="sk-or-...",
    base_url="https://openrouter.ai/api/v1",
    model="x-ai/grok-4.1-fast",
    score_threshold=0.5,
    max_concurrency=5,
)

# Initialize reranker
reranker = LLMReranker(config=config, logger=logger)

# Rerank candidates from RRF fusion
candidates = [
    ("Python programming guide", 0.8),
    ("JavaScript basics", 0.75),
    ("Data science with Python", 0.7),
]

reranked = await reranker.rerank("Python tutorials", candidates)
# Returns: [("Python programming guide", 0.95), ("Data science with Python", 0.85)]
# Note: "JavaScript basics" filtered out (score below threshold)
```

### Integration with MemoryManager

`MemoryManager` uses LLMReranker as Step 4 in the search pipeline:

```python
# In MemoryManager.__init__():
if self.config.reranker_engine == "llm":
    reranker_config = LLMRerankerConfig.from_app_config(config)
    self._llm_reranker = LLMReranker(config=reranker_config, logger=self.logger)

# In MemoryManager.search():
if self._llm_reranker is not None:
    hybrid_results = await self._llm_reranker.rerank(query, hybrid_results)
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RERANKER_ENGINE` | `llm` | Reranking engine: `llm`, `cross_encoder`, or `none` |
| `SEARCH_SCORE_THRESHOLD` | `0.5` | Minimum LLM relevance score to keep |
| `RERANK_MAX_CONCURRENCY` | `5` | Maximum parallel LLM calls |
| `LLM_MODEL` | `x-ai/grok-4.1-fast` | LLM model for scoring |
| `OPENROUTER_API_KEY` | - | API key for OpenRouter |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | API endpoint |
| `RERANKER_MIN_RESULTS` | `0` | Safety net: min results to return (0 = disabled) |
| `RERANKER_BATCH_NORMALIZE` | `true` | Enable batch min-max normalization |

### Design Rationale

**Why LLM Reranking?**

1. **Semantic understanding**: LLMs understand query intent beyond keyword matching
2. **Context awareness**: Can assess relevance considering full document context
3. **Quality filtering**: Removes false positives from vector/keyword search
4. **Flexible scoring**: 0.0-1.0 scale allows precise threshold tuning

**Architecture Decisions:**

- **Parallel scoring**: Uses `anyio.Semaphore` for concurrent LLM calls (default: 5)
- **Graceful fallback**: Falls back to fusion score on LLM errors (no hard failures)
- **JSON output**: Structured response format for reliable score parsing
- **Deterministic**: Uses `temperature=0` for consistent scoring

---

## CrossEncoderReranker (`cross_encoder_reranker.py`)

### Overview

Local cross-encoder reranker using FlagEmbedding's FlagReranker for fast, cost-free post-fusion relevance scoring:
- Uses FlagEmbedding's FlagReranker optimized for BGE reranker models
- Built-in FP16 support for faster inference
- Built-in score normalization (sigmoid to 0-1 range)
- Runs locally on CPU, CUDA, or MPS (Apple Silicon)
- Thread-safe lazy model loading
- Score threshold filtering
- Async wrapper for non-blocking execution in async contexts

### Configuration

```python
@dataclass(frozen=True)
class CrossEncoderConfig:
    model_name: str = "BAAI/bge-reranker-v2-m3"  # HuggingFace model identifier
    enabled: bool = True                         # Whether cross-encoder reranking is enabled
    top_k: int = 20                              # Number of top results to return
    device: str = "cpu"                          # Inference device: "cpu", "cuda", "mps"
    batch_size: int = 32                         # Batch size for inference
    score_threshold: float = 0.0                 # Minimum score to keep (0.0 = keep all)
    use_fp16: bool = True                        # Enable FP16 for faster inference
    normalize: bool = True                       # Normalize scores to 0-1 with sigmoid
    max_length: int = 512                        # Max token length for query-doc pairs
    min_results: int = 0                         # Safety net: min results to return (0 = disabled)
    batch_normalize: bool = True                 # Enable batch min-max normalization
```

**Factory Method:**
```python
config = CrossEncoderConfig.from_app_config(app_config)
```

This factory method creates a `CrossEncoderConfig` from the application's `Config` object when `reranker_engine == "cross_encoder"`.

### Class Definition

```python
from FlagEmbedding import FlagReranker

class CrossEncoderReranker(BaseModel):
    """Cross-encoder reranker using FlagEmbedding's FlagReranker."""

    config: CrossEncoderConfig
    logger: Any = None  # StructuredLogger

    _model: FlagReranker | None = PrivateAttr(default=None)
    _init_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
```

### Key Methods

#### `rerank(query: str, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]`
Rerank candidates by FlagReranker scores (synchronous):
```python
# candidates from RRF fusion: [(message, fusion_score), ...]
reranked = reranker.rerank(
    query="Python tutorials",
    candidates=[("Python basics guide", 0.8), ("JavaScript intro", 0.6)],
)
# Returns: [("Python basics guide", 0.95), ...] - sorted by cross-encoder score descending
```

**Behavior:**
- Scores each candidate document against the query using FlagReranker
- With `normalize=True`, scores are in [0, 1] range (sigmoid applied)
- Filters by `score_threshold` (default: 0.0 = keep all)
- Returns top_k results sorted by relevance score descending
- If disabled, returns candidates unchanged (pass-through)

#### `rerank_async(query: str, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]`
Async wrapper for cross-encoder reranking:
```python
# Non-blocking version using asyncer.asyncify
reranked = await reranker.rerank_async("Python tutorials", candidates)
```

Uses `asyncify()` from asyncer to run CPU/GPU-intensive inference in a thread pool without blocking the event loop.

### Lazy Initialization

The FlagReranker model is lazily loaded on first use with thread-safe initialization:

```python
@property
def model(self) -> FlagReranker:
    """Get FlagReranker model (thread-safe lazy initialization)."""
    if self._model is not None:
        return self._model
    with self._init_lock:
        if self._model is None:
            self._model = FlagReranker(
                self.config.model_name,
                use_fp16=self.config.use_fp16,
                devices=[self.config.device],
            )
    return self._model
```

### Usage Example

```python
from ccmemories.infrastructure import CrossEncoderConfig, CrossEncoderReranker

# Create configuration
config = CrossEncoderConfig(
    model_name="BAAI/bge-reranker-v2-m3",
    enabled=True,
    top_k=10,
    device="cpu",
    use_fp16=True,
    normalize=True,
    score_threshold=0.5,
)

# Initialize reranker
reranker = CrossEncoderReranker(config=config, logger=logger)

# Rerank candidates from RRF fusion
candidates = [
    ("Python programming guide", 0.8),
    ("JavaScript basics", 0.75),
    ("Data science with Python", 0.7),
]

reranked = reranker.rerank("Python tutorials", candidates)
# Returns: [("Python programming guide", 0.95), ("Data science with Python", 0.85)]
```

### Integration with MemoryManager

`MemoryManager` uses CrossEncoderReranker as Step 4 when `RERANKER_ENGINE=cross_encoder`:

```python
# In MemoryManager.__init__():
if self.config.reranker_engine == "cross_encoder":
    ce_config = CrossEncoderConfig.from_app_config(config)
    self._cross_encoder_reranker = CrossEncoderReranker(
        config=ce_config, logger=self.logger
    )

# In MemoryManager.search():
if self._cross_encoder_reranker is not None:
    hybrid_results = await self._cross_encoder_reranker.rerank_async(query, hybrid_results)
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RERANKER_ENGINE` | `llm` | Set to `cross_encoder` to use CrossEncoderReranker |
| `CROSS_ENCODER_MODEL` | `BAAI/bge-reranker-v2-m3` | HuggingFace model identifier |
| `CROSS_ENCODER_TOP_K` | `20` | Max results to return after reranking |
| `CROSS_ENCODER_DEVICE` | `cpu` | Inference device: `cpu`, `cuda`, `mps` |
| `CROSS_ENCODER_BATCH_SIZE` | `32` | Batch size for inference |
| `CROSS_ENCODER_SCORE_THRESHOLD` | `0.5` | Minimum score to keep (0.0 = keep all) |
| `CROSS_ENCODER_USE_FP16` | `true` | Enable FP16 for faster inference |
| `CROSS_ENCODER_NORMALIZE` | `true` | Normalize scores to 0-1 with sigmoid |
| `CROSS_ENCODER_MAX_LENGTH` | `512` | Max token length for query-doc pairs |
| `RERANKER_MIN_RESULTS` | `0` | Safety net: min results to return (0 = disabled) |
| `RERANKER_BATCH_NORMALIZE` | `true` | Enable batch min-max normalization |

### Design Rationale

**Why FlagReranker?**

1. **No API costs**: Runs entirely locally, no per-query charges
2. **Low latency**: Fast local inference, no network round-trips
3. **Privacy**: Data never leaves your infrastructure
4. **Predictable performance**: No rate limits or API failures
5. **Quality**: BGE rerankers provide high-quality relevance scoring
6. **FP16 support**: Built-in FP16 mode for faster inference
7. **Score normalization**: Built-in sigmoid normalization to 0-1 range

**Architecture Decisions:**

- **FlagReranker**: Optimized for BGE reranker models from FlagEmbedding
- **Lazy loading**: Model only loaded when first used (saves memory if unused)
- **Thread-safe**: Double-checked locking for safe concurrent access
- **Async wrapper**: Uses `asyncify()` to prevent blocking event loop
- **Score normalization**: `normalize=True` applies sigmoid for consistent 0-1 scores
- **FP16 mode**: `use_fp16=True` enables faster inference with minimal quality loss
- **Batch inference**: Efficient GPU utilization with configurable batch size

**Comparison with LLMReranker:**

| Aspect | CrossEncoderReranker | LLMReranker |
|--------|---------------------|-------------|
| Cost | Free (local) | Per-token API costs |
| Latency | Low (~50-200ms) | Higher (~500-2000ms) |
| Quality | High (specialized models) | Very high (LLM understanding) |
| Privacy | Data stays local | Data sent to API |
| Scalability | CPU/GPU bound | API rate limited |

---

## SmartReplacer (`smart_replacer.py`)

### Overview

LLM-based memory replacement detection for intelligent memory management:
- Detects when a new memory semantically replaces an existing one
- Example: "I like cats" → "I don't like cats anymore, I like dogs"
- Uses OpenRouter API with structured JSON output for reliable parsing
- Configurable confidence threshold (default: 0.7)
- Graceful degradation: if LLM call fails, memory is added normally
- HTTP/2 enabled via `DefaultAioHttpClient` for performance

### Configuration

```python
@dataclass(frozen=True)
class SmartReplacerConfig:
    api_key: str                    # OpenRouter API key
    base_url: str                   # OpenRouter API base URL
    model: str                      # LLM model identifier
    threshold: float = 0.7          # Min confidence to trigger replacement (0.0-1.0)
    enabled: bool = True            # Whether smart replacement is enabled
    timeout: float = 30.0           # HTTP request timeout in seconds
    max_retries: int = 3            # Max LLM call retries with exponential backoff
    retry_delay: float = 1.0        # Base delay (seconds) for exponential backoff
```

**Factory Method:**
```python
config = SmartReplacerConfig.from_app_config(app_config)
```

**Retry Behavior:**
- Implements exponential backoff: delay = `retry_delay * 2^(attempt-1)`
- Default: 1s, 2s, 4s between retries
- Falls back to json_object mode if structured output not supported
- Returns safe defaults `(False, 0.0, "Error: ...")` after all retries exhausted

### Class Definition

```python
class SmartReplacer(BaseModel):
    """LLM-based memory replacement detection."""

    config: SmartReplacerConfig
    logger: Any = None  # StructuredLogger

    _client: AsyncOpenAI | None = PrivateAttr(default=None)
```

### Key Methods

#### `check_replacement(new_memory: str, existing_memory: str) -> Tuple[bool, float, str]`
Check if new memory should replace existing memory:
```python
should_replace, confidence, reason = await replacer.check_replacement(
    new_memory="I don't like cats anymore, I prefer dogs",
    existing_memory="I like cats",
)
# Returns: (True, 0.92, "Updated preference from cats to dogs")
```

**Behavior:**
- Uses `REPLACEMENT_DETECTION_PROMPT` template from `config/prompts.py`
- Returns `ReplacementDecision` with `should_replace`, `confidence`, `reason`
- Uses structured JSON output with `json_schema` response format
- Falls back to `(False, 0.0, "")` on any error (graceful degradation)

### Replacement Decision

```python
class ReplacementDecision(BaseModel):
    """LLM response for replacement detection."""
    model_config = ConfigDict(strict=True)

    should_replace: bool    # Whether replacement is recommended
    confidence: float       # Confidence score (0.0-1.0)
    reason: str            # Explanation for the decision
```

### Prompt Template

The SmartReplacer uses `REPLACEMENT_DETECTION_PROMPT` from `config/prompts.py`:

```python
REPLACEMENT_DETECTION_PROMPT = """You are a memory replacement detection system...

Analyze whether the NEW memory should replace the EXISTING memory.

Replacement criteria:
- Same topic with updated information
- Contradictory statements about the same thing
- New preference replacing old preference
- Updated facts about the same entity

Existing Memory: "{existing_memory}"
New Memory: "{new_memory}"

OUTPUT: JSON with should_replace (bool), confidence (0.0-1.0), reason (string)
"""
```

### Error Handling

The SmartReplacer implements graceful degradation:

```python
try:
    response = await self._client.chat.completions.create(...)
    decision = ReplacementDecision.model_validate_json(content)
    return (decision.should_replace and decision.confidence >= threshold,
            decision.confidence, decision.reason)
except Exception as e:
    # Any error: log warning, return safe default
    if self.logger:
        self.logger.warning(f"Smart replacement check failed: {e}")
    return (False, 0.0, "")  # Don't replace on error
```

### Usage Example

```python
from ccmemories.infrastructure import SmartReplacer, SmartReplacerConfig

# Create configuration
config = SmartReplacerConfig(
    api_key="sk-or-...",
    base_url="https://openrouter.ai/api/v1",
    model="x-ai/grok-4.1-fast",
    threshold=0.7,
)

# Initialize replacer
replacer = SmartReplacer(config=config, logger=logger)

# Check if replacement is needed
should_replace, confidence, reason = await replacer.check_replacement(
    new_memory="I moved to San Francisco",
    existing_memory="I live in New York",
)

if should_replace:
    print(f"Replace! Confidence: {confidence:.2f}, Reason: {reason}")
    # Delete old memory, add new one
else:
    print("No replacement needed, add normally")
```

### Integration with MemoryManager

`MemoryManager` uses SmartReplacer during the add operation:

```python
# In MemoryManager.__init__():
if self.config.enable_smart_replace:
    smart_replacer_config = SmartReplacerConfig.from_app_config(config)
    self._smart_replacer = SmartReplacer(
        config=smart_replacer_config, logger=self.logger
    )

# In MemoryManager.add_messages_async():
async def add_messages_async(self, messages: List[str]) -> int:
    for message in messages:
        # Step 1: Check for replacement
        replacement_target = await self._check_for_replacement(message)

        # Step 2: Delete old memory if replacement needed
        if replacement_target:
            self.delete_by_message(replacement_target)

        # Step 3: Add new memory
        self._add_message(message)
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_SMART_REPLACE` | `true` | Enable smart memory replacement detection |
| `SMART_REPLACE_THRESHOLD` | `0.7` | Min LLM confidence to trigger replacement (0.0-1.0) |
| `LLM_MODEL` | `x-ai/grok-4.1-fast` | LLM model (shared with reranking) |
| `OPENROUTER_API_KEY` | - | API key for OpenRouter |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | API endpoint |

### Design Rationale

**Why Smart Replacement?**

1. **Memory consistency**: Prevents contradictory memories from coexisting
2. **Quality over quantity**: Outdated information is replaced, not accumulated
3. **Natural updates**: Handles preference changes, fact updates, location changes
4. **User experience**: AI agents retrieve accurate, up-to-date information

**Architecture Decisions:**

- **Structured output**: Uses `json_schema` response format for reliable JSON parsing
- **Graceful degradation**: Falls back to normal add on any error (no hard failures)
- **Single candidate**: Checks only the most similar memory (top 1) for efficiency
- **Configurable threshold**: Default 0.7 balances accuracy and false positives
- **HTTP/2**: Uses `DefaultAioHttpClient` for better performance

**Replacement Criteria:**

The LLM evaluates these scenarios:
- Same topic with updated information ("I'm 25" → "I'm 26")
- Contradictory statements ("I like X" → "I don't like X anymore")
- New preference replacing old ("favorite color is blue" → "favorite color is green")
- Updated facts about same entity ("lives in NYC" → "moved to SF")

---

## MessageStore (`message_store.py`)

### Overview

libSQL-backed message text storage for use with USearchEngine:
- Persistent storage with WAL mode and MVCC for concurrent writes
- Auto-increment IDs serve as USearch keys
- Thread-safe with lazy connection initialization
- User/project-level filtering via `project_id` column

**Why libSQL?** libSQL is a high-performance SQLite fork by Turso with MVCC (Multi-Version Concurrency Control) support, eliminating SQLite's single-writer bottleneck while maintaining full SQL compatibility.

### Configuration

```python
store = MessageStore(
    db_path="indexes/project/usearch/messages.db",
    logger=logger  # Optional StructuredLogger
)
```

### Key Methods

| Method | Description |
|--------|-------------|
| `insert(project_id, message) -> int` | Insert message, return ID |
| `get(msg_id) -> MessageRecord \| None` | Get record by ID |
| `get_all(project_id) -> List[str]` | Get all messages for user |
| `delete(msg_id) -> bool` | Delete by ID, return success |
| `exists(project_id, message) -> bool` | Check for duplicates |
| `get_id_by_message(project_id, message) -> int \| None` | Find ID by content |
| `close()` | Close libSQL connection |
| `ensure_initialized()` | Force connection creation |

### Usage Example

```python
from ccmemories.infrastructure.message_store import MessageStore

store = MessageStore(db_path="/tmp/messages.db")

# Insert messages
id1 = store.insert("project-a", "Hello world")
id2 = store.insert("project-a", "Goodbye world")

# Check for duplicates before insert
if not store.exists("project-a", "Hello world"):
    store.insert("project-a", "Hello world")

# Retrieve all for project
messages = store.get_all("project-a")  # ["Hello world", "Goodbye world"]

# Delete by ID
store.delete(id1)

# Cleanup
store.close()
```

---

## TantivyEngine (`tantivy_engine.py`)

### Overview

Full-text search engine wrapper around the [Tantivy](https://github.com/quickwit-oss/tantivy) library:
- Persistent index storage on disk
- Lazy initialization of writer and searcher
- Stemmed full-text search (`en_stem` tokenizer)
- User/project-level filtering via `project_id` field

### Configuration

```python
@dataclass(frozen=True)
class TantivyConfig:
    project_id: str                          # Unique project identifier for filtering
    index_path: str                          # Path to the Tantivy index directory
    soft_delete_enabled: bool = True         # O(1) tombstone marking instead of O(n) rebuild
    compaction_threshold_ratio: float = 0.2  # Compact when tombstones > this ratio of docs
    compaction_max_tombstones: int = 10000   # Force compaction above this tombstone count
    tombstone_ttl_days: int = 7              # Days before tombstones eligible for removal
```

**Factory Method:**
```python
config = TantivyConfig.from_app_config(app_config)
```

### Class Definition

```python
class TantivyEngine(BaseModel):
    """Tantivy full-text search engine wrapper with soft-delete support."""

    config: TantivyConfig
    logger: Any = None  # StructuredLogger

    _index: Optional[tantivy.Index] = PrivateAttr(default=None)
    _writer: Optional[tantivy.IndexWriter] = PrivateAttr(default=None)
    _searcher: Optional[tantivy.Searcher] = PrivateAttr(default=None)
    _schema_version: int = PrivateAttr(default=2)  # V2 includes soft-delete fields
```

### Schema

The engine uses a schema with soft-delete support (V2):

| Field | Tokenizer | Type | Purpose |
|-------|-----------|------|---------|
| `project_id` | `raw` | TEXT | Exact match filtering by project |
| `message` | `en_stem` | TEXT | Full-text search with English stemming |
| `is_deleted` | - | u64 | Tombstone flag (0 = active, 1 = deleted) |
| `deleted_at` | - | i64 | Deletion timestamp in milliseconds |

**Schema Versioning:**
- **V1**: Original schema (project_id, message only)
- **V2**: Soft-delete schema (+ is_deleted, deleted_at fields)

Indexes are automatically migrated to V2 on load via `migrate_to_v2()`.

### Key Methods

#### `add(project_id: str, message: str) -> None`
Add a document to the index:
```python
engine.add(project_id="my-project", message="Python is a programming language")
```

#### `commit() -> None`
Commit pending changes and refresh searcher (non-blocking):
```python
engine.commit()  # Flushes writes to disk, reloads searcher
```

**Performance optimization**: After `commit()`, the IndexWriter remains valid and is reused for subsequent `add()` calls. This avoids expensive writer recreation on every commit cycle. Background merge threads continue running asynchronously.

#### `flush() -> None`
Commit and wait for all background merging to complete (blocking):
```python
engine.flush()  # Commits, waits for merge threads, invalidates writer
```

**Use cases**:
- Before reading index from another process
- Before backup operations
- When guaranteed durability of all segments is required

Unlike `commit()`, the writer becomes invalid after `flush()` and will be recreated on next write.

#### `delete(project_id: str, message: str) -> bool`
Delete a document by exact message match:
```python
result = engine.delete(project_id="my-project", message="Python is a programming language")
# Returns: True if document was found and deleted, False otherwise
```

**Important:** This method commits internally for both soft-delete and rebuild modes. Callers do not need to call `commit()` after `delete()`.

**Behavior with Soft-Delete (default):**
When `soft_delete_enabled=True` and the index has V2 schema:
- Uses O(1) tombstone marking via `soft_delete()` method
- Document is marked with `is_deleted=1` and `deleted_at=<timestamp_ms>`
- Document remains in index but is filtered from search results
- Periodic `compact()` removes tombstoned documents
- Commits and invalidates tombstone cache automatically

**Fallback to Hard Delete:**
When soft-delete is disabled or index has V1 schema:
- Uses O(n) rebuild approach
- Gets all documents, filters target, recreates index, re-adds remaining
- Commits via `_rebuild_index_with_docs()` automatically

#### `soft_delete(project_id: str, message: str) -> bool`
Mark a document as deleted (tombstone) without rebuilding:
```python
result = engine.soft_delete(project_id="my-project", message="Some message")
# Returns: True if document was found and marked deleted, False otherwise
```

**Performance:** O(1) operation - finds doc address and updates in-place.

#### `get_tombstone_stats() -> dict`
Get statistics about tombstoned documents:
```python
stats = engine.get_tombstone_stats()
# Returns: {
#     "total_docs": 1000,
#     "tombstoned": 150,
#     "active": 850,
#     "tombstone_ratio": 0.15,
#     "oldest_tombstone_days": 5.2
# }
```

#### `needs_compaction() -> bool`
Check if index needs compaction based on configured thresholds:
```python
if engine.needs_compaction():
    engine.compact()
```

**Triggers compaction when:**
- `tombstone_ratio > compaction_threshold_ratio` (default: 0.2)
- `tombstoned_count > compaction_max_tombstones` (default: 10000)

#### `compact(force: bool = False) -> dict`
Remove tombstoned documents by rebuilding the index:
```python
# Only compact if thresholds exceeded
result = engine.compact()

# Force compaction regardless of thresholds
result = engine.compact(force=True)

# Returns: {
#     "removed": 150,
#     "remaining": 850,
#     "duration_ms": 1234
# }
```

#### `migrate_to_v2() -> dict`
Migrate V1 index to V2 schema with soft-delete fields:
```python
result = engine.migrate_to_v2()
# Returns: {
#     "migrated": True,
#     "from_version": 1,
#     "to_version": 2,
#     "doc_count": 1000
# }
```

**Safe to call on V2 indexes:** Returns `{"migrated": False}` if already V2.

#### `search(query: str, project_id: str, limit: int) -> List[Tuple[str, float]]`
Execute full-text search:
```python
results = engine.search(
    query="programming language",
    project_id="my-project",
    limit=10,
)
# Returns: [("Python is a programming language", 2.5), ...]
```

### Lazy Initialization

Writer and searcher are lazily initialized on first access with thread-safe double-checked locking:

```python
@property
def writer(self) -> tantivy.IndexWriter:
    if self._writer is not None:
        return self._writer
    with self._writer_lock:
        if self._writer is None:
            self._writer = self._index.writer()
        return self._writer

@property
def searcher(self) -> tantivy.Searcher:
    if self._searcher is not None:
        return self._searcher
    with self._searcher_lock:
        if self._searcher is None:
            self._searcher = self._index.searcher()
        return self._searcher
```

### Writer Lifecycle (Optimized)

The IndexWriter lifecycle is optimized for performance by reusing writers across commit cycles:

**Key insight from tantivy-py**:
- `commit()` takes `&mut self` (mutable borrow) - does NOT consume the writer
- `wait_merging_threads()` takes `self` (ownership) - DOES consume the writer

**Lifecycle methods**:
- **`commit()`**: Flushes writes, reloads index, refreshes searcher. Writer remains valid for reuse.
- **`flush()`**: Like commit but also waits for merging threads. Writer becomes invalid.
- **`close()`**: Full cleanup - commits, waits for threads, releases all resources.

**Thread safety**: All writer operations are protected by `_writer_lock`

```python
def commit(self) -> None:
    """Non-blocking commit - writer stays valid for reuse."""
    with self._writer_lock:
        if self._writer:
            self._writer.commit()
            # Writer remains valid for subsequent add() calls
            # Background merge threads continue asynchronously

def flush(self) -> None:
    """Blocking flush - waits for merges, invalidates writer."""
    with self._writer_lock:
        if self._writer:
            self._writer.commit()
            self._writer.wait_merging_threads()
            self._writer = None  # Consumed by wait_merging_threads()
```

**Performance impact**:
- Before: Writer recreated after every `commit()` (~5-10ms overhead)
- After: Writer reused across multiple add-commit cycles (~0ms overhead)
- Improvement: ~100x faster for add-after-commit operations

**When to use each method**:
| Method | Use case |
|--------|----------|
| `commit()` | Normal operations, high-throughput indexing |
| `flush()` | Before backup, cross-process access, guaranteed durability |
| `close()` | Application shutdown, resource cleanup |

### Internal Helper Methods

These methods support the delete operation:

#### `_get_all_docs(project_id: str) -> List[str]`
Get all messages for a specific project (used for verification):
```python
docs = engine._get_all_docs("my-project")
# Returns: ["Message 1", "Message 2", ...]
```

#### `_get_all_docs_all_projects() -> List[Tuple[str, str]]`
Get all documents from all projects (used by delete rebuild):
```python
all_docs = engine._get_all_docs_all_projects()
# Returns: [("project-a", "Message 1"), ("project-b", "Message 2"), ...]
```

#### `_recreate_writer_if_needed() -> None`
Force writer recreation (caller must hold `_writer_lock`):
```python
with engine._writer_lock:
    engine._recreate_writer_if_needed()
```

### Index Persistence

The engine automatically handles index persistence:
- Creates directory if it doesn't exist
- Attempts to load existing index on startup
- Creates new index if none exists
- Uses `reuse=True` for safe reopening

```python
try:
    self._index = tantivy.Index.open(index_path)
except Exception:
    self._index = tantivy.Index(schema, path=index_path, reuse=True)
```

### Usage Example

```python
from ccmemories.infrastructure import TantivyConfig, TantivyEngine

config = TantivyConfig(
    project_id="my-project",
    index_path="indexes/my-project/tantivy",
)

engine = TantivyEngine(config=config, logger=logger)

# Add documents
engine.add("my-project", "Python is great for data science")
engine.add("my-project", "JavaScript powers the web")
engine.commit()

# Search
results = engine.search("data science", project_id="my-project", limit=5)
for message, score in results:
    print(f"{score:.2f}: {message}")
```

---

## CachedEmbeddings (`cached_embeddings.py`)

### Overview

LRU caching wrapper for query embedding operations:
- Wraps any `langchain_core.embeddings.Embeddings` provider
- Caches `embed_query()` and `aembed_query()` results using MD5 hash as cache key
- Thread-safe LRU eviction using `OrderedDict` with `threading.Lock`
- Does NOT cache `embed_documents()` (documents cached once on ingestion)
- Configurable cache size (default: 100 entries)
- Cache hit/miss statistics

### Configuration

```python
cached_embedder = CachedEmbeddings(
    embedder=base_embedder,  # Any Embeddings provider
    cache_size=100,          # Maximum cached embeddings
    enabled=True,            # Enable/disable caching
    logger=logger,           # Optional: for cache hit/miss logging
)
```

### Class Definition

```python
class CachedEmbeddings(BaseModel, Embeddings):
    """LRU caching wrapper for any Embeddings provider."""

    embedder: Embeddings      # Wrapped embeddings provider
    cache_size: int = 100     # Max cached entries
    enabled: bool = True      # Enable/disable caching
    logger: Any = None        # Optional StructuredLogger

    _cache: OrderedDict[str, List[float]] = PrivateAttr(default_factory=OrderedDict)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _hits: int = PrivateAttr(default=0)
    _misses: int = PrivateAttr(default=0)
```

### Key Methods

#### `embed_query(text: str) -> List[float]`
Embed query text with LRU caching:
```python
# First call computes embedding (cache miss)
embedding1 = cached_embedder.embed_query("Python programming")

# Second call returns cached result (cache hit)
embedding2 = cached_embedder.embed_query("Python programming")

assert embedding1 == embedding2  # Same result, second call is instant
```

#### `aembed_query(text: str) -> List[float]`
Async version of `embed_query` with caching:
```python
embedding = await cached_embedder.aembed_query("Python programming")
```

#### `embed_documents(texts: List[str]) -> List[List[float]]`
Pass-through to wrapped embedder (NOT cached):
```python
# Documents are typically embedded once during ingestion
embeddings = cached_embedder.embed_documents(["doc1", "doc2"])
```

#### `get_cache_stats() -> dict`
Get cache hit/miss statistics:
```python
stats = cached_embedder.get_cache_stats()
# Returns: {
#     "hits": 150,
#     "misses": 50,
#     "size": 45,
#     "max_size": 100
# }
```

#### `clear_cache() -> None`
Clear cache and reset statistics:
```python
cached_embedder.clear_cache()
```

### Usage Example

```python
from ccmemories.infrastructure import LangchainQwenEmbeddings, CachedEmbeddings

# Create base embedder
base_embedder = LangchainQwenEmbeddings({
    "model": "qwen/qwen3-embedding-8b",
    "embedding_dims": 4096,
})

# Wrap with caching
cached_embedder = CachedEmbeddings(
    embedder=base_embedder,
    cache_size=100,
    enabled=True,
)

# Use in search operations
embedding = cached_embedder.embed_query("semantic search query")

# Check cache performance
print(cached_embedder.get_cache_stats())
```

### Integration with MemoryManager

`MemoryManager` uses CachedEmbeddings when `EMBEDDING_CACHE_ENABLED=true`:

```python
# In MemoryManager.__init__():
if self.config.embedding_cache_enabled:
    self._embedder = CachedEmbeddings(
        embedder=base_embedder,
        cache_size=self.config.embedding_cache_size,
        enabled=True,
        logger=self.logger,
    )
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_CACHE_ENABLED` | `true` | Enable query embedding caching |
| `EMBEDDING_CACHE_SIZE` | `100` | Maximum cached embeddings |

### Design Rationale

**Why Cache Query Embeddings?**

1. **Search refinement**: Same query may be searched multiple times during result refinement
2. **Hybrid search**: Query is embedded once, reused for USearch + potential retries
3. **Cost reduction**: Fewer API calls for repeated queries
4. **Latency reduction**: Cached queries return instantly

**Why NOT Cache Document Embeddings?**

1. **One-time operation**: Documents are embedded once during ingestion
2. **Memory consumption**: Caching all document embeddings would use excessive memory
3. **Limited benefit**: Documents are rarely re-embedded

---

## LangchainQwenEmbeddings (`qwen3_embedding.py`)

### Overview

Custom embedding implementation using OpenRouter API with:
- Both sync and async client support
- HTTP/2 enabled via `DefaultAioHttpClient`
- Configurable retry logic (3 attempts)
- Batching for large document sets (batch size: 64)
- Concurrency control via `anyio.Semaphore` (max 32 concurrent)

### Class Definition

```python
class LangchainQwenEmbeddings(BaseModel, Embeddings):
    """Langchain-compatible embeddings via OpenRouter."""

    config: Any = None
    _client: OpenAI | None = PrivateAttr(default=None)
    _async_client: AsyncOpenAI | None = PrivateAttr(default=None)

    def __init__(self, config: BaseEmbedderConfig | dict[str, Any] | None = None):
        # Initialize with config
        # Set up sync and async OpenAI clients
        # Configure HTTP/2 support
```

### Key Methods

#### Synchronous Methods

```python
def embed_query(self, text: str) -> list[float]:
    """Embed query text using synchronous client with retries."""
    text = text.replace("\n", " ")
    embeddings = self._sync_embed_with_retry(
        input=[text],
        model=self.config.model,
        dimensions=self.config.embedding_dims,
    )
    return embeddings[0] if embeddings else []

def embed_documents(self, texts: list[str]) -> list[list[float]]:
    """Embed documents with batching (batch_size=64)."""
    # Processes in batches to avoid oversized requests
```

#### Asynchronous Methods

```python
async def aembed_query(self, text: str) -> list[float]:
    """Async version: Embed query text with retries."""
    embeddings = await self._async_embed_with_retry(...)
    return embeddings[0] if embeddings else []

async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
    """Async version with concurrency limiting via anyio.Semaphore."""
    max_concurrent = min(32, len(texts))
    semaphore = anyio.Semaphore(max_concurrent)
    # Process texts concurrently within semaphore limits
```

### Configuration

The class reads configuration from:

| Config Key | Env Variable | Default | Description |
|------------|--------------|---------|-------------|
| `api_key` | `OPENROUTER_API_KEY` | - | OpenRouter API key |
| `openai_base_url` | `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | API base URL |
| `model` | `EMBEDDING_MODEL` | - | Embedding model name |
| `embedding_dims` | `QWEN_EMBEDDING_DIMS` | `1536` | Embedding dimensions |

### Integration with USearchEngine

When `EMBEDDER_PROVIDER=langchain`, USearchEngine uses this class:

```python
# In MemoryManager initialization
if config.embedder_provider == "langchain":
    embedder_config = {
        "provider": "langchain",
        "config": {
            "class": "ccmemories.infrastructure.qwen3_embedding.LangchainQwenEmbeddings",
            "config": {
                "model": config.embedding_model,
                "embedding_dims": config.qwen_embedding_dims,
                "api_key": config.openrouter_api_key,
            }
        }
    }
```

---

## Key Features

### HTTP/2 Support

Uses `DefaultHttpxClient` and `DefaultAioHttpClient` for HTTP/2:
- Better connection reuse and multiplexing
- Reduced latency for multiple requests
- Improved performance under load

```python
self._client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    http_client=DefaultHttpxClient(http2=True),
)

self._async_client = AsyncOpenAI(
    api_key=api_key,
    base_url=base_url,
    http_client=DefaultAioHttpClient(http2=True),
)
```

### Retry Logic

Both sync and async methods implement retry with configurable attempts:

```python
def _sync_embed_with_retry(self, **kwargs: Any) -> list[list[float]]:
    """Call sync embeddings API with basic retry."""
    max_attempts = 3
    last_exc: Exception | None = None
    for _ in range(1, max_attempts + 1):
        try:
            response = self._client.embeddings.create(**kwargs)
            return [d.embedding for d in response.data]
        except Exception as exc:
            last_exc = exc
    raise RuntimeError("Embedding request failed after retries") from last_exc
```

### Concurrency Control

Async methods use `anyio.Semaphore` to limit concurrent requests:

```python
async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
    max_concurrent = min(32, len(texts))
    semaphore = anyio.Semaphore(max_concurrent)

    async def bounded_embed(index: int, text: str) -> None:
        async with semaphore:
            embeddings = await self._async_embed_with_retry(...)
            results[index] = embeddings[0]

    async with anyio.create_task_group() as tg:
        for i, text in enumerate(texts):
            tg.start_soon(bounded_embed, i, text)
```

### Batching

Document embedding uses batching to avoid oversized requests:

```python
batch_size = 64
for i in range(0, len(cleaned_texts), batch_size):
    batch = cleaned_texts[i : i + batch_size]
    batch_embeddings = self._sync_embed_with_retry(input=batch, ...)
    results.extend(batch_embeddings)
```

---

## Usage Examples

### TantivyEngine Direct Usage

```python
from ccmemories.infrastructure import TantivyConfig, TantivyEngine

config = TantivyConfig(
    project_id="test-project",
    index_path="/tmp/tantivy-test",
)
engine = TantivyEngine(config=config)

# Index documents
engine.add("test-project", "Machine learning fundamentals")
engine.add("test-project", "Deep learning with neural networks")
engine.commit()

# Search
results = engine.search("neural networks", project_id="test-project", limit=5)
```

### LangchainQwenEmbeddings Direct Usage

```python
from ccmemories.infrastructure import LangchainQwenEmbeddings

# Initialize with config
embeddings = LangchainQwenEmbeddings({
    "model": "openai/text-embedding-3-large",
    "embedding_dims": 3072,
    "api_key": "sk-or-...",
})

# Synchronous usage
query_embedding = embeddings.embed_query("What is Python?")

# Asynchronous usage
async def generate_embeddings():
    docs = ["Doc 1", "Doc 2", "Doc 3"]
    return await embeddings.aembed_documents(docs)
```

### With Environment Variables

```python
import os
from ccmemories.infrastructure import LangchainQwenEmbeddings

# Environment variables:
# OPENROUTER_API_KEY=sk-or-...
# OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

embeddings = LangchainQwenEmbeddings({
    "model": "qwen3-embedding",
    "embedding_dims": 4096,
})
```

---

## Error Handling

### TantivyEngine

Graceful degradation on search failure:
```python
try:
    # Search logic
except Exception as e:
    if self.logger:
        self.logger.warning("Tantivy full-text search failed", extra={"error": str(e)})
    return []  # Return empty instead of raising
```

### LangchainQwenEmbeddings

Errors are wrapped to prevent API key leakage:
```python
try:
    response = self._client.embeddings.create(**kwargs)
    return [d.embedding for d in response.data]
except Exception as exc:
    last_exc = exc
# After all retries fail:
raise RuntimeError("Embedding request failed after retries") from last_exc
```

---

## Testing

### TantivyEngine Unit Tests

```python
class TestTantivyEngine:
    def test_add_and_search(self, tmp_path):
        """Test basic add and search operations."""
        config = TantivyConfig(project_id="test", index_path=str(tmp_path))
        engine = TantivyEngine(config=config)

        engine.add("test", "Hello world")
        engine.commit()

        results = engine.search("hello", project_id="test", limit=5)
        assert len(results) == 1
        assert "Hello world" in results[0][0]

    def test_project_id_filtering(self, tmp_path):
        """Test that results are filtered by project_id."""
        config = TantivyConfig(project_id="test", index_path=str(tmp_path))
        engine = TantivyEngine(config=config)

        engine.add("project-a", "Message for A")
        engine.add("project-b", "Message for B")
        engine.commit()

        results = engine.search("message", project_id="project-a", limit=5)
        assert len(results) == 1
        assert "A" in results[0][0]
```

### LangchainQwenEmbeddings Unit Tests

```python
class TestLangchainQwenEmbeddings:
    def test_sync_embed_query(self):
        """Test synchronous query embedding."""
        with patch.object(embeddings._client, 'embeddings') as mock:
            mock.create.return_value = Mock(data=[Mock(embedding=[0.1, 0.2])])
            result = embeddings.embed_query("test")
            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_async_embed_documents(self):
        """Test asynchronous document embedding."""
        with patch.object(embeddings._async_client, 'embeddings') as mock:
            mock.create.return_value = Mock(data=[Mock(embedding=[0.1, 0.2])])
            results = await embeddings.aembed_documents(["doc1", "doc2"])
            assert len(results) == 2
```

---

## Dependencies

- `openai`: OpenAI Python client with async support
- `anyio`: Async concurrency primitives
- `langchain`: Embedding provider interface
- `langchain_core`: Core embeddings base class
- `pydantic`: Model validation
- `tantivy`: Full-text search library
- `usearch`: HNSW vector search library
- `libsql`: High-performance SQLite fork with MVCC

---

## Best Practices

1. **API Keys**: Store in environment variables, never in code
2. **Concurrency**: Respect rate limits with semaphore controls
3. **Batching**: Process large document sets in batches
4. **Error Handling**: Implement retry logic for transient failures
5. **Index Commits**: Call `commit()` after batch operations, not after every add
6. **HTTP/2**: Use for better performance with multiple requests
7. **Lazy Init**: Both engines use lazy initialization for resources

---

## Architecture Notes

This infrastructure provides:
- **USearchEngine**: Primary semantic search engine with USearch HNSW + libSQL storage
- **TantivyEngine**: Full-text search with persistence, project filtering, and O(1) soft-delete
- **MessageStore**: libSQL-backed text storage for USearch vectors (MVCC for concurrent writes)
- **CachedEmbeddings**: LRU caching wrapper for query embeddings (reduces API calls)
- **LangchainQwenEmbeddings**: Async-capable embedding client with retry logic and batching
- **LLMReranker**: AI-powered relevance scoring with parallel execution and graceful fallback
- **CrossEncoderReranker**: Local cross-encoder reranking for cost-free, low-latency scoring
- **SmartReplacer**: LLM-based memory replacement detection for consistent memory updates

**Pluggable Reranking Architecture:**
The search pipeline supports pluggable reranking via `RERANKER_ENGINE`:
- `llm`: Use LLMReranker (API-based, highest quality)
- `cross_encoder`: Use CrossEncoderReranker (local, no API costs)
- `none`: Skip reranking (fastest, use fusion scores only)

**Smart Memory Replacement:**
The add operation supports intelligent replacement detection via `ENABLE_SMART_REPLACE`:
- When enabled (default), checks if new memories replace existing ones
- Uses LLM with structured JSON output for reliable decision-making
- Graceful degradation: adds normally if LLM call fails

**Tantivy Soft-Delete Architecture:**
Delete operations use O(1) tombstone marking instead of O(n) index rebuild:
- Documents marked with `is_deleted=1` and `deleted_at=<timestamp>`
- Search automatically filters tombstoned documents
- Periodic `compact()` removes tombstones when thresholds exceeded
- Configurable via `TANTIVY_SOFT_DELETE_ENABLED` (default: true)

**Embedding Caching:**
Query embeddings are cached to reduce API calls and latency:
- LRU cache with configurable size (default: 100 entries)
- MD5 hash of query text as cache key
- Thread-safe with hit/miss statistics
- Configurable via `EMBEDDING_CACHE_ENABLED` (default: true)

All engines follow Pydantic BaseModel patterns for consistency and type safety. Data persists in `indexes/{project_id}/usearch/` and `indexes/{project_id}/tantivy/`.
