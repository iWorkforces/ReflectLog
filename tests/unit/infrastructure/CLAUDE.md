# tests/unit/infrastructure/

This directory contains unit tests for the infrastructure layer components.

## Structure

```
infrastructure/
├── test_cached_embeddings.py      # CachedEmbeddings (LRU cache) tests
├── test_cross_encoder_reranker.py # CrossEncoderReranker (FlagReranker) tests
├── test_message_store.py          # MessageStore (libSQL) tests
├── test_qwen3_embedding.py        # LangchainQwenEmbeddings tests
├── test_smart_replacer.py         # SmartReplacer (LLM replacement) tests
├── test_tantivy_engine.py         # TantivyEngine tests (includes soft-delete)
└── test_usearch_engine.py         # USearchEngine tests
```

## Purpose

Test the `ccmemories/infrastructure/` module in isolation:
- `CachedEmbeddings` class for LRU query embedding caching
- `CrossEncoderReranker` class for local FlagReranker-based reranking
- `SmartReplacer` class for LLM-based memory replacement detection
- `USearchEngine` class for semantic vector search
- `TantivyEngine` class for full-text search (includes soft-delete/tombstone tests)
- `MessageStore` class for libSQL storage
- `LangchainQwenEmbeddings` class for embedding generation
- Index initialization, persistence, and recovery
- Error handling and edge cases

## Current Test Files

| File | Tests For |
|------|-----------|
| `test_cached_embeddings.py` | `ccmemories/infrastructure/cached_embeddings.py` |
| `test_cross_encoder_reranker.py` | `ccmemories/infrastructure/cross_encoder_reranker.py` |
| `test_smart_replacer.py` | `ccmemories/infrastructure/smart_replacer.py` |
| `test_usearch_engine.py` | `ccmemories/infrastructure/usearch_engine.py` |
| `test_tantivy_engine.py` | `ccmemories/infrastructure/tantivy_engine.py` |
| `test_message_store.py` | `ccmemories/infrastructure/message_store.py` |
| `test_qwen3_embedding.py` | `ccmemories/infrastructure/qwen3_embedding.py` |

## Test Scenarios

### CachedEmbeddings Tests (`test_cached_embeddings.py`)

```python
class TestCachedEmbeddingsInitialization:
    """Tests for CachedEmbeddings.__init__()"""
    # - Initialization with embedder and cache_size
    # - Cache disabled mode (enabled=False)
    # - Logger assignment

class TestCachedEmbeddingsQueryCaching:
    """Tests for embed_query() and aembed_query() caching"""
    # - Cache miss: calls underlying embedder
    # - Cache hit: returns cached result without calling embedder
    # - MD5 hash collision handling
    # - LRU eviction when cache is full
    # - Disabled cache bypasses caching

class TestCachedEmbeddingsDocumentPassthrough:
    """Tests for embed_documents() and aembed_documents()"""
    # - Documents are NOT cached (pass-through)
    # - Calls underlying embedder directly
    # - No cache statistics updated

class TestCachedEmbeddingsStats:
    """Tests for get_cache_stats() and clear_cache()"""
    # - Hit count increments on cache hit
    # - Miss count increments on cache miss
    # - Clear removes all cached entries
    # - Stats reset on clear
```

### CrossEncoderReranker Tests (`test_cross_encoder_reranker.py`)

```python
class TestCrossEncoderConfig:
    """Tests for CrossEncoderConfig dataclass"""
    # - Default configuration values
    # - from_app_config factory method
    # - Configuration with custom values

class TestCrossEncoderRerankerInitialization:
    """Tests for CrossEncoderReranker.__init__()"""
    # - Lazy model loading (model not loaded until first use)
    # - Thread-safe initialization with _init_lock
    # - Configuration validation

class TestCrossEncoderRerankerRerank:
    """Tests for rerank() method"""
    # - Rerank candidates with FlagReranker scoring
    # - Score normalization (when normalize=True)
    # - Score threshold filtering
    # - Top-k limiting
    # - Empty candidates list
    # - Disabled reranker (pass-through)
    # - Single candidate (compute_score returns float)

class TestCrossEncoderRerankerAsync:
    """Tests for rerank_async() method"""
    # - Async wrapper using asyncify
    # - Non-blocking behavior in event loop
```

### SmartReplacer Tests (`test_smart_replacer.py`)

```python
class TestReplacementDecision:
    """Tests for ReplacementDecision Pydantic model"""
    # - Valid decision parsing
    # - Invalid JSON handling
    # - Confidence range validation

class TestSmartReplacerConfig:
    """Tests for SmartReplacerConfig dataclass"""
    # - Default configuration values
    # - from_app_config factory method
    # - Configuration with custom values

class TestSmartReplacerInitialization:
    """Tests for SmartReplacer.__init__()"""
    # - Lazy client initialization
    # - Configuration storage
    # - Logger assignment

class TestCheckReplacement:
    """Tests for check_replacement() method"""
    # - Replacement detected (should_replace=True, high confidence)
    # - No replacement needed (should_replace=False)
    # - Confidence below threshold (returns False)
    # - LLM returns invalid JSON (graceful degradation)
    # - API error (graceful degradation)
    # - Empty response handling

class TestStructuredOutputFallback:
    """Tests for structured output and fallback behavior"""
    # - Structured JSON output parsing
    # - Fallback to regex extraction on parse error
    # - Complete failure returns safe defaults

class TestSmartReplacerIntegration:
    """Integration tests with mocked LLM"""
    # - Full replacement flow with mock API
    # - Multiple replacement checks
    # - Error recovery
```

### USearchEngine Tests (`test_usearch_engine.py`)

```python
class TestUSearchEngineInitialization:
    """Tests for USearchEngine.__init__()"""
    # - Successful initialization with config
    # - Index creation for new project
    # - Index loading for existing project
    # - Embedder configuration

class TestUSearchEngineAdd:
    """Tests for add() method"""
    # - Add single message
    # - Add multiple messages
    # - Duplicate handling
    # - Empty message handling

class TestUSearchEngineSearch:
    """Tests for search() method"""
    # - Semantic similarity search
    # - Score threshold filtering
    # - Result limit handling
    # - Empty index handling

class TestUSearchEngineGetAll:
    """Tests for get_all() method"""
    # - Empty index returns empty list
    # - Returns all stored messages
    # - Message ordering

class TestUSearchEngineDelete:
    """Tests for delete_by_id() method"""
    # - Delete existing message
    # - Delete non-existent message
    # - Index consistency after delete
```

### TantivyEngine Tests (`test_tantivy_engine.py`)

```python
class TestTantivyEngineInitialization:
    """Tests for TantivyEngine.__init__()"""
    # - Index creation with correct schema
    # - Index loading from existing directory
    # - Schema validation (project_id, message fields)
    # - V2 schema detection (is_deleted, deleted_at fields)

class TestTantivyEngineAdd:
    """Tests for add() method"""
    # - Add single document
    # - Add multiple documents
    # - Commit and sync behavior

class TestTantivyEngineSearch:
    """Tests for search() method"""
    # - Full-text search with stemming
    # - Exact phrase matching
    # - Project ID filtering
    # - Score ranking
    # - Tombstone filtering (soft-deleted docs excluded)

class TestTantivyEngineExactMatch:
    """Tests for exact_match_exists() method"""
    # - Exact match found
    # - No exact match
    # - Case sensitivity

class TestTantivySoftDelete:
    """Tests for soft-delete functionality (7 tests)"""
    # - soft_delete() marks document with tombstone
    # - is_deleted=1 and deleted_at=timestamp set
    # - Tombstoned docs excluded from search results
    # - Tombstoned docs excluded from get_all()
    # - delete() uses soft-delete when enabled
    # - delete() falls back to hard delete when disabled
    # - Edge cases: delete non-existent doc, empty index

class TestTantivyCompaction:
    """Tests for compaction functionality (7 tests)"""
    # - get_tombstone_stats() returns correct counts
    # - needs_compaction() threshold checks (ratio and count)
    # - compact() rebuilds index without tombstones
    # - compact(force=True) ignores thresholds
    # - Compaction preserves live documents
    # - Compaction resets tombstone count
    # - Empty index compaction handling

class TestTantivySchemaMigration:
    """Tests for V1 to V2 schema migration (3 tests)"""
    # - migrate_to_v2() exports and re-imports all docs
    # - schema_version property returns correct version
    # - No-op when already on V2 schema

class TestTantivyTombstoneHelpers:
    """Tests for tombstone helper methods (4 tests)"""
    # - _get_tombstoned_messages() returns correct set
    # - Helper handles empty index
    # - Helper handles no tombstones
    # - Edge cases: deleted_at boundaries
```

### MessageStore Tests (`test_message_store.py`)

```python
class TestMessageStoreInitialization:
    """Tests for MessageStore.__init__()"""
    # - Database creation
    # - Table schema validation
    # - Connection handling

class TestMessageStoreOperations:
    """Tests for CRUD operations"""
    # - Insert message with ID
    # - Get message by ID
    # - Get all messages
    # - Delete message by ID
    # - Update message
```

### LangchainQwenEmbeddings Tests (`test_qwen3_embedding.py`)

```python
class TestQwenEmbeddingsInitialization:
    """Tests for LangchainQwenEmbeddings.__init__()"""
    # - Configuration from environment
    # - HTTP/2 client setup
    # - Concurrency limiter setup

class TestQwenEmbeddingsGeneration:
    """Tests for embed_documents() and embed_query()"""
    # - Single text embedding
    # - Batch text embeddings
    # - Embedding dimensions validation
    # - Error handling for API failures
```

## Mocking Patterns

### Mock External Dependencies

```python
from unittest.mock import Mock, patch, AsyncMock

# Mock FlagReranker for cross-encoder tests
@patch('ccmemories.infrastructure.cross_encoder_reranker.FlagReranker')
def test_cross_encoder_reranker(mock_flag_reranker_class):
    mock_model = Mock()
    mock_model.compute_score.return_value = [0.9, 0.7, 0.3]
    mock_flag_reranker_class.return_value = mock_model
    # Test code

# Mock USearch index
@patch('ccmemories.infrastructure.usearch_engine.Index')
def test_usearch_initialization(mock_index_class):
    mock_index = Mock()
    mock_index_class.return_value = mock_index
    # Test code

# Mock Tantivy
@patch('ccmemories.infrastructure.tantivy_engine.tantivy')
def test_tantivy_initialization(mock_tantivy):
    mock_index = Mock()
    mock_tantivy.Index.return_value = mock_index
    # Test code

# Mock libSQL
@patch('ccmemories.infrastructure.message_store.libsql.connect')
def test_message_store(mock_connect):
    mock_conn = Mock()
    mock_connect.return_value = mock_conn
    # Test code

# Mock OpenAI client for embeddings
@patch('ccmemories.infrastructure.qwen3_embedding.OpenAI')
def test_embeddings(mock_openai):
    mock_client = Mock()
    mock_openai.return_value = mock_client
    # Test code

# Mock AsyncOpenAI client for SmartReplacer
@patch('ccmemories.infrastructure.smart_replacer.AsyncOpenAI')
async def test_smart_replacer(mock_async_openai):
    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content='{"should_replace": true, "confidence": 0.9, "reason": "Test"}'))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_async_openai.return_value = mock_client
    # Test code
```

### Mock File System

```python
import tempfile
import shutil

@pytest.fixture
def temp_index_dir():
    """Create temporary directory for index files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_index_persistence(temp_index_dir):
    # Use temp_index_dir for index files
    pass
```

## Running Infrastructure Tests

```bash
# All infrastructure unit tests
uv run pytest tests/unit/infrastructure/ -v

# Specific test file
uv run pytest tests/unit/infrastructure/test_usearch_engine.py -v

# Specific test class
uv run pytest tests/unit/infrastructure/test_usearch_engine.py::TestUSearchEngineSearch -v

# With coverage
uv run pytest tests/unit/infrastructure/ --cov=ccmemories.infrastructure --cov-report=term-missing
```

## Coverage Goals

Aim for **85%+ coverage** of `ccmemories/infrastructure/`:
- All public methods
- Index initialization and persistence
- Error handling paths
- Edge cases (empty index, missing files)

## Best Practices

1. **Mock external I/O**: Always mock file system, network, and database operations
2. **Test error paths**: Verify proper error handling for failures
3. **Test persistence**: Verify index save/load cycle
4. **Use fixtures**: Share common setup across tests
5. **Test boundaries**: Empty inputs, large inputs, invalid inputs
