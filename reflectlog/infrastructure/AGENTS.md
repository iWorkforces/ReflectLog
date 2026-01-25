# Agent Guidelines for reflectlog/infrastructure/

This directory contains external integrations and infrastructure components for ReflectLogMCP. It wraps third-party libraries (USearch, Tantivy, LLM providers) with a consistent interface for use by the application layer.

## Directory Structure

```
infrastructure/
├── __init__.py                 # Package exports and public API (re-exports all)
├── search/                     # Search engine implementations
│   ├── __init__.py            # Re-exports from parent
│   ├── base.py                 # SearchEngineBase class
│   ├── usearch_engine.py       # USearch vector search wrapper
│   └── tantivy_engine.py       # Tantivy full-text search wrapper
├── embeddings/                 # Embedding provider implementations
│   ├── __init__.py            # Re-exports from parent
│   ├── langchain_qwen.py       # Qwen3 embedding implementation
│   └── cached.py               # LRU cache for query embeddings
├── reranking/                  # Reranker implementations
│   ├── __init__.py            # Re-exports from parent
│   ├── llm_reranker.py        # LLM-based relevance scoring
│   └── cross_encoder.py        # Local cross-encoder reranking
├── memory/                     # Memory storage implementations
│   ├── __init__.py            # Re-exports from parent
│   ├── message_store.py        # libSQL-based message storage
│   └── smart_replacer.py       # LLM-based memory replacement detection
├── llm/                        # LLM provider implementations
│   ├── __init__.py            # Re-exports from parent
│   └── base.py                 # Base provider class and protocols
├── usearch_engine.py           # Legacy - kept for backward compatibility
├── tantivy_engine.py           # Legacy - kept for backward compatibility
├── llm_reranker.py             # Legacy - kept for backward compatibility
├── llm_provider_base.py        # Legacy - kept for backward compatibility
├── cross_encoder_reranker.py   # Legacy - kept for backward compatibility
├── smart_replacer.py           # Legacy - kept for backward compatibility
├── qwen3_embedding.py          # Legacy - kept for backward compatibility
├── cached_embeddings.py        # Legacy - kept for backward compatibility
└── message_store.py            # Legacy - kept for backward compatibility
```

## Core Responsibilities

### Search Engine Wrappers

The infrastructure layer provides clean abstractions over third-party search engines:

- **USearchEngine**: Wraps USearch HNSW vector search with libSQL message storage
- **TantivyEngine**: Wraps Tantivy full-text search with soft-delete support

### Reranking Components

- **LLMReranker**: LLM-based relevance scoring with provider abstraction
- **CrossEncoderReranker**: Local reranking using BAAI/bge-reranker-v2-m3 model

### Support Components

- **SmartReplacer**: Detects when new memories update existing ones
- **MessageStore**: libSQL-based storage for message text and metadata
- **CachedEmbeddings**: LRU cache for query embeddings to reduce API calls

## Key Components

### USearchEngine

Wraps the USearch vector search library:

```python
class USearchEngine:
    '''USearch vector search with libSQL message storage.'''

    def __init__(
        self,
        config: USearchConfig,
        embedder: Embeddings,
        logger: StructuredLogger,
    ):
        self.config = config
        self.embedder = embedder
        self.logger = logger
        self._index: usearch.Index | None = None
        self._conn: sqlite3.Connection | None = None

    def search(self, query: str, limit: int) -> list[tuple[str, float, str]]:
        '''Search for similar documents.'''
        query_vector = self.embedder.embed_query(query)
        results = self._index.search(query_vector, limit)
        return self._enrich_results(results)

    def add(self, message: str) -> None:
        '''Add a document to the index.'''
        vector = self.embedder.embed_query(message)
        self._index.add(vector)
        self._store_message(message)

    def get_all(self) -> list[str]:
        '''Get all stored messages.'''
        cursor = self._conn.execute("SELECT message FROM messages WHERE is_deleted = 0")
        return [row[0] for row in cursor.fetchall()]
```

### TantivyEngine

Wraps the Tantivy full-text search engine:

```python
class TantivyEngine:
    '''Tantivy full-text search with soft-delete support.'''

    def __init__(
        self,
        config: TantivyConfig,
        logger: StructuredLogger,
    ):
        self.config = config
        self.logger = logger
        self._index: tantivy.Index | None = None

    def search(self, query: str, limit: int) -> list[tuple[str, float, str]]:
        '''Search for documents matching query.'''
        searcher = self._index.searcher()
        parser = tantivy.QueryParser.for_index(self._index)
        parsed_query = parser.parse(query)

        results = searcher.search(parsed_query, limit=limit)
        return [(hit["message"], hit.score, hit["id"]) for hit in results]

    def add(self, message: str, doc_id: str) -> None:
        '''Add a document to the index.'''
        writer = self._index.writer()
        writer.add_document({
            "message": message,
            "project_id": self.config.project_id,
            "id": doc_id,
            "is_deleted": False,
        })
        writer.commit()

    def delete(self, doc_id: str) -> None:
        '''Soft-delete a document.'''
        writer = self._index.writer()
        writer.delete_term("id", doc_id)
        writer.commit()
```

### LLMReranker

Provides LLM-based relevance scoring:

```python
class LLMReranker:
    '''LLM-based relevance scoring with provider abstraction.'''

    def __init__(
        self,
        config: LLMRerankerConfig,
        logger: StructuredLogger,
    ):
        self.config = config
        self.logger = logger
        self._provider: IRerankerProvider | None = None

    async def rerank(
        self,
        query: str,
        documents: list[str],
        timestamp_map: dict[str, str] | None = None,
    ) -> list[tuple[str, float]]:
        '''Score documents by relevance to query using LLM.'''
        if timestamp_map:
            prompt = SCORING_PROMPT_WITH_AGE
        else:
            prompt = SCORING_PROMPT

        # Batch score for efficiency
        scored = await self._provider.score_batch(query, documents, prompt)

        return [(doc, score) for doc, score in scored]
```

### MessageStore

libSQL-based storage for message persistence:

```python
class MessageStore:
    '''libSQL-based message storage with archiving support.'''

    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        '''Initialize database schema.'''
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                message TEXT NOT NULL,
                vector BLOB,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                is_deleted BOOLEAN DEFAULT FALSE,
                deleted_at TEXT
            )
        """)
        self._conn.commit()
```

### CachedEmbeddings

LRU cache for query embeddings:

```python
class CachedEmbeddings:
    '''LRU cache for query embeddings.'''

    def __init__(
        self,
        embedder: Embeddings,
        cache_size: int = 100,
    ):
        self._embedder = embedder
        self._cache: dict[str, list[float]] = {}
        self._lru: collections.OrderedDict[str, None] = collections.OrderedDict()
        self._cache_size = cache_size
        self._hits = 0
        self._misses = 0

    def embed_query(self, query: str) -> list[float]:
        '''Embed query with LRU caching.'''
        cache_key = hashlib.md5(query.encode()).hexdigest()

        if cache_key in self._cache:
            self._hits += 1
            return self._cache[cache_key]

        self._misses += 1
        embedding = self._embedder.embed_query(query)

        # Add to cache with LRU eviction
        if len(self._cache) >= self._cache_size:
            oldest = self._lru.popitem(last=False)
            del self._cache[oldest]

        self._cache[cache_key] = embedding
        self._lru[cache_key] = None

        return embedding
```

## Key Patterns

### Configuration from App Config

Use factory methods to create infrastructure components from application configuration:

```python
@classmethod
def from_app_config(cls, config: Config, logger: StructuredLogger) -> "USearchEngine":
    '''Create USearchEngine from application Config.'''
    usearch_config = USearchConfig(
        index_path=config.usearch_index_path_template.format(
            project_id=config.project_id
        ),
        metric="cosine",
        dimensions=config.embedding_dims,
    )
    embedder = LangchainQwenEmbeddings(
        model="qwen3-embedding",
        embedding_dims=config.embedding_dims,
    )
    return cls(usearch_config, embedder, logger)
```

### Protocol-Based Provider Abstraction

Abstract LLM providers for flexibility:

```python
class IRerankerProvider(Protocol):
    '''Protocol for reranker providers.'''

    async def score(
        self,
        query: str,
        document: str,
        prompt: str,
    ) -> float:
        '''Score a single document.'''
        ...

    async def score_batch(
        self,
        query: str,
        documents: list[str],
        prompt: str,
    ) -> list[float]:
        '''Score multiple documents efficiently.'''
        ...
```

### Lazy Initialization

Defer expensive resource initialization:

```python
@property
def _index(self) -> usearch.Index:
    '''Lazy load USearch index.'''
    if self._index_internal is None:
        self._index_internal = usearch.Index(
            ndim=self.config.dimensions,
            metric=self.config.metric,
            path=self.config.index_path,
        )
    return self._index_internal
```

## Error Handling

### Wrapper Exceptions

Wrap third-party exceptions with domain-specific types:

```python
try:
    self._index.search(query_vector, limit)
except usearch.SearchError as e:
    raise VectorSearchError(f"USearch search failed: {e}") from e
except usearch.IOError as e:
    raise VectorSearchError(f"USearch I/O error: {e}") from e
```

### Graceful Degradation

Provide fallback behavior when possible:

```python
async def rerank(
    self,
    query: str,
    documents: list[str],
    timestamp_map: dict[str, str] | None = None,
) -> list[tuple[str, float]]:
    try:
        return await self._llm_rerank(query, documents, timestamp_map)
    except LLMError as e:
        self.logger.warning(
            "LLM reranking failed, using fusion scores",
            extra={"error": str(e)}
        )
        # Return fusion scores as fallback
        return [(doc, 0.5) for doc in documents]
```

## Testing Guidelines

### Unit Tests

- Mock external API calls (LLM providers)
- Test with temporary indices
- Verify error handling paths
- Test cache behavior

### Integration Tests

- Test with real USearch/Tantivy indices
- Verify persistence behavior
- Test with real LLM API (if configured)
- Benchmark performance

### Mock Patterns

```python
@pytest.fixture
def mock_llm_provider():
    provider = MagicMock(spec=IRerankerProvider)
    provider.score.return_value = 0.9
    provider.score_batch.return_value = [0.9, 0.8, 0.7]
    return provider

@pytest.fixture
def temp_usearch_index(tmp_path):
    index_path = tmp_path / "test.usearch"
    return usearch.Index(ndim=384, path=str(index_path))
```

## Dependencies

### Internal Dependencies

- `application/config/`: Configuration dataclasses
- `application/utils/logging.py`: StructuredLogger
- `application/exceptions.py`: Exception classes

### External Dependencies

- `usearch`: Vector search library
- `tantivy`: Full-text search engine
- `langchain`: Embeddings interface
- `libsql`: SQLite-compatible database
- `openai` / `anthropic`: LLM providers
- `sentence-transformers`: Cross-encoder models

## Important Notes

### Thread Safety

- USearch is not thread-safe; serialize writes
- Tantivy handles concurrent reads
- MessageStore (libSQL) is thread-safe

### Performance

- Lazy initialization reduces startup time
- Embedding cache reduces API calls
- Batch processing for efficiency

### Resource Management

- Close connections on shutdown
- Use context managers where possible
- Monitor memory usage for large indices
