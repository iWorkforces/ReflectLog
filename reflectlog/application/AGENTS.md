# Agent Guidelines for reflectlog/application/

This directory contains the business logic layer of ReflectLogMCP. It orchestrates the interaction between MCP tools, memory management, configuration, and utilities.

## Directory Structure

```
application/
├── __init__.py              # Package exports and public API
├── constants.py             # Global constants used across the application
├── exceptions.py            # Custom exception hierarchy
├── mcp_server.py            # FastMCPServer orchestration
├── types.py                 # Type definitions and protocols
├── memory/                  # Memory management system
│   ├── __init__.py
│   ├── manager.py           # MemoryManager (facade)
│   ├── engine_factory.py    # EngineFactory for search engine initialization
│   ├── search_pipeline.py   # SearchPipeline with pluggable stages
│   ├── add_pipeline.py      # AddPipeline with pluggable phases
│   ├── search_strategies.py # Original search strategies (legacy)
│   ├── add_phases.py        # Original add phases (legacy)
│   ├── match_utils.py       # Match utilities
│   ├── protocols.py         # Local protocols
│   ├── fusion/              # RRF fusion algorithms
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── ranx_fusion.py
│   └── reranking/           # Score normalization utilities
│       ├── __init__.py
│       └── normalization.py
├── tools/                   # MCP tool implementations
│   ├── __init__.py
│   ├── base.py
│   ├── add.py
│   ├── get_all.py
│   ├── search.py
│   ├── remove.py
│   └── health_check.py
├── config/                  # Configuration and prompt management
│   ├── __init__.py
│   ├── settings.py
│   ├── prompts.py
│   └── validation.py
└── utils/                   # Utility functions and helpers
    ├── __init__.py
    ├── logging.py
    ├── metrics.py
    ├── security.py
    ├── validation.py
    ├── retry.py
    ├── circuit_breaker.py
    └── numba_utils.py
```

## Core Responsibilities

### MCP Server Orchestration

The `mcp_server.py` module provides the main entry point for the MCP server:

- Initializes FastMCPServer with all tools
- Manages MemoryManager lifecycle
- Handles tool registration and routing
- Coordinates configuration loading

### Memory Management

The `memory/` submodule implements the hybrid search engine:

- Combines USearch (semantic) and Tantivy (full-text) search
- Implements RRF (Reciprocal Rank Fusion) for result ranking
- Provides smart memory replacement detection
- Manages add, search, and delete operations

### Tools

The `tools/` submodule implements the external MCP interface:

- **add.py**: AddTool for storing messages
- **get_all.py**: GetAllTool for retrieving all messages
- **search.py**: SearchTool for hybrid searching
- **remove.py**: RemoveTool for deleting messages
- **health_check.py**: HealthCheckTool for server status

### Configuration

The `config/` submodule manages application settings:

- Loads environment variables into Config dataclass
- Provides LLM prompt templates
- Validates configuration at startup
- Manages transport mode settings

## Key Patterns

### Dependency Injection

Components receive dependencies through constructor injection:

```python
class MemoryManager:
    def __init__(
        self,
        config: Config,
        logger: StructuredLogger,
        usearch_engine: USearchEngine | None = None,
        tantivy_engine: TantivyEngine | None = None,
    ):
        self.config = config
        self.logger = logger
        self._semantic_engine = usearch_engine
        self._fulltext_engine = tantivy_engine
```

### Protocol-Based Abstractions

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

For expensive resources, use lazy initialization:

```python
@property
def reranker(self) -> LLMReranker | None:
    '''Get LLM reranker with lazy initialization.'''
    if self._reranker is None and self.config.reranker_engine == "llm":
        with self._reranker_lock:
            if self._reranker is None:
                self._reranker = LLMReranker(self.config, self.logger)
    return self._reranker
```

## Error Handling

### Exception Hierarchy

Follow the custom exception hierarchy defined in `exceptions.py`:

```
ReflectLogError (base)
├── ConfigurationError
├── MemorySearchError
├── MemoryStorageError
├── DuplicateDetectionError
└── SmartReplacementError
```

### Structured Logging

Use the StructuredLogger from `utils/logging.py`:

```python
self.logger.info(
    "Operation completed",
    extra={
        "operation": "search",
        "query_length": len(query),
        "result_count": len(results),
        "duration_ms": duration,
    }
)
```

## Testing Guidelines

### Unit Tests

- Mock external dependencies (USearch, Tantivy, LLM APIs)
- Test individual components in isolation
- Verify error handling paths
- Use pytest fixtures for common setup

### Integration Tests

- Test with real search engine instances
- Verify cross-component interactions
- Test configuration loading
- Validate persistence behavior

## Common Operations

### Adding Messages

The add operation uses a 3-phase pipeline:

```python
# Phase 1: Parallel duplicate detection
duplicates = await detect_duplicates(messages)

# Phase 2: Parallel smart replacement
replacements = await detect_replacements(new_messages)

# Phase 3: Sequential storage
for message in filtered_messages:
    usearch_engine.add(message)
    tantivy_engine.add(message)
```

### Searching

The search operation follows a 4-step pipeline:

```python
# Step 1: Parallel search
usearch_results = usearch_engine.search(query, limit=overfetch)
tantivy_results = tantivy_engine.search(query, limit=overfetch)

# Step 2: RRF Fusion
fused_results = ranx_fusion.fuse(usearch_results, tantivy_results)

# Step 3: Fusion threshold filter
filtered_results = [r for r in fused_results if r.score >= threshold]

# Step 4: Reranking
reranked_results = reranker.rerank(query, filtered_results)
```

## Configuration

### Config Dataclass

The `Config` class from `config/settings.py` provides centralized configuration:

```python
@dataclass
class Config:
    project_id: str
    openrouter_api_key: str | None = None
    transport: str = "stdio"
    port: int = 9103
    llm_model: str = "x-ai/grok-4.1-fast"
    embedding_model: str = "openai/text-embedding-3-large"
    # ... more fields
```

### Environment Variables

Key configuration through environment variables:

- `PROJECT_ID`: Unique project identifier
- `OPENROUTER_API_KEY`: OpenRouter API key for LLM/embeddings
- `MCP_TRANSPORT`: Transport mode (stdio, http, sse, streamable-http)
- `ENABLE_HYBRID_SEARCH`: Enable Tantivy full-text search
- `RERANKER_ENGINE`: Reranking engine (llm, cross_encoder, none)

## Dependencies

### Internal Dependencies

- `infrastructure/`: Search engines, rerankers, embedders
- `utility/`: Platform-specific utilities

### External Dependencies

- `fastmcp`: MCP server framework
- `pydantic`: Configuration and validation
- `usearch`: Vector search engine
- `tantivy`: Full-text search engine
- `ranx`: RRF fusion library
- `structlog`: Structured logging

## Important Notes

### Thread Safety

- USearch is not thread-safe; use `_write_lock` for writes
- Follow lock hierarchy: `_write_lock` before `_lock`
- Use RLock for methods calling other protected methods

### Performance

- Lazy initialization reduces startup time
- Phased parallel add provides 5-8x speedup
- Query embedding cache reduces API calls
- Adaptive overfetch adjusts based on index size

### Data Consistency

- USearchEngine is source of truth for `get_all()`
- Both engines must be kept in sync
- Use transactions where supported
