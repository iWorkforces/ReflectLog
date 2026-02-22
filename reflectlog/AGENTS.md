# Agent Guidelines for reflectlog/

This document provides guidelines for AI assistants working on the ReflectLogMCP project. ReflectLogMCP is an MCP (Model Context Protocol) server that provides persistent, project-based memory storage for AI agents.

## Project Overview

ReflectLogMCP combines semantic vector search (USearch) with full-text search (Tantivy) to provide intelligent memory retrieval for AI coding agents. The system uses Reciprocal Rank Fusion (RRF) for optimal result ranking and supports both LLM-based and local cross-encoder reranking.

### Core Features

- **Hybrid Search**: Combines semantic similarity (USearch) + exact phrase matching (Tantivy)
- **RRF Fusion**: Reciprocal Rank Fusion for optimal result ranking
- **Pluggable Reranking**: LLM-based or local cross-encoder relevance scoring
- **Temporal-Aware Scoring**: Recency decay for handling contradictory memories
- **Smart Memory Replacement**: LLM-based detection of memory updates
- **Multiple Transport Modes**: stdio, HTTP, SSE, streamable-http
- **Lazy Initialization**: Fast startup with on-demand component loading

## Directory Structure

```
reflectlog/
├── core/                      # Protocol definitions and abstractions
│   ├── __init__.py
│   ├── config.py              # Configuration protocols (IServerConfig, ISearchConfig, etc.)
│   ├── config_adapters.py     # Config adapters for protocol-based DI
│   ├── memory.py              # Memory operation protocols (IMemoryStore, IMemoryManager)
│   ├── search.py              # Search engine protocols (ISearchBackend, IFusionAlgorithm)
│   ├── reranking.py           # Reranker protocols (IReranker, IRerankerProvider)
│   ├── tools.py               # Tool registration protocols (ITool, IToolRegistry)
│   └── logging.py             # Logging protocols (ILoggingService, LogLevel)
│
├── application/               # Business logic (depends on core)
│   ├── __init__.py
│   ├── mcp_server.py          # FastMCPServer orchestrator
│   ├── constants.py           # Global constants
│   ├── exceptions.py          # Custom exception hierarchy
│   ├── types.py               # Type definitions
│   ├── memory/                # Memory management
│   │   ├── __init__.py
│   │   ├── manager.py         # MemoryManager (facade)
│   │   ├── engine_factory.py  # EngineFactory for search engine initialization
│   │   ├── search_pipeline.py # SearchPipeline with pluggable stages
│   │   ├── add_pipeline.py    # AddPipeline with pluggable phases
│   │   ├── fusion/            # RRF fusion algorithms
│   │   ├── search_strategies.py  # Original search strategies
│   │   ├── add_phases.py      # Original add phases
│   │   └── match_utils.py     # Match utilities
│   ├── tools/                 # MCP tool implementations
│   ├── config/                # Configuration and prompts
│   └── utils/                 # Utilities
│
├── infrastructure/            # Implementations (depend on core)
│   ├── __init__.py
│   ├── search/                # Search engine implementations
│   ├── embeddings/            # Embedding provider implementations
│   ├── reranking/             # Reranker implementations
│   ├── memory/                # Memory storage implementations
│   └── llm/                   # LLM provider implementations
│
├── plugins/                   # Plugin system for extensibility
│   ├── __init__.py
│   ├── discovery.py           # Plugin discovery mechanisms
│   ├── registry.py            # Plugin registry
│   └── loading.py             # Plugin loading and lifecycle
│
├── utility/                   # Platform-specific utilities
└── server.py                  # CLI entry point
```

## Code Style Guidelines

### Python Version and Type Hints

This project requires Python 3.14 or later. All code must use native union syntax for type hints:

```python
# Correct (Python 3.14+)
def process(query: str | None) -> list[str] | None:
    pass

# Incorrect (legacy syntax)
from typing import Optional, List
def process(query: Optional[str]) -> Optional[List[str]]:
    pass
```

### Imports

Use absolute imports with explicit module paths:

```python
from reflectlog.application.config import Config
from reflectlog.infrastructure import USearchEngine
```

Group imports in the following order with blank lines between groups:

1. Standard library imports
2. Third-party imports
3. Local application imports

Use `TYPE_CHECKING` guard for type-only imports:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reflectlog.application.memory import MemoryManager
```

### Type Checking Conventions

The project uses `ty` for static type checking with strict rules enabled. Follow these conventions to avoid type errors:

#### Circular Import Prevention

Use `TYPE_CHECKING` guards for forward references and to prevent circular imports:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reflectlog.core import IStructuredLogger
    from reflectlog.infrastructure import TantivyEngine

    from .manager import MemoryManager
```

#### Logger Type Annotations

Always use `IStructuredLogger | None` for logger parameters in class constructors:

```python
class MyClass:
    def __init__(
        self,
        config: Config,
        logger: IStructuredLogger | None = None,
    ):
        self._logger = logger
```

#### Engine Type References

Use forward references for engine types to avoid circular imports:

```python
def __init__(
    self,
    tantivy_engine: TantivyEngine | None,
    memory_manager: MemoryManager,
):
    self._tantivy_engine = tantivy_engine
    self._memory_manager = memory_manager
```

#### Running Type Checks

```bash
# Run type checking on production code
./start-type-check.sh

# The configuration is in pyproject.toml under [tool.pyright]
# reportAny is set to "warning" to catch untyped code
```


### Naming Conventions

- **Classes**: PascalCase (e.g., `MemoryManager`, `SearchError`)
- **Functions/Methods**: snake_case (e.g., `get_memories()`, `_init_engine()`)
 **Constants**: UPPER_SNAKE_CASE (e.g., `LOG_ADD_MEMORY_PREVIEW_LIMIT`)
- **Private Members**: Leading underscore (e.g., `_lock`, `_client`)
- **Type Variables**: PascalCase with T prefix (e.g., `TResult`, `TConfig`)

### Docstrings

Use triple single quotes for docstrings:

```python
def search(self, query: str, limit: int = 10) -> list[str]:
    '''Search for memories matching the query.

    Args:
        query: The search query string.
        limit: Maximum number of results to return.

    Returns:
        List of matching memory strings.
    '''
```

## Error Handling

### Exception Hierarchy

Follow the custom exception hierarchy defined in `application/exceptions.py`. Always chain exceptions when re-raising:

```python
from reflectlog.application.exceptions import MemorySearchError

try:
    results = self._engine.search(query)
except SearchEngineError as e:
    raise MemorySearchError(f"Search failed for query: {query}") from e
```

### Logging

Use the structured logging utilities from `application/utils/logging.py`. Always include relevant context:

```python
self.logger.info(
    "Search completed",
    extra={
        "query": query[:100],
        "result_count": len(results),
        "duration_ms": duration,
    }
)
```

Never log sensitive information such as API keys or memory content.

## Async Code Guidelines

### Asyncio Mode

The project uses `asyncio_mode = auto` in pytest. When writing async code:

- Use `async def` for asynchronous functions
- Avoid blocking calls in async context
- Use `anyio` for cross-platform async operations

### Lazy Initialization

For expensive resources such as embedders and rerankers, use lazy initialization:

```python
@property
def reranker(self) -> LLMReranker:
    '''Get reranker with lazy initialization.'''
    if self._reranker is None:
        with self._lock:
            if self._reranker is None:
                self._reranker = LLMReranker(self.config)
    return self._reranker
```

## Thread Safety

### Lock Hierarchy

Follow the lock hierarchy documented in `MemoryManager`. The `_write_lock` must be acquired before `_lock` to prevent deadlocks:

```python
# Correct lock acquisition order
with self._write_lock:
    with self._lock:
        # Critical section
        pass

# Incorrect - may cause deadlock
with self._lock:
    with self._write_lock:
        pass
```

### USearch Thread Safety

USearch is not thread-safe. Serialize all write operations using `_write_lock`:

```python
def add_memories(self, memories: list[str]) -> int:
    '''Add memories to the memory store.'''
    with self._write_lock:
        # All USearch operations must be under write lock
        for memory in memories:
            self._engine.add(memory)
```

Use `RLock` for methods that may call other protected methods.

## Testing Requirements

### Test Organization

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use pytest markers: `@pytest.mark.unit` and `@pytest.mark.integration`

### Mocking

Mock external services such as LLM APIs and embedding services in unit tests:

```python
@pytest.fixture
def mock_llm_provider(self):
    provider = MagicMock(spec=IRerankerProvider)
    provider.rerank.return_value = [("result", 0.9)]
    return provider
```

## Build and Test Commands

### Installation

```bash
uv sync
```

### Type Checking

```bash
./start-type-check.sh
```

### Linting

```bash
./start-lint.sh --all        # Check, fix, and format
./start-lint.sh --check      # Check only
./start-lint.sh --fix        # Fix issues automatically
```

### Testing

```bash
./start-unittest.sh                      # Run all tests
./start-unittest.sh --coverage           # With coverage report
./start-unittest.sh --parallel           # Parallel execution
./start-unittest.sh --file tests/unit/application/test_memory_manager.py  # Single file
./start-unittest.sh --pattern test_add   # Tests matching pattern
```

### Running the Server

```bash
uv run reflectlog --transport http --port 9103
```

## Configuration

### Environment Variables

Key configuration variables are documented in `application/config/settings.py`. Always validate configuration at startup using the `ConfigurationValidator`.

### Configuration Patterns

- Use dataclasses with `field(default_factory=...)` for mutable defaults
- Provide sensible defaults for all optional configuration
- Validate configuration in `__post_init__` methods

## Key Abstractions

### MemoryManager

The core class that orchestrates memory operations. Located in `application/memory/manager.py`, it provides:

 `add_memories_async()`: Store memories with semantic embeddings
 `get_all()`: Retrieve all stored memories
- `search()`: Hybrid semantic + full-text search
 `delete_by_memory()`: Remove memories by exact match

### Search Pipeline

The search operation follows a 4-step pipeline:

1. **Parallel Search**: USearch + Tantivy execute concurrently
2. **RRF Fusion**: Results combined using Reciprocal Rank Fusion
3. **Fusion Filter**: Low-confidence matches filtered by threshold
4. **Rerank**: LLM or cross-encoder relevance scoring

### Add Pipeline

The add operation follows a 3-phase architecture:

1. **Phase 1**: Parallel duplicate detection (batch + storage)
2. **Phase 2**: Parallel smart replacement detection (LLM-based)
3. **Phase 3**: Sequential database writes (SQLite constraint)

## Common Patterns

### Context Managers

Use context managers for resource management:

```python
with self._tantivy_engine.writer() as writer:
    writer.delete_document(doc_id)
```

### Protocol-Based Design

Define protocols for abstractions to enable swapping implementations:

```python
class ISemanticSearchEngine(Protocol):
    def search(self, query: str, limit: int) -> list[tuple[str, float, str]]:
        ...
```

### Factory Functions

Use factory functions for object creation:

```python
def create_memory_manager(config: Config, logger: StructuredLogger) -> MemoryManager:
    return MemoryManager(config, logger)
```

## Important Notes

### Data Persistence

- USearch stores data in `indexes/{project_id}/usearch/`
- Tantivy stores data in `indexes/{project_id}/tantivy/`
- Both persist data across server restarts

### Source of Truth

- `get_all()` returns data from USearchEngine (semantic backend)
- Both engines must be kept in sync for consistency

### Performance Considerations

- Lazy initialization reduces startup time by 500-2000ms
- Phased parallel add provides 5-8x speedup for bulk operations
- Query embedding cache reduces API calls for repeated queries
