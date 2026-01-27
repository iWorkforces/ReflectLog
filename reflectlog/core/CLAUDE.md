# reflectlog/core/

This directory contains protocol definitions and abstractions that form the foundation of ReflectLogMCP's architecture. All application and infrastructure components depend on these protocols rather than concrete implementations.

## Structure

```
core/
├── __init__.py           # Package exports and public API
├── config.py              # Configuration protocols (IServerConfig, ISearchConfig, IRerankerConfig, etc.)
├── config_adapters.py     # Config adapters for protocol-based dependency injection
├── memory.py              # Memory operation protocols (IMemoryStore, IMemoryManager, IMemoryBackend)
├── search.py              # Search engine protocols (ISearchBackend, IFusionAlgorithm, ISearchResult)
├── reranking.py           # Reranker protocols (IReranker, IRerankerProvider, IRankingResult)
├── tools.py               # Tool registration protocols (ITool, IToolRegistry, IToolResult)
└── logging.py             # Logging protocols (ILoggingService, ILogSink, LogLevel)
```

## Purpose

The core package provides:

1. **Protocol-Based Design**: Interface contracts that enable runtime component substitution and compile-time type checking
2. **Dependency Injection**: Protocol-based DI allows swapping implementations without modifying consumers
3. **Testability**: Mock implementations can easily satisfy protocol contracts
4. **Type Safety**: Runtime protocol checking with `@runtime_checkable` ensures interface compliance

## Protocol Definitions

### Configuration Protocols (`config.py`)

Defines protocols for abstracting configuration sources:

- **`IServerConfig`**: Server-level configuration (transport, host, port, project_id, log_level)
- **`ISearchConfig`**: Search-related configuration (search_limit, enable_hybrid_search, enable_rrf_fusion, fusion_rrf_k, reranker_engine, etc.)
- **`IRerankerConfig`**: Reranker configuration (search_score_threshold, max_concurrency, enable_recency_boost, recency_decay_rate)
- **`IReplacementConfig`**: Smart replacement configuration (threshold, enabled, max_retries, retry_delay)
- **`IStorageConfig`**: Storage configuration (embedding_dims, metric, usearch_exact_search, tantivy_soft_delete_enabled, etc.)
- **`IEmbedderConfig`**: Embedding provider configuration (model, dims, batch_size, cache_size)
- **`IAppConfig`**: Combined protocol for all configuration (extends all above protocols)

### Configuration Adapters (`config_adapters.py`)

Provides adapter classes that wrap application `Config` and expose specific protocol interfaces:

- **`ConfigAdapter[IAppConfig]`**: Full config adapter satisfying IAppConfig
- **`ServerConfigAdapter`**: Server config only (IServerConfig)
- **`SearchConfigAdapter`**: Search config only (ISearchConfig)
- **`StorageConfigAdapter`**: Storage config only (IStorageConfig)
- **`RerankerConfigAdapter`**: Reranker config only (IRerankerConfig)
- **`ReplacementConfigAdapter`**: Replacement config only (IReplacementConfig)
- **`EmbedderConfigAdapter`**: Embedder config only (IEmbedderConfig)

Factory functions: `create_*_config_adapter()` for creating specific adapters.

### Memory Protocols (`memory.py`)

Defines interfaces for memory storage and management:

- **`IMemoryStore`**: Low-level storage operations (add, get, get_all, delete, update)
- **`IMemoryBackend`**: Search engine backend interface (search, add, delete, get_all)
- **`IMemoryManager`**: High-level memory management interface (add_messages, search, get_all, delete_by_message)

### Search Protocols (`search.py`)

Defines interfaces for search backends and fusion algorithms:

- **`ISearchResult`**: Dataclass representing a single search result (content, score, memory_id, created_at, metadata)
- **`SearchContext`**: Dataclass carrying search operation context (query, limit, overfetch_limit, project_id, etc.)
- **`ISearchBackend`**: Protocol for search engine implementations (name, search, add, delete methods)
- **`IFusionAlgorithm`**: Protocol for result fusion algorithms (method property, fuse method)

### Reranking Protocols (`reranking.py`)

Defines interfaces for reranking components:

- **`IRankingResult`**: Protocol for reranking result (document, score properties)
- **`IReranker`**: Protocol for reranker interface (async rerank method)
- **`IRerankerProvider`**: Protocol for reranker LLM providers (score method)

### Tool Protocols (`tools.py`)

Defines interfaces for MCP tool registration:

- **`ITool`**: Protocol for individual MCP tools (name, description, execute methods)
- **`IToolRegistry`**: Protocol for tool registry management (register, get, get_all methods)
- **`IToolResult`**: Dataclass representing tool execution result (success, data, error properties)

### Logging Protocols (`logging.py`)

Defines interfaces for logging services:

- **`LogLevel`**: Enum for log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **`ILogSink`**: Protocol for log sink implementations (emit method)
- **`ILoggingService`**: Protocol for logging service (debug, info, warning, error, critical methods)

## Key Patterns

### Protocol-Based Dependency Injection

Components depend on protocols, not concrete implementations:

```python
class SearchService:
    def __init__(
        self,
        backend: ISearchBackend,
        config: ISearchConfig,
    ):
        self._backend = backend
        self._config = config

    async def search(self, query: str) -> list[str]:
        limit = self._config.search_limit
        results = await self._backend.search(query, limit)
        return [r.content for r in results]
```

Benefits:
- Runtime component substitution
- Compile-time type checking
- Easy test mocking
- Loose coupling between layers

### Configuration Adapters

Adapters make concrete configuration satisfy protocol contracts:

```python
from reflectlog.core.config_adapters import create_search_config_adapter

# Create adapter from application Config
search_adapter = create_search_config_adapter(app_config)

# Use adapter where ISearchConfig is expected
service = SearchService(
    backend=usearch_engine,
    config=search_adapter,  # Satisfies ISearchConfig
)
```

### Runtime Protocol Checking

Use `@runtime_checkable` to enable runtime verification:

```python
@runtime_checkable
class ISearchBackend(Protocol):
    async def search(self, query: str, limit: int) -> list[ISearchResult]: ...

# Can check at runtime
if isinstance(engine, ISearchBackend):
    results = await engine.search(query, limit)
```

### Dataclasses for Context

Use dataclasses for complex context objects:

```python
@dataclass(frozen=True)
class ISearchResult:
    """Result from a search operation."""
    content: str
    score: float
    memory_id: str
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
```

## Usage Examples

### Creating Protocol-Compliant Components

```python
@runtime_checkable
class CustomSearchBackend:
    """Custom search backend implementation."""

    @property
    def name(self) -> str:
        return "custom_backend"

    async def search(
        self,
        query: str,
        project_id: str,
        limit: int,
    ) -> list[ISearchResult]:
        # Implementation
        results = [...]  # Your search logic
        return results

    async def add(
        self,
        project_id: str,
        documents: list[str],
    ) -> None:
        # Implementation
        pass

    async def delete(
        self,
        document_id: str,
    ) -> None:
        # Implementation
        pass
```

### Using Configuration Adapters

```python
from reflectlog.core.config_adapters import (
    create_storage_config_adapter,
    create_reranker_config_adapter,
)

# Create fine-grained adapters
storage_adapter = create_storage_config_adapter(app_config)
reranker_adapter = create_reranker_config_adapter(app_config)

# Use with protocol-based constructors
memory_manager = MemoryManager(
    storage_config=storage_adapter,
    reranker_config=reranker_adapter,
)
```

### Protocol-Based Testing

```python
from unittest.mock import MagicMock

# Create mock that satisfies protocol
mock_backend = MagicMock(spec=ISearchBackend)
mock_backend.search.return_value = [
    ISearchResult(content="doc1", score=0.9, memory_id="1"),
    ISearchResult(content="doc2", score=0.8, memory_id="2"),
]

# Use in tests
service = SearchService(backend=mock_backend, config=search_adapter)
results = await service.search("test query")
```

## Testing Guidelines

### Unit Testing Protocols

When testing components that use protocols:

1. **Mock protocol implementations** using `unittest.mock.MagicMock(spec=Protocol)`
2. **Test adapter behavior** for configuration injection
3. **Verify protocol compliance** with runtime checks
4. **Test type safety** with ty type checking

### Protocol Verification

```python
# Static type checking (ty)
_: type[ISearchBackend] = CustomSearchBackend

# Runtime verification
if isinstance(custom_engine, ISearchBackend):
    print("Engine implements ISearchBackend protocol")
```

## Dependencies

### Internal Dependencies

- `typing`: Protocol, runtime_checkable, dataclass, field
- No external dependencies - pure Python protocols

### External Dependencies

None - this is a pure Python protocol package

## Important Notes

### Protocol vs Abstract Base Class

Use protocols when:
- Multiple independent implementations exist
- Runtime type checking is needed
- Duck typing is preferred

Use ABCs when:
- Shared implementation is needed
- Method overrides must be enforced
- Class hierarchy is important

### Performance

- Protocol checks have minimal runtime overhead with `@runtime_checkable`
- Type checking happens at compile time (via ty)
- Adapters have negligible overhead (property forwarding)

### Evolution

Protocols can be extended by adding new optional methods:
- Existing implementations automatically satisfy extended protocols if they don't define new methods
- New methods should have sensible defaults in adapters
- Document breaking changes clearly

### Type Safety

All protocols use `@runtime_checkable` for:
- Runtime type checking
- Better error messages
- Debugging support
- IDE autocomplete improvements

## Protocol Catalog

### Complete Protocol List

| Protocol | Purpose | Module |
|----------|---------|---------|
| `IAppConfig` | Combined application configuration | `config.py` |
| `IServerConfig` | Server-level settings | `config.py` |
| `ISearchConfig` | Search configuration | `config.py` |
| `IRerankerConfig` | Reranker settings | `config.py` |
| `IReplacementConfig` | Smart replacement settings | `config.py` |
| `IStorageConfig` | Storage settings | `config.py` |
| `IEmbedderConfig` | Embedding provider settings | `config.py` |
| `IMemoryStore` | Low-level storage operations | `memory.py` |
| `IMemoryBackend` | Search engine backend | `memory.py` |
| `IMemoryManager` | High-level memory management | `memory.py` |
| `ISearchBackend` | Search engine implementation | `search.py` |
| `IFusionAlgorithm` | Result fusion algorithm | `search.py` |
| `IReranker` | Reranker interface | `reranking.py` |
| `IRerankerProvider` | Reranker LLM provider | `reranking.py` |
| `IRankingResult` | Reranking result | `reranking.py` |
| `ITool` | MCP tool interface | `tools.py` |
| `IToolRegistry` | Tool registry | `tools.py` |
| `IToolResult` | Tool execution result | `tools.py` |
| `LogLevel` | Log level enum | `logging.py` |
| `ILogSink` | Log sink implementation | `logging.py` |
| `ILoggingService` | Logging service | `logging.py` |

## Best Practices

### When to Define New Protocols

Define a new protocol when:
1. Multiple implementations will exist
2. Runtime substitution is needed
3. Test mocking is important
4. Type safety at boundaries is required

### Protocol Design Principles

1. **Keep protocols focused**: Single responsibility per protocol
2. **Use descriptive names**: ISearchBackend, not IEngine
3. **Document thoroughly**: Clear docstrings for each method
4. **Provide async methods**: For I/O operations
5. **Use dataclasses**: For result types and context objects
6. **Make optional methods explicit**: Use default values or Optional types

### Adapter Design

1. **Forward properties**: Simple property forwarding from wrapped config
2. **Factory functions**: Use `create_*_adapter()` pattern
3. **Type safety**: Ensure adapters satisfy protocol contracts
4. **Minimal overhead**: No unnecessary computation in adapters

## Resources

- [PEP 544 - Structural Subtyping](https://peps.python.org/pep-0544/)
- [PEP 545 - Type Hinting Generics](https://peps.python.org/pep-0545/)
- [Python Protocols Documentation](https://docs.python.org/3/library/typing.html#typing.Protocol)
- [Protocol-Oriented Programming](https://mypy.readthedocs.io/en/stable/protocols.html)
