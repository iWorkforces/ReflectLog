# Agent Guidelines for reflectlog/core/

This directory contains protocol definitions and abstractions that define the interface contracts for the application layer. All components depend on these protocols rather than concrete implementations, enabling runtime component substitution, compile-time type checking, dependency injection, and testability through mock implementations.

## Directory Structure

```
core/
├── __init__.py          # Package exports and public API
├── config.py            # Configuration protocols (IServerConfig, ISearchConfig, etc.)
├── config_adapters.py   # Config adapters for protocol-based dependency injection
├── memory.py            # Memory operation protocols (IMemoryStore, IMemoryManager)
├── search.py            # Search engine protocols (ISearchBackend, IFusionAlgorithm)
├── reranking.py         # Reranker protocols (IReranker, IRerankerProvider)
├── tools.py             # Tool registration protocols (ITool, IToolRegistry)
└── logging.py           # Logging protocols (ILoggingService, LogLevel)
```

## Core Responsibilities

### Configuration Protocols (config.py)

The `config.py` module defines protocols for abstracting configuration sources:

```python
@runtime_checkable
class IServerConfig(Protocol):
    """Protocol for server-level configuration."""
    @property
    def transport(self) -> Literal["stdio", "http", "sse", "streamable-http"]: ...
    @property
    def host(self) -> str: ...
    @property
    def port(self) -> int: ...
    @property
    def project_id(self) -> str: ...
    @property
    def log_level(self) -> str: ...

@runtime_checkable
class ISearchConfig(Protocol):
    """Protocol for search-related configuration."""
    @property
    def search_limit(self) -> int: ...
    @property
    def enable_hybrid_search(self) -> bool: ...
    @property
    def enable_rrf_fusion(self) -> bool: ...
    @property
    def fusion_rrf_k(self) -> int: ...
    @property
    def reranker_engine(self) -> Literal["llm", "cross_encoder", "none"]: ...

# Additional protocols:
# - IStorageConfig: Storage path and embedding configuration
# - IRerankerConfig: LLM and cross-encoder settings
# - IEmbedderConfig: Embedding provider configuration
# - IReplacementConfig: Smart replacement settings
# - IAppConfig: Combined protocol for all configuration
```

### Configuration Adapters (config_adapters.py)

The `config_adapters.py` module provides adapter classes that wrap application configuration and expose specific protocol interfaces:

```python
class ConfigAdapter[IAppConfig]:
    """Adapter that makes Config satisfy IAppConfig protocol."""

    def __init__(self, config: "Config") -> None:
        self._config = config

    # Forwards all IServerConfig properties
    @property
    def transport(self) -> Literal["stdio", "http", "sse", "streamable-http"]:
        return self._config.transport

    # ... forwards all other protocol properties

# Protocol-specific adapters for fine-grained dependencies:
# - ServerConfigAdapter: IServerConfig only
# - SearchConfigAdapter: ISearchConfig only
# - StorageConfigAdapter: IStorageConfig only
# - RerankerConfigAdapter: IRerankerConfig only
# - EmbedderConfigAdapter: IEmbedderConfig only
# - ReplacementConfigAdapter: IReplacementConfig only
```

### Memory Protocols (memory.py)

The `memory.py` module defines interfaces for memory storage and management:

```python
@runtime_checkable
class IMemoryStore(Protocol):
    """Protocol for memory storage operations."""
    def add(self, message: str) -> str: ...
    def search(self, query: str, limit: int) -> list[tuple[str, float, str]]: ...
    def get_all(self) -> list[str]: ...
    def delete(self, memory_id: str) -> bool: ...

@runtime_checkable
class IMemoryBackend(Protocol):
    """Protocol for search engine backend."""
    def search(self, query: str, limit: int) -> list[tuple[str, float, str]]: ...
    def add(self, message: str) -> str: ...
    def delete(self, memory_id: str) -> bool: ...
    def get_all(self) -> list[str]: ...
```

### Search Protocols (search.py)

The `search.py` module defines interfaces for search backends and fusion:

```python
@runtime_checkable
class ISearchBackend(Protocol):
    """Protocol for search engine backend."""
    @property
    def name(self) -> str: ...
    def search(
        self,
        query: str,
        project_id: str,
        limit: int,
    ) -> list[tuple[str, float, str]]: ...

@runtime_checkable
class IFusionAlgorithm(Protocol):
    """Protocol for result fusion algorithms."""
    @property
    def method(self) -> str: ...
    def fuse(
        self,
        *result_sets: list[tuple[str, float]],
    ) -> list[tuple[str, float]]: ...
```

### Reranking Protocols (reranking.py)

The `reranking.py` module defines interfaces for reranking components:

```python
@runtime_checkable
class IRankingResult(Protocol):
    """Protocol for reranking result."""
    @property
    def document(self) -> str: ...
    @property
    def score(self) -> float: ...

@runtime_checkable
class IReranker(Protocol):
    """Protocol for reranking interface."""
    async def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        timestamp_map: dict[str, str] | None = None,
    ) -> list[tuple[str, float]]: ...
```

### Logging Protocols (logging.py)

The `logging.py` module defines interfaces for logging services:

```python
@runtime_checkable
class ILogSink(Protocol):
    """Protocol for log sink implementations."""
    def emit(self, level: LogLevel, message: str, **kwargs) -> None: ...

@runtime_checkable
class ILoggingService(Protocol):
    """Protocol for logging service."""
    def debug(self, message: str, **kwargs) -> None: ...
    def info(self, message: str, **kwargs) -> None: ...
    def warning(self, message: str, **kwargs) -> None: ...
    def error(self, message: str, **kwargs) -> None: ...
```

## Key Patterns

### Protocol-Based Design

Define protocols for abstractions to enable swapping implementations:

```python
class USearchEngine:
    '''USearch implementation of IMemoryBackend.'''

    def search(
        self,
        query: str,
        project_id: str,
        limit: int,
    ) -> list[tuple[str, float, str]]:
        '''Search implementation using USearch.'''
        ...
```

### Configuration Adapters

Use adapters to satisfy protocol interfaces with existing configuration:

```python
def create_memory_manager(
    config: IAppConfig,
    logger: StructuredLogger,
) -> MemoryManager:
    '''Create MemoryManager with protocol-based config.'''
    adapter = ConfigAdapter(config)
    return MemoryManager(config=adapter, logger=logger)
```

### Runtime Protocol Checking

Use `@runtime_checkable` protocol for runtime type checking:

```python
@runtime_checkable
class ISearchBackend(Protocol):
    def search(self, query: str, limit: int) -> list[tuple[str, float, str]]: ...

# Can check if object satisfies protocol at runtime
if isinstance(engine, ISearchBackend):
    results = engine.search(query, limit)
```

## Usage Examples

### Protocol-Based Dependency Injection

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
        results = self._backend.search(query, limit)
        return [r[0] for r in results]
```

### Adapter Composition

```python
from reflectlog.core.config_adapters import create_config_adapter

config = Config.from_environment()
adapter = create_config_adapter(config)

# Use adapter where IAppConfig is expected
service = SearchService(
    backend=usearch_engine,
    config=adapter,  # Satisfies ISearchConfig
)
```

## Testing Guidelines

### Mock Protocols

Use `unittest.mock.MagicMock` to create protocol mocks:

```python
@pytest.fixture
def mock_search_backend():
    backend = MagicMock(spec=ISearchBackend)
    backend.search.return_value = [("doc1", 0.9), ("doc2", 0.8)]
    return backend

def test_search_service(mock_search_backend):
    service = SearchService(
        backend=mock_search_backend,
        config=SearchConfigAdapter(default_config),
    )
    results = service.search("test query")
    assert len(results) == 2
```

### Protocol Verification

Verify implementations satisfy protocols:

```python
# Static type checking
from reflectlog.core.memory import IMemoryBackend
_: type[IMemoryBackend] = USearchEngine  # Type checker verifies

# Runtime verification
if isinstance(engine, IMemoryBackend):
    print("Engine implements IMemoryBackend")
```

## Dependencies

### Internal Dependencies

- `application/config/`: Configuration dataclasses
- `application/utils/`: Logging utilities

### External Dependencies

- `typing`: Protocol, runtime_checkable
- No external dependencies for protocols

## Important Notes

### Protocol vs Abstract Base Class

Use protocols (structural typing) instead of ABCs when:
- Multiple independent implementations exist
- Runtime type checking is needed
- Duck typing is preferred

Use ABCs when:
- Shared implementation is needed
- Method overrides must be enforced
- Class hierarchy is important

### Performance

Protocol checks have minimal runtime overhead with `@runtime_checkable`. For hot paths, consider caching protocol checks or using explicit type tags.

### Evolution

Protocols can be extended by adding new optional methods. Existing implementations automatically satisfy the protocol if they don't define the new methods.
