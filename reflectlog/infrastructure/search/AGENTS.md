# Agent Guidelines for reflectlog/infrastructure/search/

This directory contains base classes and abstractions for search engine implementations. Actual engine implementations (USearch, Tantivy) are located in the parent `infrastructure/` directory and re-exported here.

## Directory Structure

```
search/
├── __init__.py            # Re-exports from parent
└── base.py                 # SearchEngineBase class
```

## Core Responsibilities

### SearchEngineBase (base.py)

The `SearchEngineBase` class provides common functionality for search engine implementations and ensures conformance to `ISearchBackend` protocol from `core/search.py`.

Key features:
- Default implementations for all required methods
- Proper type hints for protocol compliance
- Extensible for backend-specific customization
- Lifecycle management (search, add, delete, commit, close)

## Base Class

```python
class SearchEngineBase:
    """Base class for search engine implementations.

    This class provides default implementations for all ISearchBackend
    methods. Subclasses should override with backend-specific logic.

    Attributes:
        _name: Backend identifier for logging.
    """

    _name: str = "base"

    @property
    def name(self) -> str:
        """Backend identifier for logging."""
        return self._name

    async def search(
        self,
        query: str,
        project_id: str,
        limit: int,
    ) -> list[ISearchResult]:
        """Default search implementation returns empty results."""
        return []

    async def add(
        self,
        project_id: str,
        documents: list[str],
    ) -> None:
        """Default add implementation does nothing."""
        pass

    async def delete(
        self,
        document_id: str,
    ) -> None:
        """Default delete implementation does nothing."""
        pass

    async def commit(self) -> None:
        """Default commit does nothing."""
        pass

    async def close(self) -> None:
        """Default close does nothing."""
        pass

    def ensure_initialized(self) -> None:
        """Default initialization does nothing."""
        pass
```

## Usage Pattern

When implementing a new search engine:

```python
from reflectlog.infrastructure.search.base import SearchEngineBase
from reflectlog.core.search import ISearchBackend, ISearchResult

class MySearchEngine(SearchEngineBase):
    """Custom search engine implementation."""

    _name = "my_engine"

    def __init__(self, config: MySearchConfig):
        self.config = config
        self._index = None

    def ensure_initialized(self) -> None:
        """Initialize the search index."""
        if self._index is None:
            self._index = create_my_index(self.config)

    async def search(
        self,
        query: str,
        project_id: str,
        limit: int,
    ) -> list[ISearchResult]:
        """Search implementation."""
        self.ensure_initialized()
        results = self._index.search(query, limit)
        return [
            ISearchResult(
                document=hit.text,
                score=hit.score,
                metadata={"id": hit.id}
            )
            for hit in results
        ]

    async def add(
        self,
        project_id: str,
        documents: list[str],
    ) -> None:
        """Add documents to index."""
        self.ensure_initialized()
        for doc in documents:
            self._index.add(doc)

    async def commit(self) -> None:
        """Commit pending changes."""
        if self._index:
            self._index.commit()

    async def close(self) -> None:
        """Close resources."""
        if self._index:
            self._index.close()
```

## Re-Exports

The `__init__.py` module re-exports actual engine implementations from the parent directory:

```python
# Re-export from parent module for backward compatibility
from reflectlog.infrastructure.usearch_engine import USearchEngine, USearchConfig
from reflectlog.infrastructure.tantivy_engine import TantivyEngine, TantivyConfig

__all__ = [
    "USearchEngine",
    "USearchConfig",
    "TantivyEngine",
    "TantivyConfig",
]
```

## Testing Guidelines

### Unit Tests

Test search engine implementations with mock indices:

```python
@pytest.fixture
def mock_search_engine():
    engine = MySearchEngine(test_config)
    yield engine
    # Cleanup is automatic via close()

def test_search_basic(mock_search_engine):
    mock_search_engine.add("test_project", ["hello world"])
    results = mock_search_engine.search("test", "test_project", 10)
    assert len(results) > 0
```

### Integration Tests

Test with real index files:

```python
@pytest.fixture
def temp_index(tmp_path):
    index_path = tmp_path / "test.index"
    engine = MySearchEngine(MySearchConfig(index_path=str(index_path)))
    yield engine
    engine.close()
```

## Dependencies

### Internal Dependencies

- `core/search.py`: `ISearchBackend`, `ISearchResult` protocols
- `infrastructure/usearch_engine.py`: USearch implementation
- `infrastructure/tantivy_engine.py`: Tantivy implementation

### External Dependencies

- No external dependencies for base class (implementations may vary)

## Important Notes

### Protocol Compliance

All search engine implementations must satisfy `ISearchBackend` protocol. The base class provides default implementations, but subclasses should override with actual functionality.

### Lifecycle Management

- **ensure_initialized()**: Lazy initialization pattern
- **commit()**: Persist pending changes
- **close()**: Release resources on shutdown

### Thread Safety

Implementations must handle concurrent operations appropriately:
- USearch is not thread-safe; serialize writes
- Tantivy handles concurrent reads safely

### Error Handling

Wrap backend-specific exceptions in domain types:

```python
try:
    results = self._index.search(query)
except BackendSpecificError as e:
    raise SearchError(f"Search failed: {e}") from e
```

## Future Expansion

When adding new search engines:
1. Create implementation file in parent `infrastructure/` (e.g., `elasticsearch_engine.py`)
2. Inherit from `SearchEngineBase`
3. Override methods with backend-specific logic
4. Add re-export to `search/__init__.py`
5. Update this documentation
