# ReflectLogMCP Test Suite

**Generated:** 2026-01-27
**Commit:** 94d13da
**Branch:** develop

## OVERVIEW

Comprehensive test suite covering unit and integration tests for the ReflectLogMCP memory system. Enforces 90% minimum coverage threshold.

## STRUCTURE

```
tests/
├── conftest.py                 # Pytest configuration and fixtures
├── unit/                      # Unit tests (mocked dependencies)
│   ├── __init__.py
│   ├── test_server.py          # Server startup tests
│   ├── application/            # Application layer tests
│   │   ├── test_validation.py
│   │   ├── test_mcp_server.py
│   │   └── memory/          # Memory management tests
│   │       └── reranking/    # Reranking algorithm tests
│   ├── infrastructure/         # Infrastructure layer tests
│   └── application/utils/      # Utility function tests
└── integration/               # Integration tests (real components)
    └── tests/                # End-to-end scenarios
```

## TEST CONFIGURATION

### Pytest Settings (conftest.py)

```python
# pytest configuration
pytest_plugins = ["anyio", "pytest_asyncio"]
asyncio_mode = "auto"

# Coverage requirements
--cov=reflectlog
--cov-report=html
--cov-report=term-missing
--cov-fail-under=90
```

### Coverage Requirements

- **Minimum threshold:** 90%
- **Reports:** HTML + terminal with missing lines
- **Precision:** 2 decimal places
- **Uncovered lines:** Shown even if covered (skip_covered=false)

## KEY FIXTURES

### Memory Manager Fixture

```python
@pytest.fixture
async def memory_manager(tmp_path, monkeypatch):
    """Create isolated MemoryManager with temp storage."""
    project_id = "test_project"
    monkeypatch.setenv("PROJECT_ID", project_id)
    manager = await create_memory_manager(project_id)
    yield manager
    # Cleanup handled automatically
```

### Mock Search Engines

```python
@pytest.fixture
def mock_usearch_engine():
    """Mock USearch semantic search engine."""
    engine = MagicMock(spec=ISemanticSearchEngine)
    engine.search.return_value = [
        ("result1", 0.9, "id1"),
        ("result2", 0.8, "id2"),
    ]
    return engine

@pytest.fixture
def mock_tantivy_engine():
    """Mock Tantivy full-text search engine."""
    engine = MagicMock(spec=IFulltextSearchEngine)
    engine.search.return_value = [
        ("result1", 0.95),
        ("result2", 0.85),
    ]
    return engine
```

### LLM Provider Mocks

```python
@pytest.fixture
def mock_openai_provider(monkeypatch):
    """Mock OpenAI embedder provider."""
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = EmbeddingResponse(data=[
        Embedding(embedding=[0.1] * 4096)
    ])
    return mock_client
```

## UNIT TEST PATTERNS

### Mock External Dependencies

```python
def test_search_with_mocked_engines(
    mock_usearch_engine,
    mock_tantivy_engine,
):
    """Test search logic with mocked backends."""
    manager = MemoryManager(
        usearch_engine=mock_usearch_engine,
        tantivy_engine=mock_tantivy_engine,
    )

    results = manager.search("test query")
    assert len(results) > 0
```

### Test Exception Handling

```python
def test_search_handles_engine_error(mock_usearch_engine):
    """Search should fall back gracefully on engine errors."""
    mock_usearch_engine.search.side_effect = USearchError("Backend failed")

    manager = MemoryManager(usearch_engine=mock_usearch_engine)

    # Should not raise, handle gracefully
    results = manager.search("query")
    # Verify fallback behavior
```

### Test Configuration Loading

```python
def test_config_from_environment(monkeypatch):
    """Configuration should load from env vars."""
    monkeypatch.setenv("PROJECT_ID", "test_proj")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")

    config = Config.from_environment()
    assert config.project_id == "test_proj"
    assert config.openrouter_api_key == "test_key"
```

## INTEGRATION TEST PATTERNS

### Real Engine Tests

```python
@pytest.mark.integration
async def test_add_search_with_real_engines(memory_manager):
    """End-to-end test with real USearch + Tantivy."""
    # Add messages
    await memory_manager.add_messages_async([
        "I prefer Python for web development",
        "JavaScript is great for frontend",
    ])

    # Search semantically
    results = memory_manager.search("web frameworks")
    assert len(results) > 0
    assert "Python" in results[0]
```

### Persistence Tests

```python
@pytest.mark.integration
async def test_persistence_across_restart(memory_manager, tmp_path):
    """Data should persist across manager recreation."""
    # Add data
    await memory_manager.add_messages_async(["test message"])

    # Re-create manager (simulates restart)
    new_manager = await create_memory_manager("test_project")

    # Data should still be present
    all_messages = new_manager.get_all()
    assert "test message" in all_messages
```

### Concurrency Tests

```python
@pytest.mark.integration
async def test_concurrent_adds(memory_manager):
    """Concurrent adds should be thread-safe."""
    messages = [f"message {i}" for i in range(100)]

    # Run concurrent adds
    tasks = [
        memory_manager.add_messages_async(messages[i::10])
        for i in range(10)
    ]
    results = await asyncio.gather(*tasks)

    # All should succeed
    total_stored = sum(r.stored_count for r in results)
    assert total_stored == 100
```

## TEST ORGANIZATION

### Markers

```python
@pytest.mark.unit        # Unit tests (mocked dependencies)
@pytest.mark.integration # Integration tests (real components)
```

### Running Tests

```bash
# All tests
./start-unittest.sh

# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# With coverage
./start-unittest.sh --coverage

# Parallel execution
./start-unittest.sh --parallel

# Specific pattern
./start-unittest.sh --pattern test_add
```

## TEST DATA

### Test Messages

Use consistent test messages for reproducibility:

```python
TEST_MESSAGES = [
    "I prefer Python for web development",
    "JavaScript is great for frontend",
    "FastAPI is my favorite web framework",
]
```

### Test Queries

```python
TEST_QUERIES = [
    "web frameworks",
    "backend development",
    "API design",
]
```

## COMMON PATTERNS

### Async Test Structure

```python
@pytest.mark.asyncio
async def test_async_operation(memory_manager):
    """Async test with MemoryManager."""
    result = await memory_manager.add_messages_async(["test"])
    assert result.stored_count == 1
```

### Fixture Composition

```python
@pytest.fixture
def memory_manager_with_data(memory_manager):
    """MemoryManager pre-populated with test data."""
    memory_manager.add_messages([
        "test message 1",
        "test message 2",
    ])
    yield memory_manager
    # cleanup handled by parent fixture
```

### Parameterized Tests

```python
@pytest.mark.parametrize("query,expected", [
    ("web frameworks", 2),
    ("backend", 1),
    ("api", 1),
])
def test_search_variations(memory_manager, query, expected):
    """Test different search queries."""
    results = memory_manager.search(query)
    assert len(results) == expected
```

## PERFORMANCE TESTS

### Benchmark Operations

```python
def test_add_performance(memory_manager, benchmark):
    """Benchmark message addition performance."""
    messages = [f"message {i}" for i in range(1000)]

    def add_batch():
        memory_manager.add_messages(messages)

    result = benchmark(add_batch)
    # Verify result
    assert result.stored_count == 1000
```

### Search Performance

```python
def test_search_performance(memory_manager, benchmark):
    """Benchmark search performance."""
    # Pre-populate with 10k messages
    memory_manager.add_messages([f"msg {i}" for i in range(10000)])

    def search_query():
        return memory_manager.search("test query")

    results = benchmark(search_query)
    assert len(results) > 0
```

## ERROR TEST CASES

### Invalid Input Tests

```python
def test_add_empty_messages(memory_manager):
    """Adding empty list should handle gracefully."""
    result = memory_manager.add_messages([])
    assert result.stored_count == 0
    assert result.skipped_count == 0
```

### Duplicate Tests

```python
def test_duplicate_prevention(memory_manager):
    """Duplicates should be skipped."""
    message = "test message"
    memory_manager.add_messages([message])

    result = memory_manager.add_messages([message])
    assert result.skipped_count == 1
    assert result.stored_count == 0
```

### Search Edge Cases

```python
def test_search_empty_query(memory_manager):
    """Empty search query should return empty results."""
    results = memory_manager.search("")
    assert len(results) == 0
```

## NOTES

- **Coverage Enforced:** Tests fail if coverage < 90%
- **Parallel Execution:** Use `pytest-xdist` for faster runs
- **Isolation:** Each test uses temporary storage directories
- **Mock Strategy:** External services (OpenAI, USearch, Tantivy) mocked in unit tests
- **Real Components:** Integration tests use actual USearch + Tantivy engines
- **Async Tests:** All async tests use `pytest-asyncio` with `auto` mode
- **Fixture Cleanup:** Automatic cleanup via pytest fixtures
- **Type Checking:** Tests run through `ty` type checker
