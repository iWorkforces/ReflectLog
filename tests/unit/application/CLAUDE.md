# tests/unit/application/

This directory contains unit tests for the application layer components.

## Structure

```
application/
├── test_mcp_server.py              # FastMCPServer tests
├── test_memory_manager.py          # MemoryManager tests
├── test_ranx_fusion.py             # RRF fusion algorithm tests
├── test_validation.py              # Message validation tests
├── test_logging_utils.py           # Logging utilities tests
├── test_dynamic_instructions.py    # Dynamic instruction generation tests
├── test_mcp_server_error_handling.py # Server error handling tests
├── memory/                          # Memory subsystem tests
│   └── reranking/                   # Score normalization tests
│       └── test_normalization.py
└── utils/                           # Utility module tests
    ├── test_logging.py              # Structured logging tests
    ├── test_numba_utils.py          # Numba JIT function tests
    └── test_security.py             # Security utilities tests
```

## Purpose

Test the `ccmemories/application/memory/` module in isolation:
- `MemoryManager` class (hybrid USearch + Tantivy)
- Dual-engine storage logic
- RRF fusion algorithm
- Deduplication logic
- Message validation and error handling
- Tantivy index management

## Current Test Files

| File | Tests For |
|------|-----------|
| `test_mcp_server.py` | `ccmemories/application/mcp_server.py` |
| `test_memory_manager.py` | `ccmemories/application/memory/manager.py` |
| `test_ranx_fusion.py` | `ccmemories/application/memory/fusion/ranx_fusion.py` |
| `test_validation.py` | `ccmemories/application/utils/validation.py` |
| `test_logging_utils.py` | `ccmemories/application/utils/logging.py` |
| `test_dynamic_instructions.py` | Dynamic instruction generation logic |
| `test_mcp_server_error_handling.py` | Server error handling paths |
| `memory/reranking/test_normalization.py` | `ccmemories/application/memory/reranking/normalization.py` |
| `utils/test_logging.py` | `ccmemories/application/utils/logging.py` |
| `utils/test_numba_utils.py` | `ccmemories/application/utils/numba_utils.py` |
| `utils/test_security.py` | `ccmemories/application/utils/security.py` |

## Test Organization for `test_memory_manager.py`

### Test Classes (Recommended Structure)

```python
class TestMemoryManagerInitialization:
    """Tests for MemoryManager.__init__()"""
    # - Successful initialization with USearch + Tantivy
    # - Missing PROJECT_ID
    # - USearch configuration
    # - Tantivy index creation/loading
    # - Logger initialization
    # - Hybrid mode configuration

class TestMessageValidation:
    """Tests for MemoryManager._validate_messages()"""
    # - Empty list (valid)
    # - Valid messages
    # - Too short messages
    # - Too long messages
    # - Whitespace-only messages
    # - Non-string items
    # - Boundary cases

class TestAddMessages:
    """Tests for the add_messages() method (sync wrapper)"""
    # - Empty list (no-op)
    # - Single valid message
    # - Multiple valid messages
    # - Deduplication logic
    # - Dual-engine storage failure
    # - Logging behavior

class TestPhasedParallelAdd:
    """Tests for add_messages_async() 3-phase parallel processing (8 tests)"""
    # - Phase 1: Parallel duplicate detection (batch + storage)
    # - Phase 2: Parallel smart replacement with semaphore
    # - Phase 3: Sequential database writes
    # - Dry run mode (no storage modifications)
    # - Batch deduplication (within same request)
    # - Storage deduplication (against existing)
    # - Message ordering preservation
    # - Error handling and graceful degradation

class TestParallelSmartReplacement:
    """Tests for parallel smart replacement LLM checks (14 tests)"""
    # - Multiple candidates checked in parallel
    # - Semaphore-limited concurrency
    # - Replacement detection with high confidence
    # - No replacement with low confidence
    # - LLM failure graceful degradation
    # - Archive before deletion

class TestGetAll:
    """Tests for the get_all() method"""
    # - Empty storage
    # - Single message
    # - Multiple messages
    # - USearch source of truth
    # - Storage failure
    # - Logging behavior

class TestHybridSearch:
    """Tests for the search() method"""
    # - Semantic search (USearch)
    # - Full-text search (Tantivy)
    # - RRF fusion algorithm
    # - Score filtering and thresholds
    # - Case-insensitive matching
    # - Empty query (Pydantic validation)
    # - No results
    # - AI reranking behavior
    # - Search failure
    # - Logging (engine counts + fusion results)

class TestRemoveMessages:
    """Tests for the remove() method"""
    # - Empty list (no-op)
    # - Exact match removal
    # - Case-sensitive matching
    # - Multiple occurrences
    # - Non-existent messages (silent)
    # - Deletion failure
    # - Logging behavior

class TestTantivyIndex:
    """Tests for Tantivy index management"""
    # - Index creation
    # - Index loading
    # - Schema validation
    # - Writer/reader lifecycle
    # - Commit and sync behavior
    # - Exact match checking

class TestRRFFusion:
    """Tests for _fuse_hybrid_results()"""
    # - RRF algorithm correctness
    # - Score normalization
    # - Duplicate removal
    # - Ranking consistency
    # - Edge cases (empty results)
```

## Critical Test Scenarios

### Initialization
```python
def test_initialization_without_project_id_raises_runtime_error():
    """PROJECT_ID is required - should raise RuntimeError."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError, match="PROJECT_ID"):
            MemoryManager(config, logger)

def test_usearch_engine_is_initialized():
    """Verify USearchEngine is properly initialized."""
    with patch.dict(os.environ, {'PROJECT_ID': 'test'}):
        with patch('ccmemories.application.memory.manager.USearchEngine') as mock:
            MemoryManager(config, logger)

            # Verify USearchEngine was initialized
            mock.assert_called_once()
            # Get the config passed to USearchEngine
            call_kwargs = mock.call_args
            assert call_kwargs is not None

def test_tantivy_index_creation():
    """Verify Tantivy index is created with correct schema."""
    with patch.dict(os.environ, {'PROJECT_ID': 'test'}):
        with patch('ccmemories.application.memory.manager.tantivy') as mock_tantivy:
            mock_index = Mock()
            mock_tantivy.Index.return_value = mock_index
            mock_tantivy.Index.open.side_effect = Exception("No existing index")

            MemoryManager(config, logger)

            # Verify schema creation
            mock_tantivy.SchemaBuilder.assert_called()
            mock_tantivy.Index.assert_called()
```

### Validation Edge Cases
```python
@pytest.mark.parametrize("message,should_be_valid", [
    ("x" * 30720, True),    # Max length
    ("x" * 30721, False),   # Over max
    ("x", True),            # Min length
    ("", False),            # Under min
    ("  ", False),          # Whitespace only
    ("hello world", True),  # Normal case
])
def test_message_length_boundaries(message, should_be_valid):
    """Test message length validation boundaries."""
    memory_manager = MemoryManager(config, logger)
    is_valid, _ = memory_manager._validate_messages([message])
    assert is_valid == should_be_valid
```

### Tool Behavior
```python
def test_add_empty_list_is_noop():
    """Empty list should not call memory.add()."""
    mock_memory = Mock()
    mock_tantivy = Mock()
    memory_manager = MemoryManager(config, logger)
    memory_manager.memory = mock_memory
    memory_manager.tantivy_index = mock_tantivy

    memory_manager.add_messages([])

    mock_memory.add.assert_not_called()
    mock_tantivy.assert_not_called()

def test_get_all_returns_usearch_source_of_truth():
    """get_all() should return USearch results as source of truth."""
    mock_semantic_engine = Mock()
    original = ["msg1", "msg2"]
    mock_semantic_engine.get_all.return_value = original

    memory_manager = MemoryManager(config, logger)
    memory_manager._semantic_engine = mock_semantic_engine

    result = memory_manager.get_all()
    assert result == original

def test_remove_uses_exact_match_not_contains():
    """remove() should use exact match, not substring."""
    mock_memory = Mock()

    # Simulate search returning a substring match
    mock_result = Mock()
    mock_result.__str__ = Mock(return_value="hello world")
    mock_result.index = "1"
    mock_memory.search.return_value = [mock_result]

    memory_manager = MemoryManager(config, logger)
    memory_manager.memory = mock_memory

    # Try to remove "hello" - should not match "hello world"
    memory_manager.delete_by_id("1")

    # delete() should be called (exact match on index)
    mock_memory.delete.assert_called_once_with(memory_id="1")

def test_hybrid_search_rrf_fusion():
    """Hybrid search should fuse USearch and Tantivy results with RRF."""
    mock_semantic_engine = Mock()
    mock_tantivy = Mock()

    # Mock USearch results
    usearch_result = ("Python tutorial", 0.9)

    mock_semantic_engine.search.return_value = [usearch_result]

    memory_manager = MemoryManager(config, logger)
    memory_manager._semantic_engine = mock_semantic_engine
    memory_manager._tantivy_engine = mock_tantivy

    # Mock Tantivy results
    tantivy_searcher = Mock()
    doc = Mock()
    doc.get_first.side_effect = lambda field: "test-project" if field == "project_id" else "JavaScript guide"
    tantivy_searcher.doc.return_value = doc
    tantivy_searcher.search.return_value = Mock(hits=[(0.8, 0)])
    mock_tantivy.searcher.return_value = tantivy_searcher

    memory_manager.tantivy_searcher = tantivy_searcher

    # Test RRF fusion
    usearch_results = [("Python tutorial", 0.9)]
    tantivy_results = [("JavaScript guide", 0.8)]
    fused_results = memory_manager._fuse_hybrid_results(usearch_results, tantivy_results)

    # Should have both messages ranked by RRF score
    assert len(fused_results) == 2
    assert "Python tutorial" in [msg for msg, score in fused_results]
    assert "JavaScript guide" in [msg for msg, score in fused_results]
```

### MemorySearchResult Behavior
```python
def test_search_handles_memory_search_result_objects():
    """search() should handle MemorySearchResult objects correctly."""
    mock_memory = Mock()

    # Create mock MemorySearchResult objects
    result1 = Mock()
    result1.lower.return_value = "python tutorial"
    result1.__str__ = Mock(return_value="Python Tutorial")

    result2 = Mock()
    result2.lower.return_value = "javascript guide"
    result2.__str__ = Mock(return_value="JavaScript Guide")

    mock_memory.search.return_value = [result1, result2]

    server = FastMCPServer()
    server.memory_manager.memory = mock_memory

    results = server.search("python")

    # Should filter to only Python result
    assert len(results) == 1
    assert results[0] == "Python Tutorial"
```

### Error Handling
```python
def test_add_wraps_memory_exception_in_runtime_error():
    """Storage exceptions should be wrapped in RuntimeError."""
    mock_memory = Mock()
    mock_memory.add.side_effect = ValueError("Storage error")

    server = FastMCPServer()
    server.memory_manager.memory = mock_memory

    with pytest.raises(RuntimeError, match="Failed to add messages"):
        server.add(["message"])

def test_search_with_empty_query_raises_validation_error():
    """Empty query should fail Pydantic validation."""
    server = FastMCPServer()

    # Pydantic Field(min_length=1) should raise ValidationError
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        server.search("")
```

### Logging
```python
def test_add_logs_with_structured_context():
    """add() should log with structured extra context."""
    with patch('ccmemories.application.mcp_server.get_logger') as mock_logger_func:
        mock_logger = Mock()
        mock_logger_func.return_value = mock_logger

        server = FastMCPServer()
        server.add(["message1", "message2"])

        # Verify INFO log with extra context
        mock_logger.info.assert_any_call(
            "Adding 2 message(s)",
            extra={
                "tool": "add",
                "count": 2,
                "project_id": "test-project",
            }
        )
```

## Mocking Patterns

### Mock MemoryManager
```python
@pytest.fixture
def mock_memory_manager():
    """Mock MemoryManager for all tests."""
    with patch('ccmemories.application.memory.MemoryManager') as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        yield mock_instance

def test_with_fixture(mock_memory_manager):
    server = FastMCPServer()
    server.add(["test"])
    mock_memory_manager.add_messages.assert_called_once()
```

### Mock Environment
```python
@pytest.fixture
def test_env():
    """Set up test environment variables."""
    env = {
        'PROJECT_ID': 'test-project',
        'SEARCH_LIMIT': '5',
        'RERANKER_ENGINE': 'llm',
    }
    with patch.dict(os.environ, env, clear=True):
        yield
```

## Running These Tests

```bash
# All application unit tests
uv run pytest tests/unit/application/ -v

# Specific test file
uv run pytest tests/unit/application/test_mcp_server.py -v

# Specific test class
uv run pytest tests/unit/application/test_mcp_server.py::TestAddTool -v

# Specific test
uv run pytest tests/unit/application/test_mcp_server.py::TestAddTool::test_add_empty_list -v

# With coverage
uv run pytest tests/unit/application/ --cov=ccmemories.application --cov-report=term-missing
```

## Coverage Goals

Aim for **90%+ coverage** of `ccmemories/application/mcp_server.py`:
- All `__init__` logic
- All tool implementations
- All validation paths
- All error handling paths
- All logging calls

Use coverage report to identify gaps:
```bash
./start-unittest.sh --coverage
open htmlcov/index.html
```


<claude-mem-context>
# Recent Activity

<!-- This section is auto-generated by claude-mem. Edit content outside the tags. -->

### Dec 21, 2025

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #177 | 4:12 PM | 🔵 | Complete CLAUDE.md Documentation Inventory | ~680 |

### Dec 26, 2025

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #593 | 12:03 PM | 🔄 | Simplified RRF disabled test by removing unnecessary setup | ~186 |
| #589 | 12:02 PM | ✅ | Added asyncio import to graceful degradation tests | ~158 |
| #557 | 11:53 AM | 🔄 | Test Method Made Async for RRF Disabled Test | ~208 |
| #556 | " | 🔵 | Test Structure for RRF Fusion Toggle | ~216 |
| #552 | 11:52 AM | 🔵 | Test file RRF fusion configuration analysis | ~242 |
| #547 | " | 🔄 | Removed Parallel Smart Replacement Test Class | ~305 |
| #540 | 11:50 AM | 🔄 | Test method removal for async delegation | ~194 |

### Jan 8, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #682 | 8:49 AM | 🔵 | Test Coverage Analysis | ~213 |
</claude-mem-context>