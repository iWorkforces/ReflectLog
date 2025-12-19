# tests/unit/

This directory contains unit tests - fast, isolated tests that verify individual components without external dependencies.

## Structure

```
unit/
├── application/              # Application layer unit tests
│   ├── memory/
│   │   └── reranking/
│   │       └── test_normalization.py  # Batch normalization tests
│   ├── utils/
│   │   ├── test_logging.py        # StructuredLogger tests
│   │   ├── test_numba_utils.py    # Numba JIT function tests
│   │   └── test_security.py       # Security utility tests
│   ├── test_mcp_server.py
│   ├── test_mcp_server_error_handling.py
│   ├── test_memory_manager.py
│   ├── test_ranx_fusion.py
│   ├── test_rrf_fusion_toggle.py  # RRF fusion toggle tests
│   ├── test_validation.py
│   ├── test_logging_utils.py
│   └── test_dynamic_instructions.py
├── infrastructure/           # Infrastructure layer unit tests
│   ├── test_cross_encoder_reranker.py  # CrossEncoderReranker tests
│   ├── test_llm_reranker.py            # LLMReranker tests
│   ├── test_message_store.py
│   ├── test_qwen3_embedding.py
│   ├── test_smart_replacer.py          # SmartReplacer tests
│   ├── test_tantivy_engine.py
│   └── test_usearch_engine.py
└── test_server.py            # Server entry point tests
```

## Unit Testing Philosophy

**Characteristics of good unit tests**:
- **Fast**: Complete in milliseconds
- **Isolated**: No database, file system, network, or external API calls
- **Deterministic**: Same input always produces same output
- **Focused**: Test one thing at a time
- **Independent**: Can run in any order

## What to Unit Test

### Application Layer (`application/`)
- `MemoryManager` initialization (USearch + Tantivy)
- Hybrid search with RRF fusion
- Message storage (deduplication, dual engine)
- Tool implementations (`add_messages`, `get_all`, `search`, `remove`)
- Message validation logic
- Error handling and exception cases
- Edge cases (empty lists, max length, unicode, etc.)

### Infrastructure Layer (`infrastructure/`)
- `USearchEngine` vector search operations
- `TantivyEngine` full-text search operations
- `MessageStore` libSQL storage operations
- `LangchainQwenEmbeddings` embedding provider
- Index initialization and persistence
- Error handling and recovery

### Test Coverage Goals
- All public methods
- All validation logic
- All error paths
- Boundary conditions
- Edge cases

## Mocking Strategy

### Mock All External Dependencies
```python
from unittest.mock import Mock, patch, MagicMock

# Mock MemoryManager
@patch('ccmemories.application.memory.MemoryManager')
def test_server_initialization(mock_memory_manager):
    mock_instance = Mock()
    mock_memory_manager.return_value = mock_instance
    # Test server initialization
```

### Mock Environment Variables
```python
import os
from unittest.mock import patch

@patch.dict(os.environ, {
    'PROJECT_ID': 'test-project',
    'SEARCH_LIMIT': '10',
    'RERANKER_ENGINE': 'llm'
})
def test_with_env_vars():
    # Test code here
```

### Mock FastMCP Logger
```python
@patch('ccmemories.application.mcp_server.get_logger')
def test_logging(mock_get_logger):
    mock_logger = Mock()
    mock_get_logger.return_value = mock_logger

    # Test code that logs

    # Verify logging calls
    mock_logger.info.assert_called_once_with(
        "message",
        extra={"key": "value"}
    )
```

## Test Patterns

### Testing Tool Functions

**Pattern for testing MCP tools**:
```python
def test_add_messages_success():
    # Arrange
    with patch('ccmemories.application.mcp_server.Memory') as mock_memory_class:
        mock_memory = Mock()
        mock_memory_class.from_config.return_value = mock_memory

        memory_manager = MemoryManager(config, logger)
        messages = ["Test message 1", "Test message 2"]

        # Act
        server.add(messages)

        # Assert
        mock_memory.add.assert_called_once_with(messages=messages)
```

### Testing Validation

**Pattern for validation tests**:
```python
@pytest.mark.parametrize("messages,expected_valid,expected_error", [
    ([], True, ""),  # Empty list is valid
    (["valid"], True, ""),
    ([""], False, "too short"),
    (["   "], False, "whitespace"),
    (["a" * 30721], False, "too long"),
    ([123], False, "not a string"),
])
def test_validate_messages(messages, expected_valid, expected_error):
    memory_manager = MemoryManager(config, logger)
    is_valid, error = server._validate_messages(messages)

    assert is_valid == expected_valid
    if not expected_valid:
        assert expected_error in error.lower()
```

### Testing Error Handling

**Pattern for error scenarios**:
```python
def test_add_raises_runtime_error_on_memory_failure():
    with patch('ccmemories.application.mcp_server.Memory') as mock_memory_class:
        mock_memory = Mock()
        mock_memory.add.side_effect = Exception("Storage failure")
        mock_memory_class.from_config.return_value = mock_memory

        memory_manager = MemoryManager(config, logger)

        with pytest.raises(RuntimeError, match="Failed to add messages"):
            server.add(["message"])
```

### Testing Defensive Copying

**Pattern for immutability tests**:
```python
def test_get_all_returns_defensive_copy():
    with patch('ccmemories.application.mcp_server.Memory') as mock_memory_class:
        mock_memory = Mock()
        original = ["message1", "message2"]
        mock_memory.get_all.return_value = original
        mock_memory_class.from_config.return_value = mock_memory

        memory_manager = MemoryManager(config, logger)
        result = server.get_all()

        # Modify the result
        result.append("message3")

        # Verify original is unchanged
        assert len(original) == 2
```

## Running Unit Tests

```bash
# All unit tests
uv run pytest tests/unit/ -v

# With marker
uv run pytest -m unit -v

# Specific module
uv run pytest tests/unit/application/ -v

# Watch mode
./start-unittest.sh --watch

# With coverage
./start-unittest.sh --coverage
```

## Test Organization

### File Naming

- Match source structure: `ccmemories/application/mcp_server.py` → `tests/unit/application/test_mcp_server.py`
- Use `test_` prefix for test files

### Test Class Organization (Optional)
```python
class TestMemoryManager:
    """Tests for MemoryManager initialization and hybrid storage."""

    def test_initialization_success(self):
        pass

    def test_initialization_without_project_id_raises_error(self):
        pass

class TestHybridSearch:
    """Tests for hybrid USearch + Tantivy search."""

    def test_semantic_search(self):
        pass

    def test_fulltext_search(self):
        pass

    def test_rrf_fusion(self):
        pass

class TestAddMessages:
    """Tests for the add_messages() method."""

    def test_add_empty_list(self):
        pass

    def test_add_valid_messages(self):
        pass
```

## Common Test Scenarios

### Initialization Tests
- Successful MemoryManager initialization with PROJECT_ID
- Failure without PROJECT_ID
- USearch configuration correctness
- Tantivy index creation/loading
- Hybrid mode configuration

### Validation Tests
- Empty list (should be valid)
- Too short messages
- Too long messages
- Whitespace-only messages
- Non-string items
- Unicode and special characters

### Hybrid Storage Tests
- Dual engine storage (USearch + Tantivy)
- Deduplication logic
- Tantivy exact match checking
- Commit and sync behavior

### Search Tests
- Semantic search (USearch)
- Full-text search (Tantivy)
- Hybrid RRF fusion
- Score filtering and thresholds
- Case-insensitive matching
- Empty results
- Search failure modes

### Tool Tests - `add_messages()`
- Empty list (no-op)
- Single message
- Multiple messages
- Invalid messages (validation failure)
- Storage failure (RuntimeError)

### Tool Tests - `get_all()`
- Empty storage
- Multiple messages
- USearch source of truth
- Storage failure

### Tool Tests - `search()`
- Hybrid search with RRF
- Score threshold filtering
- AI reranking (when enabled)
- Search failure modes

### Tool Tests - `remove()`
- Exact match removal
- Case-sensitive matching
- Multiple occurrences
- Non-existent messages (silent)
- Empty list (no-op)
- Removal failure

## Fixtures

**Common fixtures** (create in `conftest.py` if needed):
```python
@pytest.fixture
def mock_memory_manager():
    """Mock MemoryManager instance."""
    with patch('ccmemories.application.memory.MemoryManager') as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def project_id_env():
    """Set PROJECT_ID environment variable."""
    with patch.dict(os.environ, {'PROJECT_ID': 'test-project'}):
        yield

@pytest.fixture
def sample_messages():
    """Sample test messages."""
    return [
        "First test message",
        "Second test message",
        "Third test message"
    ]
```

## Assertions Best Practices

```python
# Specific assertions
assert result == expected
assert len(items) == 3
assert "error" in message.lower()

# Mock assertions
mock_method.assert_called_once()
mock_method.assert_called_once_with(arg1, arg2)
mock_method.assert_not_called()
assert mock_method.call_count == 2

# Exception assertions
with pytest.raises(ValueError, match="Invalid message"):
    function_under_test()

# Logging assertions
mock_logger.info.assert_called_with(
    "message",
    extra={"key": "value"}
)
```

## Test Data Guidelines

- **Keep it simple**: Minimal data to prove the point
- **Use descriptive values**: "test_message_1" better than "a"
- **Avoid magic numbers**: Use constants or fixtures
- **Test boundaries**: Min/max lengths, edge cases
- **Test real-world scenarios**: Unicode, special chars, long strings
