# tests/

This directory contains all test suites for the ReflectLogMCP server.

## Structure

```
tests/
├── conftest.py               # Shared pytest fixtures
├── demo_test.py              # Demo/example tests
├── test_enhanced_scoring.py  # Enhanced scoring tests
├── test_get_all_messages.py  # Get all messages tests
├── test_llm_infer.py         # LLM inference tests
├── test_llm_infer_simple.py  # Simple LLM inference tests
├── test_real_server.py       # Real server tests
├── test_real_server_fastmcp.py  # FastMCP server tests
├── test_real_server_mcp.py   # MCP protocol server tests
├── test_real_tools_direct.py # Direct tool invocation tests
├── test_score_filtering_demo.py  # Score filtering demo tests
├── test_verbose_logging.py   # Verbose logging tests
├── unit/                     # Unit tests (isolated, fast)
│   ├── application/          # Application layer unit tests
│   │   ├── memory/
│   │   │   └── reranking/
│   │   │       └── test_normalization.py  # Batch normalization tests
│   │   ├── utils/
│   │   │   ├── test_logging.py        # StructuredLogger tests
│   │   │   ├── test_numba_utils.py    # Numba JIT function tests
│   │   │   └── test_security.py       # Security utility tests
│   │   ├── test_mcp_server.py
│   │   ├── test_mcp_server_error_handling.py
│   │   ├── test_memory_manager.py
│   │   ├── test_ranx_fusion.py
│   │   ├── test_rrf_fusion_toggle.py  # RRF fusion toggle tests
│   │   ├── test_validation.py
│   │   ├── test_logging_utils.py
│   │   └── test_dynamic_instructions.py
│   ├── infrastructure/       # Infrastructure layer unit tests
│   │   ├── test_cross_encoder_reranker.py  # CrossEncoderReranker tests
│   │   ├── test_llm_reranker.py            # LLMReranker tests
│   │   ├── test_message_store.py
│   │   ├── test_qwen3_embedding.py
│   │   ├── test_smart_replacer.py          # SmartReplacer tests
│   │   ├── test_tantivy_engine.py
│   │   └── test_usearch_engine.py
│   └── test_server.py        # Server entry point tests
└── integration/              # Integration tests (with real dependencies)
    ├── test_memory_manager_usearch.py
    ├── test_mcp_workflows.py
    └── test_qwen_embeddings_integration.py
```

## Test Organization Philosophy

### Unit Tests (`unit/`)
- **Purpose**: Test individual components in isolation
- **Dependencies**: Mock all external dependencies (USearch, SQLite, Tantivy, OpenRouter)
- **Speed**: Fast (< 1 second per test)
- **Focus**: Business logic, validation, error handling, edge cases

### Integration Tests (`integration/`)
- **Purpose**: Test interaction between components and external systems
- **Dependencies**: Real USearch indices, real Tantivy indices, real or stubbed OpenRouter calls
- **Speed**: Slower (may take several seconds)
- **Focus**: End-to-end flows, data persistence, external API integration

## Running Tests

```bash
# All tests
./start-unittest.sh

# With coverage
./start-unittest.sh --coverage

# Parallel execution (faster)
./start-unittest.sh --parallel

# Specific file
./start-unittest.sh --file tests/unit/application/memory/test_manager.py

# Pattern matching
./start-unittest.sh --pattern "test_add"

# Only unit tests
uv run pytest tests/unit/ -v -m unit

# Only integration tests
uv run pytest tests/integration/ -v -m integration

# Exclude slow tests
uv run pytest -m "not slow"
```

## Pytest Configuration

Configuration is in `pyproject.toml` under `[tool.pytest.ini_options]`:

**Test discovery**:
- `testpaths = ["tests"]`
- `python_files = ["test_*.py", "*_test.py"]`
- `python_classes = ["Test*"]`
- `python_functions = ["test_*"]`

**Async support**:
- `asyncio_mode = "auto"` - Automatically detect async tests
- `asyncio_default_fixture_loop_scope = "function"` - New loop per test

**Markers**:
- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (slower, external deps)
- `@pytest.mark.slow` - Slow tests (> 5 seconds)

**Options**:
- `-v` - Verbose output
- `--strict-markers` - Error on unknown markers
- `--strict-config` - Error on config issues
- `--tb=short` - Shorter traceback format

## Writing Tests

### Test File Naming
- Unit tests: `tests/unit/{module}/test_{component}.py`
- Integration tests: `tests/integration/test_{feature}.py`

### Test Function Naming
- Descriptive: `test_add_empty_list_is_noop()`
- Pattern: `test_{method}_{scenario}_{expected_result}()`

### Test Structure (AAA Pattern)
```python
def test_something():
    # Arrange - Set up test data and mocks
    mock_memory = Mock()
    mock_tantivy = Mock()
    memory_manager = MemoryManager(config, logger)

    # Act - Execute the code under test
    result = memory_manager.some_method()

    # Assert - Verify the outcome
    assert result == expected_value
    mock_memory.some_method.assert_called_once()
    mock_tantivy.assert_called_once()
```

### Async Tests
```python
@pytest.mark.asyncio
async def test_async_operation():
    # Tests automatically get an event loop
    result = await some_async_function()
    assert result is not None
```

## Mocking Guidelines

### Mock External Dependencies
```python
from unittest.mock import Mock, patch

@patch('reflectlog.application.mcp_server.Memory')
def test_with_mocked_memory(mock_memory_class):
    mock_instance = Mock()
    mock_memory_class.from_config.return_value = mock_instance
    # Test code here
```

### Mock Environment Variables
```python
@patch.dict(os.environ, {'PROJECT_ID': 'test-project'})
def test_with_env_var():
    # Test code here
```

## Coverage Requirements

- **Minimum coverage**: 80% (configurable in `start-unittest.sh:22`)
- **Coverage config**: `[tool.coverage.*]` in `pyproject.toml`
- **Omit from coverage**:
  - `*/tests/*`
  - `*/__pycache__/*`
  - `*/venv/*`, `*/.venv/*`
- **Exclude lines**:
  - `pragma: no cover`
  - `if __name__ == .__main__.:`
  - `if TYPE_CHECKING:`
  - Abstract methods

### Viewing Coverage
```bash
# Generate HTML report
./start-unittest.sh --coverage

# Open in browser
open htmlcov/index.html
```

## Test Data Best Practices

1. **Use fixtures** for common test data:
```python
@pytest.fixture
def sample_messages():
    return ["Message 1", "Message 2", "Message 3"]
```

2. **Parametrize** for multiple test cases:
```python
@pytest.mark.parametrize("input,expected", [
    ("", False),
    ("valid", True),
])
def test_validation(input, expected):
    assert validate(input) == expected
```

3. **Keep test data realistic** but minimal

## Common Test Scenarios

### Testing MCP Tools
- Empty input handling
- Invalid input validation
- Successful operation
- Error handling (USearch failures, storage errors)
- Edge cases (max length, unicode, special chars)
- Defensive copying (for `get_all`)
- Exact matching (for `remove`)

### Testing Validation
- Minimum length boundary
- Maximum length boundary
- Whitespace-only strings
- Empty strings vs empty lists
- Type errors (non-string items)

### Testing Search
- Semantic similarity (USearch)
- Full-text search (Tantivy)
- Hybrid RRF fusion
- Score filtering and thresholds
- Case-insensitivity
- Empty results
- AI reranking behavior

## Continuous Integration

Pre-push hook runs:
1. Type checking: `./start-type-check.sh`
2. Linting: `./start-lint.sh --all`

**Note**: Tests are NOT run in pre-push hook. Run tests manually before pushing:
```bash
./start-unittest.sh --coverage
```

## Test Performance

- **Unit tests**: Should complete in < 10 seconds total
- **Integration tests**: May take 30+ seconds
- **Parallel execution**: Use `--parallel` for faster runs (uses all CPU cores)
- **Mark slow tests**: `@pytest.mark.slow` for tests > 5 seconds

## Debugging Tests

```bash
# Show print statements
uv run pytest tests/unit/ -v -s

# Stop on first failure
uv run pytest tests/unit/ -v -x

# Run last failed tests
uv run pytest tests/unit/ --lf

# Run failed tests first, then others
uv run pytest tests/unit/ --ff

# Verbose output with full tracebacks
uv run pytest tests/unit/ -vv
```
