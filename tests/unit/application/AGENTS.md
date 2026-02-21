# Application Layer Unit Tests

**Generated:** 2026-02-21
**Commit:** 4c3af26
**Branch:** develop

## OVERVIEW

Unit tests for application layer business logic. All external dependencies mocked.

## STRUCTURE

```
tests/unit/application/
├── test_mcp_server.py               # FastMCPServer orchestration (52KB)
├── test_memory_manager.py           # MemoryManager facade (43KB)
├── test_graceful_degradation.py     # Fallback behavior
├── test_mcp_server_error_handling.py # Error propagation
├── test_dynamic_instructions.py     # Instruction building
├── test_ranx_fusion.py              # RRF/CombSUM/MNZ fusion
├── test_rrf_fusion_toggle.py        # Fusion on/off switching
├── test_search_hypothesis.py        # Property-based search tests
├── test_validation.py               # Input validation
└── test_logging_utils.py           # Log formatting
```

## WHERE TO LOOK

| Test | Purpose |
|------|---------|
| test_mcp_server.py | Tool registration, request handling |
| test_memory_manager.py | Add/search/remove orchestration |
| test_graceful_degradation.py | Engine failure handling |
| test_ranx_fusion.py | RRF algorithm correctness |

## KEY PATTERNS

### Mock Injection
```python
@pytest.fixture
def mock_engines():
    usearch = MagicMock(spec=ISemanticSearchEngine)
    tantivy = MagicMock(spec=IFulltextSearchEngine)
    return usearch, tantivy

def test_search_with_mocks(mock_engines):
    manager = MemoryManager(
        usearch_engine=mock_engines[0],
        tantivy_engine=mock_engines[1]
    )
```

### Config Reset Pattern
```python
@pytest.fixture(autouse=True)
def reset_config_singleton():
    yield
    with config_settings._config_lock:
        config_settings._config = None
```

### Error Injection
```python
def test_engine_failure_graceful(mock_usearch):
    mock_usearch.search.side_effect = USearchError("fail")
    # Should not raise, should fallback
    results = manager.search("query")
```

## ANTI-PATTERNS

- Never import real infrastructure classes
- Never skip config reset between tests
- Never use `@pytest.mark.integration` here

## NOTES

- **All mocked**: No real USearch/Tantivy/LLM calls
- **Fast**: Each test <100ms
- **Coverage focus**: Targets 90%+ on application layer
