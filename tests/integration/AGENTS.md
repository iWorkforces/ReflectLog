# Integration Tests

**Generated:** 2026-02-21
**Commit:** 4c3af26
**Branch:** develop

## OVERVIEW

End-to-end tests using real USearch + Tantivy engines. Tests concurrency, thread safety, and MCP workflows.

## STRUCTURE

```
tests/integration/
├── test_chaos.py                    # Randomized stress testing
├── test_concurrent_operations.py    # Parallel add/search/delete
├── test_mcp_workflows.py            # Full MCP tool workflows
├── test_memory_manager_usearch.py   # USearch-specific integration
├── test_qwen_embeddings_integration.py  # Real embedding API tests
└── test_thread_safety.py            # Multi-threaded access
```

## WHERE TO LOOK

| Test | Purpose |
|------|---------|
| test_concurrent_operations.py | Race conditions, parallel batch ops |
| test_thread_safety.py | Lock hierarchy, deadlock prevention |
| test_mcp_workflows.py | Full add→search→remove cycles |
| test_chaos.py | Random operations, edge case discovery |

## KEY PATTERNS

### Real Engine Initialization
```python
@pytest.fixture
async def real_memory_manager(tmp_path):
    '''Uses actual USearch + Tantivy, not mocks.'''
    config = Config(project_id="test_integration", ...)
    manager = MemoryManager(config)
    yield manager
    await manager.close()
```

### Concurrency Tests
```python
async def test_concurrent_adds_different_projects():
    '''Multiple projects adding simultaneously.'''
    async with anyio.create_task_group() as tg:
        for project in projects:
            tg.start_soon(add_messages, project, messages)
```

### Chaos Testing
```python
@pytest.mark.parametrize("seed", range(100))
def test_random_operations(seed):
    '''Fuzz testing with randomized operations.'''
    random.seed(seed)
    # Random mix of add/search/delete
```

## ANTI-PATTERNS

- Never mock USearch/Tantivy in integration tests
- Never skip cleanup - always close managers
- Never share managers between tests (isolation)

## NOTES

- **Slow**: Integration tests take 10-60s each
- **Requires cleanup**: Temp directories auto-removed
- **NUMBA enabled**: JIT compilation active (unlike unit tests)
- **Markers**: `@pytest.mark.integration`
