# Integration Tests

**Generated:** 2026-08-29  **Commit:** 7df1375  **Branch:** develop

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

## ANTI-PATTERNS

- Never mock USearch/Tantivy in integration tests
- Never skip cleanup - always close managers
- Never share managers between tests (isolation)

## NOTES
- Slow (10-60s each), NUMBA JIT enabled, `@pytest.mark.integration`
