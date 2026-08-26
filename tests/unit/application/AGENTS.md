# Application Layer Unit Tests

**Generated:** 2026-08-26  **Commit:** 95567fa  **Branch:** develop

## OVERVIEW

Unit tests for application layer business logic. All external dependencies mocked.

## STRUCTURE

```
tests/unit/application/
├── test_mcp_server.py               # FastMCPServer orchestration
├── test_memory_manager.py           # MemoryManager facade
├── test_graceful_degradation.py     # Fallback behavior
├── test_ranx_fusion.py              # RRF/CombSUM/MNZ fusion
├── test_search_hypothesis.py        # Property-based search tests
├── test_validation.py               # Input validation
└── (see subdirectory AGENTS.md for memory/, config/, utils/)
```

## WHERE TO LOOK

| Test | Purpose |
|------|---------|
| test_mcp_server.py | Tool registration, request handling |
| test_memory_manager.py | Add/search/remove orchestration |
| test_graceful_degradation.py | Engine failure handling |
| test_ranx_fusion.py | RRF algorithm correctness |

## ANTI-PATTERNS

- Never import real infrastructure classes
- Never skip config reset between tests
- Never use `@pytest.mark.integration` here

## NOTES

- **All mocked**: No real USearch/Tantivy/LLM calls
- **Fast**: Each test <100ms
- **Coverage focus**: Targets 90%+ on application layer
