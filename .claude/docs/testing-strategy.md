# Testing Strategy for Type Migrations

## Phase 1: Baseline

Run tests before migration to establish baseline:

```bash
./start-unittest.sh 2>&1 | grep -E "passed|failed|skipped"
```

Note any pre-existing failures.

## Phase 2: Incremental Testing

Test file-by-file during migration:

```bash
# Test single module
uv run pytest tests/unit/application/test_mcp_server.py -v

# Test specific class
uv run pytest tests/unit/application/test_memory_manager.py::TestHybridMemoryManager -v
```

## Critical Tests to Run

1. **Server initialization** - Type errors often surface here
   ```bash
   uv run pytest tests/unit/application/test_mcp_server.py::TestFastMCPServerInitialization -v
   ```

2. **Tool registration** - FastMCP introspects function signatures
   ```bash
   uv run pytest tests/unit/application/test_mcp_server.py -v -k "Tool"
   ```

3. **Memory operations** - Complex type pipelines
   ```bash
   uv run pytest tests/unit/application/test_memory_manager.py -v
   ```

## Common Failure Patterns

### `NameError` in Tool Handlers

FastMCP uses `inspect.signature()` which fails if types aren't defined:

```
NameError: name 'Dict' is not defined
```

**Fix:** Ensure all types in annotations are imported or use Python 3.14 native syntax.

### Mock Return Value Mismatches

If tests use mocks with typed return values, ensure mocks match:

```python
# Test might fail if mock returns wrong type
mock_reranker.rerank.return_value = [("doc", 0.5)]
# Should match actual return type: list[tuple[str, float]]
```

## Full Test Run

After migration complete:

```bash
./start-unittest.sh 2>&1 | tail -5
```

**Expected:** Similar or better pass rate than baseline. Pre-existing failures are acceptable.
