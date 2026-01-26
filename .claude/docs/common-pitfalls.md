# Common Python 3.14 Migration Pitfalls

## 1. Incomplete File Coverage

**Problem:** Only updating some files but not all that use legacy types.

**Solution:** Search comprehensively before starting:

```bash
grep -r "List\[\|Dict\[\|Tuple\[\|Optional\[\|Union\[" reflectlog --include="*.py" | cut -d: -f1 | sort -u
```

## 2. Missing `Any` Import

**Problem:** `Any` is used in annotations but not imported.

**Error:**
```
NameError: name 'Any' is not defined
```

**Solution:** Add `from typing import Any` when using `dict[str, Any]`.

## 3. FastMCP Tool Signature Introspection

**Problem:** FastMCP uses `inspect.signature()` to read type hints. If types are undefined, tool registration fails.

**Error:**
```
NameError: name 'Dict' is not defined
at health_check() -> Dict[str, Any]:
```

**Solution:** Convert to Python 3.14 syntax or ensure import.

## 4. Inconsistent Replacements

**Problem:** Using `replaceAll` without context replaces too much or too little.

**Solution:** Be specific with context:

```python
# Instead of blanket replacement
async def health_check() -> dict[str, Any]:  # Good - specific

# Avoid ambiguous patterns
def _search_semantic(...) -> list[tuple[str, float, str]]:  # Good - has context
```

## 5. Changing Function Signatures

**Problem:** Editing introduces bugs like missing parameters.

**Error:**
```
_concatenate_results() takes 3 positional arguments but 4 were given
```

**Solution:** Preserve original signature when converting types:

```python
# Before (legacy)
def _concatenate_results(self, semantic, tantivy) -> List[Tuple]:
    pass

# After (broken - missing limit parameter)
def _concatenate_results(self, semantic, tantivy) -> list[tuple]:
    pass

# After (correct)
def _concatenate_results(self, semantic, tantivy, limit) -> list[tuple]:
    pass
```

## 6. Protocol Methods Not Calling Parent

**Problem:** Dataclass `__init__` in subclasses doesn't call parent.

**Error:**
```
Method "__init__" does not call the method of the same name in parent class
```

**Solution:** This is a pre-existing issue, not migration-related. Fix separately.

## 7. Stale LSP Cache

**Problem:** LSP reports errors after fix due to caching.

**Solution:** Run tests directly instead of relying on LSP diagnostics.

```bash
uv run pytest tests/unit/application/test_mcp_server.py -v
```

## 8. Mock Type Mismatches

**Problem:** Tests use mocks with old type annotations.

**Solution:** Update mock return types to match new signatures.

```python
# Old
mock_reranker.rerank.return_value = [("doc", 0.5)]

# New (still works, but ensure types align)
mock_reranker.rerank.return_value = [("doc", 0.5)]  # list[tuple[str, float]]
```
