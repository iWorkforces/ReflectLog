# Python 3.14 Migration Checklist

## Pre-Migration

- [ ] Verify project requires Python 3.14+ in `pyproject.toml`
- [ ] Update CI/CD to use Python 3.14
- [ ] Run existing tests to establish baseline

## Migration Steps

### 1. Find Legacy Imports

```bash
grep -r "from typing import.*List\|from typing import.*Optional\|from typing import.*Dict" reflectlog --include="*.py"
```

### 2. Convert Type Annotations

Run these replacements per file:

| Pattern | Replacement |
|---------|-------------|
| `List[` | `list[` |
| `Dict[` | `dict[` |
| `Tuple[` | `tuple[` |
| `Set[` | `set[` |
| `Optional[` | `... | None` |
| `Union[` | `... | ... |` |

### 3. Update Imports

Remove unused imports:
- `Optional`, `List`, `Dict`, `Tuple`, `Set`, `Union`

Keep:
- `TYPE_CHECKING`, `Protocol`, `Any`, `Literal`, `TypedDict`, `Callable`, `TypeVar`, `Generic`, `TypeAlias`, `Annotated`

### 4. Verify Imports

```bash
grep -r "from typing import" reflectlog --include="*.py" | grep -v "TYPE_CHECKING\|Protocol\|Literal\|Callable\|TypeVar\|Generic\|TypeAlias\|Annotated"
```

Should return no results.

## Common Issues

### `NameError: name 'Dict' is not defined`

**Cause:** `Dict` used in annotation but import removed.

**Fix:** Convert to `dict[...]` syntax.

### `NameError: name 'Any' is not defined`

**Cause:** `Any` used but not imported.

**Fix:** Add `from typing import Any`.

### Function Signature Mismatches

**Cause:** Editing introduced signature changes.

**Fix:** Ensure method definitions match calls.

## Post-Migration

- [ ] Run type checker (`./start-type-check.sh`)
- [ ] Run full test suite
- [ ] Fix any `NameError` in tool registration (common in FastMCP tools)
