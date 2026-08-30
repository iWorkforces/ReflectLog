# Type Stubs Directory

**Generated:** 2026-08-30  **Commit:** 062b44f  **Branch:** develop

## OVERVIEW
Minimal third-party stubs for `ty.extra-paths` and `pyright.stubPath`. Dual typecheck (`ty` + pyright) must pass. No `type: ignore`.

## STRUCTURE
```
stubs/
├── fastmcp/
│   ├── __init__.pyi
│   ├── exceptions.pyi
│   ├── client/
│   ├── server/
│   │   ├── dependencies.pyi
│   │   └── middleware.pyi
│   └── utilities/
├── tantivy/                     # __init__.pyi
├── usearch/                     # __init__.pyi + index.pyi
├── ranx/                        # __init__.pyi
├── numba/                       # core/errors.pyi
├── sentence_transformers/
├── hypothesis/
├── flagembedding/
└── claude_agent_sdk/
```

## WHERE TO LOOK
| Stub | Purpose |
|------|---------|
| `fastmcp/exceptions.pyi` | FastMCP exception types |
| `fastmcp/server/dependencies.pyi` | Server dependency injection |
| `fastmcp/server/middleware.pyi` | Server middleware types |
| `tantivy/` | FTS engine |
| `usearch/` | HNSW index |
| `ranx/` | Fusion (RRF and others) |

## CONVENTIONS
- Only stub symbols the tree actually imports.
- Native 3.14 unions; no `Any` unless the library is untyped at that call site.
- `pyproject.toml`: `[tool.ty] extra-paths = ["stubs"]`, `[tool.pyright] stubPath = "stubs"`.

```python
# stubs/newlib/__init__.pyi
def important_function(arg: str) -> int: ...
```

## ANTI-PATTERNS
- Never stub already-typed libraries.
- Never use `getattr` / `optional_attr()` / `type(obj).__dict__` in stubs.
- Never leave stubs stale after a dependency bump.

## NOTES
FastMCP now includes `exceptions.pyi` and `server/{dependencies,middleware}.pyi`. Keep them in sync with the installed fastmcp version.
