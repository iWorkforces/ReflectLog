# Type Stubs Directory

**Generated:** 2026-04-11  **Commit:** 6f2b0f8  **Branch:** develop

## OVERVIEW

Custom type stubs for third-party libraries lacking type hints. Configured in pyproject.toml via `ty.extra-paths` and `pyright.stubPath`.

## STRUCTURE

```
stubs/
├── fastmcp/                     # FastMCP server framework
│   ├── __init__.pyi
│   ├── client/
│   └── utilities/
├── tantivy/                     # Full-text search engine
│   └── __init__.pyi
├── usearch/                     # Vector search engine
│   ├── __init__.pyi
│   └── index.pyi
├── ranx/                        # Ranking fusion library
│   └── __init__.pyi
├── numba/                       # JIT compiler
│   └── core/
├── sentence_transformers/       # Embedding models
│   └── __init__.pyi
└── hypothesis/                  # Property-based testing
    └── __init__.pyi
```

## WHERE TO LOOK

| Stub | Library | Purpose |
|------|---------|---------|
| fastmcp/ | fastmcp | MCP server framework |
| tantivy/ | tantivy-py | Full-text search |
| usearch/ | usearch | Vector similarity |
| ranx/ | ranx | RRF/CombSUM fusion |
| numba/ | numba | JIT compilation |

## KEY PATTERNS

### Adding New Stubs
```python
# stubs/newlib/__init__.pyi
def important_function(arg: str) -> int: ...
class ImportantClass:
    def method(self) -> None: ...
```

### Configuration
```toml
# pyproject.toml
[tool.ty]
extra-paths = ["stubs"]

[tool.pyright]
stubPath = "stubs"
```

## ANTI-PATTERNS

- Never add stubs for typed libraries
- Never use `Any` in stubs - be precise
- Never forget to update when upgrading libs

## NOTES

- **Minimal stubs**: Only what's actually used
- **Version-locked**: Stubs match installed versions
- **Contrib welcome**: Submit upstream when complete
