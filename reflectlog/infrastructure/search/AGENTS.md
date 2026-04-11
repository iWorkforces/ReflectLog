# Agent Guidelines for reflectlog/infrastructure/search/

**Generated:** 2026-04-11  **Commit:** 6f2b0f8  **Branch:** develop

**Generated:** 2026-04-11  **Commit:** 6f2b0f8  **Branch:** develop

## OVERVIEW
Search engine base classes and abstractions. Actual implementations (USearch, Tantivy) are in parent `infrastructure/` directory.

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Base class | search/base.py | SearchEngineBase with default ISearchBackend implementations |
| Protocol | core/search.py | ISearchBackend, ISearchResult |

## CONVENTIONS

**Default Implementations** - SearchEngineBase provides empty defaults for ISearchBackend methods. Subclasses override.

**Lifecycle Management** - ensure_initialized() for lazy init, commit() to persist, close() to release.

**Protocol Compliance** - All engines must satisfy ISearchBackend from core/search.py.

## ANTI-PATTERNS

- Never return raw backend results; always wrap in ISearchResult
- Never skip ensure_initialized() before index operations
- Never ignore commit() errors
- Never use triple single quotes; use """ only
