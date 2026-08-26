# Agent Guidelines for reflectlog/infrastructure/search/

**Generated:** 2026-08-26  **Commit:** 95567fa  **Branch:** develop

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

## NOTES

Concrete USearch and Tantivy engines remain in the parent directory.
