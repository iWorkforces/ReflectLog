# Agent Guidelines for reflectlog/infrastructure/search/

**Generated:** 2026-08-26  **Commit:** 95567fa  **Branch:** develop

## OVERVIEW
Search engine base classes and abstractions. Actual implementations (USearch, Tantivy) are in parent `infrastructure/` directory.

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Base class | search/base.py | SearchEngineBase (legacy ISearchBackend defaults) |
| Live engines | infrastructure/usearch_engine.py, tantivy_engine.py | Sync tuple search used by SearchPipeline |
| Live pipeline | application/memory/search_strategies.py | SearchPipeline + SearchContext + SearchResult |

## CONVENTIONS

**Default Implementations** - SearchEngineBase provides empty defaults for the unused ISearchBackend protocol.

**Lifecycle Management** - ensure_initialized() for lazy init, is_ready() for health, commit() to persist, close() to release.

**Live search contract** - USearch/Tantivy expose sync ``search()`` returning tuples. Do not wrap those in ``ISearchResult``.

## ANTI-PATTERNS

- Never treat ISearchBackend / ISearchPipeline as the MemoryManager search API
- Never skip ensure_initialized() before index operations
- Never ignore commit() errors
- Never use triple single quotes; use """ only

## NOTES

Concrete USearch and Tantivy engines remain in the parent directory.
