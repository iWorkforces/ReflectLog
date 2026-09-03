# Agent Guidelines for reflectlog/infrastructure/search/

**Generated:** 2026-09-03
**Commit:** e401dbc
**Branch:** develop

## OVERVIEW
Empty package marker. Live engines stay FLAT in parent `infrastructure/`.

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Semantic | `../usearch_engine.py` | HNSW + SQLite SoT |
| FTS | `../tantivy_engine.py` | Tombstone+commit |
| Identity | `../memory_store.py` | `get_all` SoT |
| Pipeline | `application/memory/search_strategies.py` | SearchPipeline |

## CONVENTIONS

- No engine modules in this package. Import from parent.
- `__init__.py` is a docstring marker only.

## ANTI-PATTERNS

- Never treat this package as the engine home.
- Never add USearch / Tantivy / MemoryStore implementations here.
- Never add extra guides under sibling empty markers `llm/`, `memory/`, `reranking/`.

## NOTES

Use parent `infrastructure/AGENTS.md` for engine layout and fail-closed search rules.
