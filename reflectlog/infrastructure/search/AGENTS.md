# Agent Guidelines for reflectlog/infrastructure/search/

**Generated:** 2026-08-30
**Commit:** 062b44f
**Branch:** develop

## OVERVIEW
Empty package marker. Live engines stay FLAT in parent `infrastructure/`.

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Semantic | `../usearch_engine.py` | HNSW + SQLite SoT |
| FTS | `../tantivy_engine.py` | Tombstone+commit |
| Identity | `../memory_store.py` | `get_all` SoT |
| Pipeline | `application/memory/search_strategies.py` | SearchPipeline + CE skip ≤1 |

## CONVENTIONS

- Sync `search()` returns tuples.
- Search `OSError` → `SearchError`, never `[]`.
- HNSW fail-closed if SQLite missing/empty/unreadable.
- SQLite insert then vectors; rollback HNSW+rows on embed/index fail.
- Tantivy delete = tombstone+commit. `compact()` is maintenance only.
- CE skip ≤1 is SearchPipeline, not `CrossEncoderReranker.rerank`.
- Factories: `from_config()`, not `from_app_config()`.

## ANTI-PATTERNS

- Never treat this package as the engine home.
- Never swallow search `OSError` as `[]`.
- Never compact Tantivy on delete.
- Never load HNSW without readable SQLite.
- Never skip `ensure_initialized()` / ignore `commit()` errors.
- No `getattr` / `optional_attr` / `from_app_config()`.

## NOTES

No engine code here. Use parent `infrastructure/AGENTS.md` for layout.
No extra guides under sibling empty markers `llm/`, `memory/`, `reranking/`.
