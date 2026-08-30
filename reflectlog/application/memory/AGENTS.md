# Memory Package

**Generated:** 2026-08-30
**Commit:** 062b44f
**Branch:** develop

## OVERVIEW
`MemoryManager` facade for add, search, delete, and journal replay. Persist is batch-only — there is no `_add_memory` helper.

## STRUCTURE
```
memory/
├── manager.py                 # Locks, public API, lazy CE / SmartReplacer
├── add_phases.py              # 3-phase add (dedup → replace → store)
├── search_strategies.py       # SearchPipeline, SearchContext, SearchResult
├── engine_factory.py          # Engine + fusion construction
├── match_utils.py             # SQLite identity + Tantivy query escape
├── replacement_recovery.py    # Restart converge of unfinished intents
├── fusion/                    # RanxFusionEngine (see child guide)
└── reranking/                 # Pointer only (see child guide)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Public API | `manager.py` | `_write_lock` then `_lock`; `delete_memories` → `list[str]` |
| Persist / replace | `add_phases.py` | `_embed_for_persist` then lock; NEW then OLD |
| Restart converge | `replacement_recovery.py` | `reconcile_pending_replacements`; later-write-wins |
| Exact identity | `match_utils.has_exact_match` | SQLite `(workspace_id, content)` only |
| Hybrid search | `search_strategies.py` | Parallel backends → raw RRF → CE if >1 |

## CONVENTIONS

- Phases implement `IDuplicateDetectionPhase`, `IReplacementDetectionPhase`, `IStoragePhase`. Construct `SearchPipeline` with both engines, a `FusionEngine`, `Config`, logger, and optional manager for lazy CE.
- Embed outside `_write_lock`. Phase 3: record journal → persist NEW + commit → delete OLD → extra HNSW/USearch commit → mark complete only if indexes agree.
- Identity is SQLite unique `(workspace_id, content)`. Tantivy `en_stem` is not exact match; `has_exact_match` does not consult FTS.
- Search: parallel USearch + Tantivy, raw RRF (fusion gate threshold 0.0), skip CE when ≤1 hit. Recency runs after CE normalize + threshold, not in this package.
- Dual-backend search outage (or semantic error + empty FTS) raises `SearchError`. Tantivy delete shortfall after USearch success raises `InconsistentStateError`.
- Mock APIs belong on protocols. Do not invent methods via MagicMock auto-attrs.

## ANTI-PATTERNS

- No `_add_memory` / single-item persist path; use `_add_memories_batch`.
- Do not treat Tantivy phrase/stem hits as identity.
- Do not apply recency before CE normalize + threshold.
- Do not compact Tantivy inside `delete` / `delete_batch`.
- Do not skip pipeline stages for direct engine access from tools.

## NOTES

`fusion/` and `reranking/` have their own guides. Scoring helpers live in `utility/scoring.py`.
