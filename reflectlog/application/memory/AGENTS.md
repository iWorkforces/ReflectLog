# Memory Management Layer

**Generated:** 2026-08-29  **Commit:** 7df1375  **Branch:** develop
## OVERVIEW
3-phase add + 4-step search. Embed before `_write_lock`. Search Step 4 is CE or skip.

## STRUCTURE
```
memory/
├── manager.py              # MemoryManager orchestrator
├── add_phases.py            # 3-phase add with dedup + smart replacement
├── search_strategies.py     # 4-step hybrid search with RRF fusion
├── engine_factory.py        # Engine creation + cached embeddings
├── match_utils.py           # Exact match detection
├── fusion/                  # Fusion algorithms (RRF, CombSUM, Borda)
└── reranking/               # Recency decay
```

## WHERE TO LOOK

| Task | Location | Notes |
|-------|-----------|--------|
| 3-phase add | `add_phases.py` | Phase 1: dedup, Phase 2: replace, Phase 3: store |
| 4-step search | `search_strategies.py` | Backend → Fusion → Filter → Rerank |
| Engine factory | `engine_factory.py` | Engine creation + cached embeddings |
| RRF fusion | `fusion/ranx_fusion.py` | Reciprocal Rank Fusion (k=60 default) |
| Score normalization | `utility/scoring.py` | Batch min-max to 0-1 range (JIT-compiled) |
| Recency decay | `utility/scoring.py` | Exponential: score * exp(-rate * hours) |

## CONVENTIONS

**SearchPipeline** - Canonical class is `search_strategies.SearchPipeline`. Construct it with semantic/tantivy engines, a `FusionEngine`, `Config`, logger, and optional memory manager for lazy rerankers. Execute with a `SearchContext`; it returns a `SearchResult`.

**Pluggable Phases** - AddPipeline uses `IDuplicateDetectionPhase`, `IReplacementDetectionPhase`, `IStoragePhase` protocols.

**Embed-then-lock persist** - `_embed_for_persist()` then `_write_lock`. `delete_memories` returns `list[str]` of deleted contents; Tantivy fail → `InconsistentStateError`.
**Identity** - SQLite unique `(workspace_id, content)` only. Tantivy stemming is not exact match.
**CE Threshold Semantics** - When sigmoid is on, do not batch min-max. Gate on calibrated CE scores. `reranker_min_results` keeps at least the best hit.

**RRF Formula** - `score(doc) = sum(1/(k+rank))` with `k=60` default. Raw RRF is used for the fusion gate (default threshold 0.0).

**Recency Decay** - Applied after normalization: `decayed = normalized * exp(-rate * hours_old)`.

**FusionEngine Protocol** - All fusion engines implement `fuse(*result_sets)` returning sorted `(doc, score)` tuples.

**Temporal-Aware Reranking** - Timestamp_map built from search results drives recency factor calculations.

## ANTI-PATTERNS

- Never normalize scores individually - batch normalization required for relative scores
- Never apply recency decay before normalization
- Never treat Tantivy OSError as empty success; dual search outage (or semantic error + empty FTS) raises SearchError
- Never skip pipeline stages for direct engine access
- Never assume fusion scores from different algorithms are comparable
- Never compact Tantivy inside `delete`/`delete_batch`
- Never treat MagicMock `add_batch`/`delete_memories` as present; use `type(obj).__dict__.get(...)`
