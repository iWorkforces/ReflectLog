# Memory Management Layer

**Generated:** 2026-04-11  **Commit:** 6f2b0f8  **Branch:** develop
## OVERVIEW
3-phase add pipeline + 4-step search pipeline with RRF fusion and unified score normalization.

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

**Pluggable Stages** - SearchPipeline uses `IBackendExecutor`, `IFusionStage`, `IFilterStage`, `IRerankStage` protocols.

**Pluggable Phases** - AddPipeline uses `IDuplicateDetectionPhase`, `IReplacementDetectionPhase`, `IStoragePhase` protocols.

**Unified Threshold Semantics** - Batch min-max normalization maps diverse reranker ranges (LLM: 0.7-0.9, CrossEncoder: 0.001-0.17) to consistent 0-1 range.

**RRF Formula** - `score(doc) = sum(1/(k+rank))` with `k=60` default, normalized to 0-1.

**Recency Decay** - Applied after normalization: `decayed = normalized * exp(-rate * hours_old)`.

**FusionEngine Protocol** - All fusion engines implement `fuse(*result_sets)` returning sorted `(doc, score)` tuples.

**Temporal-Aware Reranking** - Timestamp_map built from search results drives recency factor calculations.

## ANTI-PATTERNS

- Never normalize scores individually - batch normalization required for relative scores
- Never apply recency decay before normalization
- Never skip pipeline stages for direct engine access
- Never assume fusion scores from different algorithms are comparable
