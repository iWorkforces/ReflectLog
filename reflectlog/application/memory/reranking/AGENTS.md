# Reranking (pointer)

**Generated:** 2026-08-30
**Commit:** 062b44f
**Branch:** develop

## OVERVIEW
This directory is a pointer, not an implementation. Score math lives in `reflectlog/utility/scoring.py`. Search Step 4 calls `CrossEncoderReranker`; post-CE normalize / threshold / recency run in `infrastructure/reranker_post_processor.py`.

## STRUCTURE
```
reranking/
└── __init__.py          # Package marker only — no scoring exports
```

## WHERE TO LOOK

| Need | Location | Notes |
|------|----------|-------|
| Batch min-max | `utility/scoring.py` | `normalize_reranker_scores` |
| CE / fusion gate | `utility/scoring.py` | `apply_threshold_with_safety_net` |
| Recency factor | `utility/scoring.py` | `calculate_recency_factor` = `exp(-rate * hours)` |
| Recency apply | `utility/scoring.py` | `apply_recency_decay` re-sorts |
| Step 4 skip | `search_strategies.py` | Skip CE when ≤1 hit |
| Apply order | `reranker_post_processor.py` | normalize → threshold → recency |

## CONVENTIONS

- Do not add scoring functions here. Import from `utility/scoring.py`.
- Recency runs only after CE normalize + threshold. Never decay first; never gate on decayed scores.
- Threshold semantics assume a [0, 1] batch. Normalize the whole list, not each score.
- `reranker_min_results` keeps at least the best hit when the gate would empty the list.
- Pipeline order: raw RRF (threshold 0.0) → CE if >1 hit → recency → `top_k`.
- An empty `timestamp_map` disables recency; do not invent stamps.

## ANTI-PATTERNS

- Never implement scoring in this package.
- Never apply recency before CE normalize + threshold.
- Never skip batch normalization so a raw CE score can be compared to a 0–1 gate.
- Never return empty when the safety net can keep `min_results`.
- Never move Numba RRF helpers here; fusion owns those imports.

## NOTES

Use the parent memory guide for `SearchPipeline` construction and CE skip rules. This folder stays empty on purpose so `utility/` remains importable from infrastructure without a cycle.

## LIMITS

Do not add modules under `reranking/`. Do not re-export scoring symbols from `__init__.py`.
