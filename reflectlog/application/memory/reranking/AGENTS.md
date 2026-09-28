# Reranking (pointer)

**Generated:** 2026-09-28
**Commit:** aaebf3c
**Branch:** develop

## OVERVIEW
Pointer only. Score math lives in `reflectlog/utility/scoring.py`. Search Step 4 calls `CrossEncoderReranker`; post-CE normalize / threshold / recency run in `infrastructure/reranker_post_processor.py`.

## STRUCTURE
```
reranking/
└── __init__.py      # Marker — no scoring exports
```

## WHERE TO LOOK

| Need | Location | Notes |
|------|----------|-------|
| Batch min-max | `utility/scoring.py` | `normalize_reranker_scores`; single/equal → `1.0` |
| CE / fusion gate | `utility/scoring.py` | `apply_threshold_with_safety_net` |
| Recency factor | `utility/scoring.py` | `calculate_recency_factor` = `exp(-rate * hours)` |
| Recency apply | `utility/scoring.py` | `apply_recency_decay` re-sorts |
| Apply order | `reranker_post_processor.py` | normalize → threshold → recency |

## CONVENTIONS

- Do not add scoring functions here. Import from `utility/scoring.py`.
- Default CE `normalize=True` applies sigmoid and forces `batch_normalize=False`. Batch min-max is the off-sigmoid path.
- Recency only after CE normalize + threshold. Never decay first; never gate on decayed scores.
- Threshold assumes a [0, 1] batch. Normalize the whole list, not each score.
- `reranker_min_results` keeps at least the best hit when the gate would empty the list.
- Empty `timestamp_map` disables recency; do not invent stamps.

## ANTI-PATTERNS

- Never implement scoring in this package.
- Never apply recency before CE normalize + threshold.
- Never compare a raw CE logit to the 0–1 gate. Sigmoid is the default. Batch min-max runs only when sigmoid is off.
- Never return empty when the safety net can keep `min_results`.
- Never move Numba RRF helpers here; fusion owns those imports.

## NOTES

Folder stays empty so `utility/` remains importable from infrastructure without a cycle.

## LIMITS

Do not add modules under `reranking/`. Do not re-export scoring symbols from `__init__.py`.
