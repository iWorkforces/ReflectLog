# Reranking (pointer)

**Generated:** 2026-09-03
**Commit:** e401dbc
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
| Batch min-max | `utility/scoring.py` | `normalize_reranker_scores` |
| CE / fusion gate | `utility/scoring.py` | `apply_threshold_with_safety_net` |
| Recency factor | `utility/scoring.py` | `calculate_recency_factor` = `exp(-rate * hours)` |
| Recency apply | `utility/scoring.py` | `apply_recency_decay` re-sorts |
| Apply order | `reranker_post_processor.py` | normalize → threshold → recency |

## CONVENTIONS

- Do not add scoring functions here. Import from `utility/scoring.py`.
- Recency only after CE normalize + threshold. Never decay first; never gate on decayed scores.
- Threshold assumes a [0, 1] batch. Normalize the whole list, not each score.
- `reranker_min_results` keeps at least the best hit when the gate would empty the list.
- Empty `timestamp_map` disables recency; do not invent stamps.

## ANTI-PATTERNS

- Never implement scoring in this package.
- Never apply recency before CE normalize + threshold.
- Never skip batch normalization so a raw CE score can be compared to a 0–1 gate.
- Never return empty when the safety net can keep `min_results`.
- Never move Numba RRF helpers here; fusion owns those imports.

## NOTES

Folder stays empty so `utility/` remains importable from infrastructure without a cycle.

## LIMITS

Do not add modules under `reranking/`. Do not re-export scoring symbols from `__init__.py`.
