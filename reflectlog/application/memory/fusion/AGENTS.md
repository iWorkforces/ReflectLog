# Fusion

**Generated:** 2026-09-03
**Commit:** e401dbc
**Branch:** develop

## OVERVIEW
`RanxFusionEngine` fuses ranked lists. RRF is Numba `1/(k+rank)`. Other methods go through ranx.

## STRUCTURE
```
fusion/
├── __init__.py      # create_fusion_engine()
├── base.py          # FusionEngine protocol
└── ranx_fusion.py   # RanxFusionEngine
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Protocol | `base.py` | `fuse(*result_sets) -> list[tuple[str, float]]` |
| RRF | `ranx_fusion.py` | `_fuse_rrf_numba`; `k=60`; first rank kept |
| ranx path | `ranx_fusion.py` | `sum` / `mnz` / `max` / `bordafuse` — not `borda` |
| Factory | `__init__.py` | `create_fusion_engine()` |

## CONVENTIONS

- Methods: `rrf` (default), `sum`, `mnz`, `max`, `bordafuse`. `borda` is a normalization only.
- RRF stays on the raw Numba scale. Do not min-max RRF before the fusion gate.
- One non-empty list: `_one_list_results` averages dupes and returns original backend scores. Never min-max a single list.
- Two or more lists: ranx methods may min-max output; RRF still returns raw scores.
- Ranx averages duplicate scores in one list. Numba RRF keeps the first rank and ignores later dupes.
- `weights` on a ranx method that rejects them raises `RuntimeError` from `TypeError` — fail closed, no unweighted fallback.
- Lazy ranx load: `_ranx_run: type[Run] | None`, `_ranx_fuse: Callable[..., Run] | None`. Typed `Run`, not `Any`. `_convert_to_run` / `_convert_from_run` take/return `Run`.

## ANTI-PATTERNS

- Never call ranx.fuse outside `FusionEngine`.
- Never min-max a single backend list.
- Never type ranx `Run` as `Any`.
- Never treat fused scores as comparable across methods.
- Never rename `bordafuse` to `borda`.

## NOTES

Search Step 2 owns when fusion runs. Gate is 0.0 on raw RRF. Recency and CE live outside this package.
