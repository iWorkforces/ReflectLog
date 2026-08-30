# Fusion

**Generated:** 2026-08-30
**Commit:** 062b44f
**Branch:** develop

## OVERVIEW
`RanxFusionEngine` combines ranked lists. RRF is Numba on raw `1/(k+rank)` scores. Other methods go through ranx.

## STRUCTURE
```
fusion/
├── __init__.py          # create_fusion_engine()
├── base.py              # FusionEngine protocol
└── ranx_fusion.py       # RanxFusionEngine
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Protocol | `base.py` | `fuse(*result_sets) -> list[tuple[str, float]]` |
| RRF (default) | `ranx_fusion.py` | `_fuse_rrf_numba`; `k=60`; first rank kept |
| sum / mnz / max / bordafuse | `ranx_fusion.py` | ranx path; not `borda` |
| Factory | `__init__.py` | `create_fusion_engine()` |

## CONVENTIONS

- Methods: `rrf` (default), `sum`, `mnz`, `max`, `bordafuse`. There is no `borda` method (`borda` is a normalization only).
- RRF, weighted or not, stays on the raw Numba scale. Do not min-max RRF before the fusion gate.
- One non-empty list: convert to/from a ranx `Run` and return original backend scores. Never min-max a single list (a 0.8-style gate then drops near-ties).
- Two or more lists: ranx methods may min-max their *output*; RRF still returns raw scores.
- Ranx averages duplicate scores inside one list. Numba RRF keeps the first rank and ignores later dupes.
- `weights` on a ranx method that rejects them raises `RuntimeError` from `TypeError` — fail closed, no unweighted fallback.

## ANTI-PATTERNS

- Never call ranx.fuse outside `FusionEngine`.
- Never min-max a single backend list.
- Never treat fused scores as comparable across methods.
- Never rename `bordafuse` to `borda`.

## NOTES

Search Step 2 owns when fusion runs. The fusion-gate threshold is 0.0 on raw RRF. Recency and CE live outside this package.
