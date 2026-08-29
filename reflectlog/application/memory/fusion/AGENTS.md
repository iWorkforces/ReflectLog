# Fusion

**Generated:** 2026-08-29  **Commit:** 7df1375  **Branch:** develop

## OVERVIEW
Fusion engines for combining search results using the ranx library.

## STRUCTURE
```
fusion/
├── __init__.py          # Package exports and factory function
├── base.py              # FusionEngine protocol
└── ranx_fusion.py       # RanxFusionEngine implementation
```

## WHERE TO LOOK

| Task | Location | Notes |
|-------|----------|-------|
| FusionEngine protocol | base.py | Interface definition |
| RanxFusionEngine | ranx_fusion.py | ranx library wrapper |
| Factory function | __init__.py | create_fusion_engine() |

## CONVENTIONS

- Methods: `rrf` (default), `sum`, `mnz`, `max`, `bordafuse` (not `borda`).
- RRF (weighted or not) → Numba. Other methods → ranx; `TypeError` on `weights` fails closed.
- One non-empty list: convert to/from run and **return original scores** (no min-max).
- Ranx path averages duplicate scores; Numba RRF keeps first rank.

## ANTI-PATTERNS

- Never bypass FusionEngine for raw ranx
- Never min-max a single backend list (threshold 0.8 drops near-ties)
- Never assume fusion scores are comparable across methods
