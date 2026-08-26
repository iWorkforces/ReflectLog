# Fusion

**Generated:** 2026-08-26  **Commit:** 95567fa  **Branch:** develop

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

- RanxFusionEngine supports multiple methods via ranx library: rrf, sum, mnz, max, borda
- Factory: create_fusion_engine(method, normalization, k)
- Scores normalized to 0-1 range

## ANTI-PATTERNS

- Never bypass FusionEngine protocol for direct ranx calls
- Never mix normalized and raw scores across engines
- Never assume fusion scores are comparable across different methods
