# Agent Guidelines for reflectlog/application/memory/reranking/

**Generated:** 2026-08-29  **Commit:** 7df1375  **Branch:** develop

## OVERVIEW
Recency decay scoring for rerankers. Score normalization and filtering functions live in `utility/scoring.py`.

## WHERE TO LOOK

| Function | Location | Description |
|----------|----------|-------------|
| normalize_reranker_scores | `utility/scoring.py` | Batch min-max normalization to [0,1] |
| apply_threshold_with_safety_net | `utility/scoring.py` | Threshold with guaranteed min_results |
| calculate_recency_factor | `utility/scoring.py` | exp(-rate * hours_old), 1.0=newest |
| apply_recency_decay | `utility/scoring.py` | score * exp(-rate * hours), re-sorts |

## ANTI-PATTERNS

- Never normalize individually - batch normalization required for relative scores
- Never apply recency decay before normalization
- Never gate on decayed scores; CE thresholds first, then recency, then top_k
- Never skip normalization - threshold semantics require [0,1] range
- Never return empty results when safety net can provide minimum results

## NOTES

This package owns only the reranking integration point. The scoring functions remain in `reflectlog/utility/scoring.py` to preserve the dependency boundary.

Use the parent memory guide for pipeline ordering and configuration ownership.

## LIMITS

Do not move scoring helpers into this package.
