# Agent Guidelines for reflectlog/application/memory/reranking/

## OVERVIEW
Batch min-max score normalization to [0,1] with exponential recency decay for rerankers.

## WHERE TO LOOK

| Function | Normalizes | Description |
|----------|------------|-------------|
| normalize_reranker_scores | Reranker scores to [0,1] | Batch min-max; best=1.0, worst=0.0 |
| apply_threshold_with_safety_net | Filtered results | Threshold with guaranteed min_results |
| calculate_recency_factor | Timestamp to factor | exp(-rate * hours_old), 1.0=newest |
| apply_recency_decay | Normalized scores | score * exp(-rate * hours), re-sorts |

## ANTI-PATTERNS

- Never normalize individually - batch normalization required for relative scores
- Never apply recency decay before normalization
- Never skip normalization - threshold semantics require [0,1] range
- Never return empty results when safety net can provide minimum results
