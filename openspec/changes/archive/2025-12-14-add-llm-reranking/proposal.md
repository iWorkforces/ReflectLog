# Change: Add LLM Reranking After RRF Fusion

## Why

The current hybrid search pipeline combines semantic (USearch) and full-text (Tantivy) results using RRF fusion, but fusion scores are purely algorithmic and don't account for semantic relevance to the query. Adding LLM-based reranking after RRF fusion will significantly improve search result quality by using an LLM to score each candidate's actual relevance to the query, filtering out false positives that pass fusion threshold but aren't truly relevant.

## What Changes

- **Configuration**: Add `LLM_MODEL`, `SEARCH_RERANK`, and `SEARCH_SCORE_THRESHOLD` settings to `Config`
- **Infrastructure**: Create new `LLMReranker` class in `openmemories/infrastructure/llm_reranker.py` using `openai.AsyncOpenAI`
- **Prompts**: Add `SCORING_PROMPT` template to `openmemories/application/config/prompts.py`
- **Memory Manager**: Integrate LLM reranking as Step 4 in `MemoryManager.search()` after RRF fusion filtering
- **Search Flow**: Update search pipeline to optionally rerank results before returning

### Search Pipeline After Change

```
1. Parallel search (USearch + Tantivy)
2. RRF Fusion (combine rankings)
3. Fusion threshold filtering (FUSION_RANKING_THRESHOLD)
4. **NEW: LLM Reranking** (when SEARCH_RERANK=true)
   - Score each result against query using LLM
   - Filter by SEARCH_SCORE_THRESHOLD
   - Re-sort by LLM relevance scores
5. Return top `limit` results
```

## Impact

- **Affected specs**: Creates new `llm-reranking` capability
- **Affected code**:
  - `openmemories/application/config/settings.py` - Add 3 new config fields
  - `openmemories/application/config/prompts.py` - Add SCORING_PROMPT
  - `openmemories/infrastructure/llm_reranker.py` - New file
  - `openmemories/infrastructure/__init__.py` - Export LLMReranker
  - `openmemories/application/memory/manager.py` - Integrate reranking step
  - `tests/unit/infrastructure/test_llm_reranker.py` - New tests
  - `tests/unit/application/test_memory_manager.py` - Update search tests
- **Dependencies**: Uses existing `openai` package (already in pyproject.toml)
- **Performance**: Adds ~100-500ms latency per search when reranking enabled (LLM API call)
- **Backward compatible**: Yes - reranking is optional, disabled by default with `SEARCH_RERANK=false`
