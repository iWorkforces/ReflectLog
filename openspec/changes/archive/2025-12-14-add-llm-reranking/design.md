# Design: LLM Reranking After RRF Fusion

## Context

OpenMemoriesMCP performs hybrid search combining semantic vector search (USearch) and full-text search (Tantivy), fused via RRF algorithm. While RRF produces reasonable rankings, it's purely algorithmic and can't assess whether a result actually answers the user's query. LLM reranking adds a semantic understanding layer that evaluates each candidate's relevance.

**Stakeholders**: Users performing memory search, AI agents relying on relevant context retrieval.

**Constraints**:
- Must use OpenRouter API (existing provider infrastructure)
- Must be optional (backward compatible with existing behavior)
- Should not block on LLM failures (graceful fallback)
- Should limit latency impact via parallel LLM calls

## Goals / Non-Goals

### Goals
- Add LLM-based relevance scoring after RRF fusion
- Configurable via environment variables (`SEARCH_RERANK`, `LLM_MODEL`, `SEARCH_SCORE_THRESHOLD`)
- Support async parallel scoring for multiple candidates
- Provide structured logging for reranking step
- Graceful fallback to fusion scores if LLM fails

### Non-Goals
- Supporting non-OpenRouter LLM providers (future work)
- Implementing sophisticated prompt engineering or few-shot examples
- Caching LLM rerank scores (ephemeral, query-dependent)
- Training/fine-tuning custom reranking models

## Decisions

### Decision 1: LLMReranker as Infrastructure Class

**What**: Create `openmemories/infrastructure/llm_reranker.py` with `LLMReranker` class.

**Why**:
- Follows existing pattern (`LangchainQwenEmbeddings` in same directory)
- Uses `AsyncOpenAI` client for non-blocking LLM calls
- Clean separation from application logic (MemoryManager just calls reranker)
- Testable with mocked OpenAI client

**Alternatives considered**:
1. Inline LLM calls in MemoryManager - Rejected: violates separation of concerns
2. Add to fusion module - Rejected: reranking is post-fusion, different concern
3. Use LangChain LLM abstractions - Rejected: unnecessary dependency, OpenAI client is sufficient

### Decision 2: Pointwise Reranking Strategy

**What**: Score each document independently with the LLM, then sort by scores.

**Why**:
- Simple to implement and understand
- Parallelizable (all LLM calls can run concurrently)
- Works with any result set size
- LLM outputs single float score (0.0-1.0)

**Alternatives considered**:
1. Pairwise comparison (compare doc pairs) - Rejected: O(n²) LLM calls, expensive
2. Listwise ranking (LLM ranks entire list) - Rejected: context length limits, complex prompts
3. Batch scoring (single prompt, multiple docs) - Rejected: complex output parsing, less reliable

### Decision 3: Score Extraction via Structured Output

**What**: Use `response_format={"type": "json_object"}` and parse JSON output.

**Why**:
- More reliable than regex extraction from free-form text
- LLM constrained to valid JSON output
- Clear failure mode (invalid JSON = fallback to fusion score)
- Compatible with OpenRouter/OpenAI models

**Prompt Template**:
```
You are a relevance scoring system. Score how relevant a document is to a query.
Output ONLY a JSON object with a single "score" field containing a number between 0.0 and 1.0.

Query: "{query}"
Document: "{document}"

Example output: {"score": 0.85}
```

### Decision 4: Concurrency Control with Semaphore

**What**: Use `anyio.Semaphore` to limit concurrent LLM calls (default: 5).

**Why**:
- Prevents overwhelming the LLM API with too many parallel requests
- Same pattern used in `LangchainQwenEmbeddings.aembed_documents()`
- Configurable via `RERANK_MAX_CONCURRENCY` environment variable

### Decision 5: Graceful Fallback on LLM Failure

**What**: If LLM scoring fails for a document, retain its fusion score and log warning.

**Why**:
- Single document failure shouldn't fail entire search
- Fusion scores are reasonable fallback
- Enables partial degradation under API issues
- Important for production reliability

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MemoryManager.search()                       │
├─────────────────────────────────────────────────────────────────────┤
│  1. Parallel search (USearch + Tantivy)                             │
│  2. RRF Fusion (RanxFusionEngine)                                   │
│  3. Fusion threshold filtering                                       │
│  4. LLM Reranking (NEW)  ←─────────────────────────────────────────┐│
│  5. Return results                                                  ││
└─────────────────────────────────────────────────────────────────────┘│
                                                                       │
┌─────────────────────────────────────────────────────────────────────┐│
│                    LLMReranker (infrastructure)                      ││
├─────────────────────────────────────────────────────────────────────┤│
│  - AsyncOpenAI client (HTTP/2)                                      ││
│  - SCORING_PROMPT template                                          ││
│  - Parallel scoring with semaphore                                  ││
│  - JSON output parsing                                              ││
│  - Error handling with fallback                                     ││
└────────────────────────────────────────────────────────────◀────────┘│
                                                              │
                                                   Calls via asyncify()
```

### Class Design

```python
# openmemories/infrastructure/llm_reranker.py

@dataclass(frozen=True)
class LLMRerankerConfig:
    """Configuration for LLM reranking."""
    api_key: str              # OpenRouter API key
    base_url: str             # OpenRouter base URL
    model: str                # LLM model (e.g., x-ai/grok-4.1-fast)
    score_threshold: float    # Min score to keep (0.0-1.0)
    max_concurrency: int      # Max parallel LLM calls

class LLMReranker(BaseModel):
    """LLM-based document reranker."""

    config: LLMRerankerConfig
    logger: Any = None

    _client: AsyncOpenAI | None = PrivateAttr(default=None)

    async def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """Rerank candidates by LLM relevance scores.

        Args:
            query: Search query string.
            candidates: List of (message, fusion_score) tuples.

        Returns:
            Reranked list of (message, llm_score) tuples,
            filtered by score_threshold, sorted descending.
        """
```

### Configuration Schema

```python
# In settings.py Config dataclass

# LLM Reranking settings
llm_model: str = "x-ai/grok-4.1-fast"
search_rerank: bool = True
search_score_threshold: float = 0.5
rerank_max_concurrency: int = 5
```

### Integration in MemoryManager

```python
# In MemoryManager.__init__()
if self.config.search_rerank:
    reranker_config = LLMRerankerConfig(
        api_key=config.openrouter_api_key.get_secret_value(),
        base_url=config.openrouter_base_url,
        model=config.llm_model,
        score_threshold=config.search_score_threshold,
        max_concurrency=config.rerank_max_concurrency,
    )
    self._reranker = LLMReranker(config=reranker_config, logger=self.logger)

# In MemoryManager.search() after fusion filtering
if self.config.search_rerank and self._reranker:
    self.logger.info("🤖 STEP 4: LLM Reranking...")
    hybrid_results = await self._reranker.rerank(query, hybrid_results)
```

## Risks / Trade-offs

### Risk 1: Latency Impact
- **Risk**: LLM API calls add 100-500ms per search
- **Mitigation**: Parallel LLM calls, configurable concurrency, optional feature
- **Monitoring**: Log rerank duration separately

### Risk 2: API Cost
- **Risk**: Additional LLM API costs per search
- **Mitigation**: Default to disabled in cost-sensitive environments
- **Note**: Cost scales with search volume × candidates per search

### Risk 3: LLM Reliability
- **Risk**: LLM API failures could degrade search
- **Mitigation**: Graceful fallback to fusion scores, retry logic, timeout handling

### Risk 4: Score Inconsistency
- **Risk**: LLM scores may vary between identical queries
- **Mitigation**: Use temperature=0 for deterministic outputs
- **Acceptance**: Minor score variance acceptable for search ranking

## Migration Plan

1. **Phase 1**: Add configuration (backward compatible, default `SEARCH_RERANK=true`)
2. **Phase 2**: Implement LLMReranker infrastructure class
3. **Phase 3**: Integrate into MemoryManager.search()
4. **Phase 4**: Add unit and integration tests
5. **Phase 5**: Update documentation (CLAUDE.md, README)

**Rollback**: Set `SEARCH_RERANK=false` to disable feature entirely.

## Open Questions

1. ~~Should batch scoring be supported for efficiency?~~
   - **Decision**: Start with pointwise, add batching if latency becomes issue

2. ~~Which LLM model works best for reranking?~~
   - **Decision**: Default to `x-ai/grok-4.1-fast` (fast, cheap, good reasoning)

3. Should rerank scores replace or augment fusion scores?
   - **Proposal**: Replace - LLM scores are more meaningful for relevance
