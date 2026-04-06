# ReflectLogMCP Application Layer

## OVERVIEW
Orchestration layer implementing 3-phase add pipeline and 4-step search pipeline with MCP tool bindings.

## WHERE TO LOOK

| Task | Location | Notes |
|-------|-----------|--------|
│   ├── memory/        # Memory pipelines + fusion + reranking
| Search pipeline | `memory/search_pipeline.py` | 4-step staged architecture |
| Add pipeline | `memory/add_pipeline.py` | 3-phase parallel execution |
| Fusion algorithms | `memory/fusion/` | RRF, CombSUM, Borda variants |
| MCP tools | `tools/` | FastMCP tool implementations |
| Config management | `config/settings.py` | 60+ env vars, validation |

## CONVENTIONS

**Pipeline Architecture** - SearchPipeline and AddPipeline use pluggable stages/phases. Each stage returns SearchResult for composability.
| Reranking | `application/memory/reranking/` | Score normalization + recency decay
**3-Phase Add** - Phase 1: duplicate detection (parallel batch), Phase 2: smart replacement (LLM checks), Phase 3: sequential storage.

**4-Step Search** - Step 1: parallel dual-search, Step 2: RRF fusion, Step 3: threshold filter, Step 4: LLM/cross-encoder rerank.

**Adaptive Overfetch** - Multiplier adjusts 1.5-3x based on index size.

**Lazy Reranker** - LLM reranker initialized on-demand with double-checked locking pattern.

**Tool Base Class** - All tools extend BaseTool with common logging and error handling.

## ANTI-PATTERNS

- Never bypass pipeline for direct engine access
- Never use synchronous LLM calls in add pipeline
- Never assume both search engines return same number of results
