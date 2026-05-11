# ReflectLogMCP Application Layer

**Generated:** 2026-04-11  **Commit:** 6f2b0f8  **Branch:** develop

## OVERVIEW
Orchestration layer implementing 3-phase add pipeline and 4-step search pipeline with MCP tool bindings.

## WHERE TO LOOK

| Task | Location | Notes |
|-------|-----------|--------|
| Search pipeline | `memory/search_strategies.py` | 4-step staged architecture |
| Add pipeline | `memory/add_phases.py` | 3-phase parallel execution |
| Fusion algorithms | `memory/fusion/` | RRF, CombSUM, Borda variants |
| MCP tools | `tools/` | FastMCP tool implementations |
| Config management | `config/settings.py` | 60+ env vars, validation |
| Reranking | `memory/reranking/` | Score normalization + recency decay |

## CONVENTIONS

**Pipeline Architecture** - SearchPipeline and AddPipeline use pluggable stages/phases. Each stage returns SearchResult for composability.
**3-Phase Add** - Phase 1: duplicate detection (parallel batch), Phase 2: smart replacement (LLM checks), Phase 3: sequential storage.

**4-Step Search** - Step 1: parallel dual-search, Step 2: RRF fusion, Step 3: threshold filter, Step 4: LLM/cross-encoder rerank.

**Adaptive Overfetch** - Multiplier adjusts 1.5-3x based on index size.

**Lazy Reranker** - LLM reranker initialized on-demand with double-checked locking pattern.

**Canonical Type Locations** - Core domain types (`MemoryRecord`, `Embeddings`, `ISemanticSearchEngine`) live in `core/types.py`. Application-layer types (`SearchResult`, `MemoryList`, `ToolResult`, `LogContext`) live in `application/types.py`. Don't mix them up.

**Tool Base Class** - All tools extend BaseTool with common logging and error handling.

## ANTI-PATTERNS

- Never bypass pipeline for direct engine access
- Never use synchronous LLM calls in add pipeline
- Never assume both search engines return same number of results
