# reflectlog Package

**Generated:** 2026-08-26  **Commit:** 95567fa  **Branch:** develop

## OVERVIEW
MCP server providing persistent, project-based semantic memory storage. Combines USearch + Tantivy with RRF fusion and optional LLM reranking.

## STRUCTURE

```
reflectlog/
├── core/              # Protocols + canonical types (ISemanticSearchEngine, IReranker, IMemoryStore)
│                      # types.py: ISemanticSearchEngine, MemoryRecord, Embeddings, IArchiveMemoryStore
├── application/       # Business logic
│   ├── memory/        # MemoryManager, search/add pipelines, fusion
│   ├── tools/         # MCP tool implementations (add, search, get_all, remove)
│   ├── config/        # Config dataclass, validation, prompts
│   └── utils/         # Logging, metrics, retry, circuit breaker
├── infrastructure/    # USearchEngine, TantivyEngine, LLM reranker, embeddings
├── plugins/          # Plugin discovery, registry, loading
└── utility/          # Platform utilities, scoring.py (JIT-compiled RRF, normalization, filtering)
```

## WHERE TO LOOK

| Task | Location | Notes |
|-------|-----------|--------|
| Memory operations | `application/memory/manager.py` | Facade for all memory ops |
| Search pipeline | `application/memory/search_strategies.py` | 4-step hybrid search with RRF |
| Add pipeline | `application/memory/add_phases.py` | 3-phase parallel add with smart replacement |
| Config | `application/config/settings.py` | 60+ env vars, dataclass |
| Protocol interfaces | `core/` | ISemanticSearchEngine, IReranker, IMemoryStore, types.py |
| Infrastructure | `infrastructure/` | USearch, Tantivy, LLM providers |

## CONVENTIONS

**Factory Pattern** - Use `from_config()` class methods (not `from_app_config()`). Applied throughout for engine, reranker, and fusion creation.

**Canonical Type Locations** - Core domain types live in `core/types.py` (MemoryRecord, Embeddings, ISemanticSearchEngine, IArchiveMemoryStore). Pipeline `SearchContext` / `SearchResult` live in `application/memory/search_strategies.py`.

**Triple Double Quotes** - Docstrings use `"""` not `'''`. Enforced by ruff.

**Adaptive Overfetch** - Multiplier adjusts 1.5-3x based on index size.

**Temporal Scoring** - Recency decay with configurable rate (`0.01` = ~69hr half-life).

**Plugin Discovery** - Three mechanisms: entry points, directory scan, static registration.

**Exception Hierarchy** - All custom exceptions chain with `from e` to preserve tracebacks.

## ANTI-PATTERNS

- Never use triple single quotes in docstrings - use `"""` only
- Never suppress type errors in CI - build fails on type violations
- Never use bare `except:` - catch specific exceptions
- Never use `@type: ignore`, `@ts-expect-error`, `as any` - type safety strict
- Never acquire locks in wrong order - always `_write_lock` before `_lock`
- Never use legacy typing imports (`List`, `Optional`, `Union`) - use native syntax
