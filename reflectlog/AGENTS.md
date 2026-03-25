#SM|Agent Guidelines for reflectlog/

## OVERVIEW
MCP server providing persistent, project-based semantic memory storage. Combines USearch + Tantivy with RRF fusion and optional LLM reranking.

## STRUCTURE

```
reflectlog/
├── core/              # Protocol definitions (ISearchBackend, IReranker, IMemoryStore)
├── application/       # Business logic
│   ├── memory/        # MemoryManager, search/add pipelines, fusion
│   ├── tools/         # MCP tool implementations (add, search, get_all, remove)
│   ├── config/        # Config dataclass, validation, prompts
│   └── utils/         # Logging, metrics, retry, circuit breaker
├── infrastructure/    # USearchEngine, TantivyEngine, LLM reranker, embeddings
├── plugins/          # Plugin discovery, registry, loading
└── utility/          # Platform utilities
```

## WHERE TO LOOK

| Task | Location | Notes |
|-------|-----------|--------|
| Memory operations | `application/memory/manager.py` | Facade for all memory ops |
| Search pipeline | `application/memory/search_strategies.py` | 4-step hybrid search with RRF |
| Add pipeline | `application/memory/add_phases.py` | 3-phase parallel add with smart replacement |
| Config | `application/config/settings.py` | 60+ env vars, dataclass |
| Protocol interfaces | `core/` | ISearchBackend, IReranker, IMemoryStore |
| Infrastructure | `infrastructure/` | USearch, Tantivy, LLM providers |

## CONVENTIONS

**Triple Double Quotes** - Docstrings use `"""` not `'''`. Enforced by ruff `docstring-quotes = "double"`.

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
