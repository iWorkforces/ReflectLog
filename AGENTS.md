# ReflectLogMCP Knowledge Base

**Generated:** 2026-03-25
**Commit:** 7854528
**Branch:** develop

## OVERVIEW

MCP server providing persistent, project-based semantic memory storage for AI agents. Combines USearch vector search with Tantivy full-text search using RRF fusion, with optional LLM/cross-encoder reranking and smart memory replacement.

## STRUCTURE

```
./
├── reflectlog/              # Main package (~80 .py files, 21k lines)
│   ├── core/              # Protocol definitions (8 files, 2k lines)
│   ├── application/         # Business logic (41 files, 11k lines)
│   │   ├── memory/        # Memory pipelines + fusion + reranking
│   │   ├── tools/         # MCP tool implementations
│   │   ├── config/        # Settings, validation, prompts, presets
│   │   └── utils/         # Logging, metrics, retry, circuit breaker
│   ├── infrastructure/      # External integrations (16 files, 5.7k lines)
│   ├── plugins/           # Plugin system (4 files, 1.1k lines)
│   └── utility/           # Platform utilities (8 files)
├── tests/                # Unit + integration + load + security (71 files, 34k lines)
├── stubs/               # Type stubs for third-party libs (14 .pyi files)
├── indexes/              # Persistent index data (gitignored)
├── scripts/              # Build/dev scripts + git hooks
└── *.sh                  # Wrapper scripts (type-check, lint, unittest, server)
```

## WHERE TO LOOK

| Task | Location | Notes |
|-------|-----------|--------|
| Memory operations | `reflectlog/application/memory/manager.py` | 3-phase add, 4-step search pipeline |
| Search configuration | `reflectlog/application/config/settings.py` | 60+ env vars, factory pattern |
| Protocol interfaces | `reflectlog/core/` | All abstractions defined here |
| Infrastructure wrappers | `reflectlog/infrastructure/` | USearch, Tantivy, LLM providers |
| Build commands | `start-type-check.sh`, `start-lint.sh`, `start-unittest.sh` | Custom wrappers, no CI/CD |
| Utilities | `reflectlog/application/utils/` | Logging, metrics, retry, circuit breaker, security |

## CODE MAP

| Symbol | Type | Location | Refs | Role |
|---------|------|----------|--------|------|
| MemoryManager | Class | application/memory/manager.py | High | Facade for all memory operations |
| Config | Dataclass | application/config/settings.py | High | Centralized configuration |
| ISearchBackend | Protocol | core/search.py | Medium | Abstract search engine interface |
| USearchEngine | Class | infrastructure/usearch_engine.py | Medium | Semantic vector search backend |
| TantivyEngine | Class | infrastructure/tantivy_engine.py | Medium | Full-text search backend |
| ConfigAdapter | Class | core/config_adapters.py | High | Config-to-protocol adapter |
| FusionEngine | Protocol | application/memory/fusion/base.py | Medium | Fusion algorithm interface |
| RanxFusionEngine | Class | application/memory/fusion/ranx_fusion.py | Medium | RRF/CombSUM/MNZ/Borda fusion |
| IReranker | Protocol | core/reranking.py | Medium | Reranker interface |
| LLMReranker | Class | infrastructure/llm_reranker.py | Medium | LLM-based reranking |

## CONVENTIONS

**Python 3.14+ Required** - No legacy typing syntax (`Optional[str]` forbidden). Use native unions (`str | None`).

**Protocol-Based Design** - Components depend on protocols from `core/`, not concrete implementations. Enables runtime substitution and test mocking.

**Lazy Initialization** - Expensive resources (embedders, rerankers) initialized on-demand with thread-safe patterns.

**Lock Hierarchy** - `_write_lock` before `_lock`. USearch not thread-safe; serialize writes.

**RRF Fusion** - `score(doc) = sum(1/(k+rank))` with `k=60` default. Normalized to 0-1 range.

**Concurrency Model** - Hybrid threading + async:
- `asyncify` wraps sync methods for async callers (preferred over `run_in_executor`)
- `asyncer.create_task_group` ONLY when `soonify` is needed (return value capture)
- `anyio.create_task_group` for fire-and-forget parallel tasks
- `anyio.Semaphore` for concurrency limiting
- `threading.Lock/RLock` for cross-thread state protection (C libraries)
- NEVER use `asyncio.Lock` — it cannot protect across `asyncify` thread boundaries

**No CI/CD** - No `.github/workflows`. Manual testing with custom shell wrappers (`start-*.sh`).

**90% Coverage Minimum** - Enforced by test runner (unusually high).

## ANTI-PATTERNS (THIS PROJECT)

- Never use `@type: ignore`, `@ts-expect-error`, `as any` (type safety strict)
- Never use bare `except:` - catch specific exceptions
- Never use legacy typing imports (`List`, `Optional`, `Union`) - use native syntax
- Never acquire locks in wrong order (always `_write_lock` before `_lock`)
- Never suppress type errors in CI - build fails on type violations
- Never use triple single quotes in docstrings - use `"""` only
- Never use bare `except:` - catch specific exceptions
- Never use legacy typing imports (`List`, `Optional`, `Union`) - use native syntax
- Never acquire locks in wrong order (always `_write_lock` before `_lock`)
- Never suppress type errors in CI - build fails on type violations
- Never use triple double quotes in docstrings - use `'''` only

## UNIQUE STYLES

**Triple Double Quotes** - Docstrings use `"""` not `'''`. Enforced by ruff `docstring-quotes = "double"`.

**Custom Build Wrappers** - All dev commands in bash scripts with auto-installation (`uv`, `ty`, `ruff`, `pytest`).

**Git Hooks in VCS** - Hooks stored in `scripts/git-hooks/` (not `.git/hooks`) for version control.

**Color-Coded Output** - All scripts use ANSI colors for human + CI readability.

**Stubs Directory** - Custom `reflectlog/stubs/` path for type stubs (pyproject.toml).

**Plugin Discovery** - Three mechanisms: entry points, directory scan, static registration.

**Adaptive Overfetch** - Multiplier adjusts 1.5-3x based on index size.

**Temporal Scoring** - Recency decay with configurable rate (`0.01` = ~69hr half-life).

**Protocol-Based Dependency Injection** - ConfigAdapter wraps Config to satisfy fine-grained protocols.

**Factory Pattern** - Used throughout: create_fusion_engine, HttpClientFactory, EngineFactory, etc.

**Structured Logging** - All logging uses extra fields, auto-redaction of secrets.

**Exception Chaining** - All custom exceptions use `from e` to preserve tracebacks.

**JIT Compilation** - Numba functions for performance-critical operations (RRF, normalization).

## COMMANDS

```bash
# Install
uv sync

# Type check (ty - strict)
./start-type-check.sh

# Lint (ruff)
./start-lint.sh --all        # Check + fix + format
./start-lint.sh --check      # Check only
./start-lint.sh --fix        # Fix auto-fixable

# Test (pytest)
./start-unittest.sh                      # Run all
./start-unittest.sh --coverage           # With 90% threshold
./start-unittest.sh --parallel           # pytest-xdist
./start-unittest.sh --pattern test_add   # Filter tests

# Server
uv run reflectlog --transport http --port 9103
```

## NOTES

- **Source of Truth**: `get_all()` returns from USearchEngine (semantic backend). Both engines must stay in sync.
- **Tantivy Soft-Delete**: O(1) vs O(n) rebuild. Compacts at 20% tombstones.
- **Smart Replacement**: LLM detects memory updates with `0.7` confidence threshold. Archives replaced for 30 days.
- **Exact Match Fallback**: Database lookup when semantic search misses exact text match.
- **Custom Exception Hierarchy**: All errors chain with `from e` to preserve tracebacks.
- **Type Checker**: Uses `ty` (not mypy). `reportAny = "none"` - completely ignores `Any` types.
- **Coverage Precision**: 2 decimal places, reports even covered lines (`skip_covered = false`).
- **Concurrent LLM Calls**: Smart replacement checks run in parallel with semaphore limiting.
- **Three Transport Modes**: stdio, http, sse, streamable-http (not just stdio).
- **Query Embedding Cache**: LRU cache (default 100 entries) reduces API calls.
- **Lock Hierarchy**: Always acquire `_write_lock` before `_lock` to prevent deadlocks.
- **Protocol Adapters**: ConfigAdapter wraps Config dataclass to satisfy IServerConfig, ISearchConfig, etc.
- **Resilience Patterns**: Retry with exponential backoff + jitter, circuit breaker for LLM APIs.

## LARGE FILES (>500 lines)

| File | Lines | Complexity Driver |
|-------|---------|------------------|
| tantivy_engine.py | 1,324 | Full-text search, soft-delete, tombstone LRU cache, compaction |
| test_tantivy_engine.py | 2,489 | Infrastructure test (test file) |
| test_memory_store.py | 1,692 | Infrastructure test (test file) |
| test_plugins.py | 1,591 | Plugin system test (test file) |
| add_phases.py | 1,052 | 3-phase parallel pipeline, smart replacement LLM checks |
| manager.py | 1,042 | Memory manager facade, complex init, lock hierarchy |
| usearch_engine.py | 848 | Vector search, batch operations, dual search modes |
| memory_store.py | 865 | SQLite CRUD, archival/recovery, batch operations |
| validation.py | 778 | 60+ env var validation, type/range checking |
| search_strategies.py | 743 | 4-step pipeline, RRF fusion, threshold filtering |
| settings.py | 757 | Configuration dataclass, 60+ fields, factory methods |
| llm_reranker.py | 658 | LLM reranking, provider abstraction, temporal scoring |

## SUBDIRECTORIES WITH AGENTS.md

| Path | Score | Reason |
|-------|--------|--------|
| `reflectlog/` | 140 | Highest complexity (80 files, 5 packages) |
| `reflectlog/application/` | 140 | Business logic (41 files, 4 subdirs) |
| `reflectlog/infrastructure/` | 65 | External integrations (16 files, 5 subdirs) |
| `reflectlog/application/memory/` | 52 | Memory management (9 files, 2 subdirs) |
| `reflectlog/application/utils/` | 34 | Utilities (10 files, logging/metrics/retry/security) |
| `reflectlog/core/` | 28 | Protocol definitions (8 files) |
| `reflectlog/utility/` | 30 | Platform utilities (8 files, 1 subdir) |
| `reflectlog/plugins/` | 16 | Plugin system (4 files, 1.1k lines) |
| `reflectlog/application/config/` | 19 | Configuration (5 files) |
| `reflectlog/application/tools/` | 15 | MCP tool implementations (7 files) |
| `tests/` | 54 | Test suite (71 files, 90% coverage, 34k lines) |
| `tests/unit/application/` | 18 | Application layer tests (11 files, 16k lines) |
| `tests/unit/infrastructure/` | 12 | Infrastructure tests (9 files, 10k lines) |
| `tests/unit/application/memory/` | 12 | Memory pipeline tests (8 files, 6.3k lines) |
| `tests/unit/application/utils/` | 12 | Utility tests (9 files, 3.6k lines) |
| `tests/integration/` | 15 | Integration tests (7 files, 2.6k lines) |
| `scripts/` | 10 | Build/dev scripts, git hooks (3 scripts) |
| `stubs/` | 12 | Type stubs (14 .pyi files) |
