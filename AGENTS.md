# ReflectLogMCP Knowledge Base

**Generated:** 2026-01-27
**Commit:** 94d13da
**Branch:** develop

## OVERVIEW

MCP server providing persistent, project-based semantic memory storage for AI agents. Combines USearch vector search with Tantivy full-text search using RRF fusion, with optional LLM/cross-encoder reranking and smart memory replacement.

## STRUCTURE

```
./
├── reflectlog/              # Main package (77 .py files)
│   ├── core/              # Protocol definitions (8 files)
│   ├── application/         # Business logic (41 files)
│   ├── infrastructure/      # External integrations (16 files)
│   ├── plugins/           # Plugin system (4 files)
│   └── utility/           # Platform utilities (8 files)
├── tests/                # Unit + integration tests (49 files)
├── stubs/               # Type stubs for third-party libs
├── indexes/              # Persistent index data
├── scripts/              # Build/CI scripts (3 scripts)
└── *.sh                  # Custom wrapper scripts (3 scripts)
```

## WHERE TO LOOK

| Task | Location | Notes |
|-------|-----------|--------|
| Memory operations | `reflectlog/application/memory/manager.py` | 3-phase add, 4-step search pipeline |
| Search configuration | `reflectlog/application/config/settings.py` | 60+ env vars |
| Protocol interfaces | `reflectlog/core/` | All abstractions defined here |
| Infrastructure wrappers | `reflectlog/infrastructure/` | USearch, Tantivy, LLM providers |
| Build commands | `start-type-check.sh`, `start-lint.sh`, `start-unittest.sh` | Custom wrappers, no CI/CD |

## CODE MAP

| Symbol | Type | Location | Refs | Role |
|---------|------|----------|--------|------|
| MemoryManager | Class | application/memory/manager.py | High | Facade for all memory operations |
| Config | Dataclass | application/config/settings.py | High | Centralized configuration |
| ISearchBackend | Protocol | core/search.py | Medium | Abstract search engine interface |
| USearchEngine | Class | infrastructure/search/usearch_engine.py | Medium | Semantic vector search backend |
| TantivyEngine | Class | infrastructure/search/tantivy_engine.py | Medium | Full-text search backend |

## CONVENTIONS

**Python 3.14+ Required** - No legacy typing syntax (`Optional[str]` forbidden). Use native unions (`str | None`).

**Protocol-Based Design** - Components depend on protocols from `core/`, not concrete implementations. Enables runtime substitution and test mocking.

**Lazy Initialization** - Expensive resources (embedders, rerankers) initialized on-demand with thread-safe patterns.

**Lock Hierarchy** - `_write_lock` before `_lock`. USearch not thread-safe; serialize writes.

**RRF Fusion** - `score(doc) = sum(1/(k+rank))` with `k=60` default. Normalized to 0-1 range.

**No CI/CD** - No `.github/workflows`. Manual testing with custom shell wrappers (`start-*.sh`).

**90% Coverage Minimum** - Enforced by test runner (unusually high).

## ANTI-PATTERNS (THIS PROJECT)

- Never use `@type: ignore`, `@ts-expect-error`, `as any` (type safety strict)
- Never use bare `except:` - catch specific exceptions
- Never use legacy typing imports (`List`, `Optional`, `Union`) - use native syntax
- Never acquire locks in wrong order (always `_write_lock` before `_lock`)
- Never suppress type errors in CI - build fails on type violations

## UNIQUE STYLES

**Triple Single Quotes** - Docstrings use `'''not """`. Enforced by lint script.

**Custom Build Wrappers** - All dev commands in bash scripts with auto-installation (`uv`, `ty`, `ruff`, `pytest`).

**Git Hooks in VCS** - Hooks stored in `scripts/git-hooks/` (not `.git/hooks`) for version control.

**Color-Coded Output** - All scripts use ANSI colors for human + CI readability.

**Stubs Directory** - Custom `reflectlog/stubs/` path for type stubs (pyproject.toml).

**Plugin Discovery** - Three mechanisms: entry points, directory scan, static registration.

**Adaptive Overfetch** - Multiplier adjusts 1.5-3x based on index size.

**Temporal Scoring** - Recency decay with configurable rate (`0.01` = ~69hr half-life).

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

## SUBDIRECTORIES WITH AGENTS.md

| Path | Score | Reason |
|-------|--------|--------|
| `reflectlog/` | 30 | High complexity (77 files, 5 packages) |
| `reflectlog/core/` | 14 | Protocol definitions (8 files) |
| `reflectlog/application/` | 16 | Business logic (41 files) |
| `reflectlog/infrastructure/` | 14 | External integrations (16 files) |
| `reflectlog/application/memory/` | 12 | Memory management (11 files) |
| `tests/` | 10 | Test suite (49 files) |
| `reflectlog/application/utils/` | 8 | Utilities (8 files) |
