# ReflectLog Knowledge Base

**Generated:** 2026-08-29
**Commit:** 7df1375
**Branch:** develop

## OVERVIEW

Python 3.14 MCP memory server. USearch semantic + Tantivy FTS + RRF fusion, optional local cross-encoder rerank (`RERANKER_ENGINE=cross_encoder|none`), LLM only for smart replacement.

## STRUCTURE

```
./
├── reflectlog/                     # Runtime package
│   ├── application/                # Pipelines, configuration, MCP tools
│   ├── core/                       # Protocols, domain types, adapters
│   ├── infrastructure/             # Engines and external integrations
│   ├── plugins/                    # Discovery, registry, lifecycle
│   └── utility/                    # Scoring and platform credentials
├── tests/                          # Unit, integration, load, security suites
├── stubs/                          # Third-party type stubs
├── scripts/                        # Hook setup and developer utilities
├── start-*.sh                      # Validation wrappers
└── pyproject.toml                  # Package and tool configuration
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| CLI lifecycle | `reflectlog/server.py` | Argument parsing, warmup, signals, server startup |
| MCP orchestration | `reflectlog/application/mcp_server.py` | Tool selection, registration, transports |
| Memory facade | `reflectlog/application/memory/manager.py` | Engine construction, locking, public memory API |
| Add/search pipelines | `reflectlog/application/memory/` | Three-phase add; staged hybrid search |
| Environment config | `reflectlog/application/config/settings.py` | Frozen `Config`, parsing, presets |
| Contracts and types | `reflectlog/core/` | Protocol-first dependency boundaries |
| Storage/search backends | `reflectlog/infrastructure/` | USearch, Tantivy, SQLite, cross-encoder |
| Pooled HTTP | `reflectlog/utility/http.py` | Production `HttpClientFactory`; app `utils/http_client.py` is leftover |
| Score math | `reflectlog/utility/scoring.py` | Numba-compiled fusion, normalization, filtering |
| Tests | `tests/` | Unit mirrors package; integration uses real engines |

## CODE MAP

| Symbol | Type | Location | Refs | Role |
|--------|------|----------|------|------|
| `main` | Function | `reflectlog/server.py` | CLI | Configures and runs the MCP server |
| `FastMCPServer` | Class | `application/mcp_server.py` | High | Builds and exposes selected tools |
| `MemoryManager` | Class | `application/memory/manager.py` | High | Facade for add, search, get, and delete |
| `Config` | Dataclass | `application/config/settings.py` | High | Environment-derived application settings |
| `ConfigAdapter` | Class | `core/config_adapters.py` | High | Converts `Config` into fine-grained protocols |
| `USearchEngine` | Class | `infrastructure/usearch_engine.py` | High | Semantic index and SQLite-backed records |
| `TantivyEngine` | Class | `infrastructure/tantivy_engine.py` | High | Full-text index, tombstones, compaction |
| `RanxFusionEngine` | Class | `application/memory/fusion/ranx_fusion.py` | Medium | RRF and alternative fusion algorithms |
| `RerankerPostProcessor` | Class | `infrastructure/reranker_post_processor.py` | Medium | CE post-process + recency |
| `CrossEncoderReranker` | Class | `infrastructure/cross_encoder_reranker.py` | High | Local FlagReranker; search Step 4 |
| `delete_memories` | Method | `application/memory/manager.py` | High | Returns `list[str]` of deleted contents |

## CONVENTIONS

- Python 3.14+: native unions only; do not import legacy `typing` collection aliases.
- Use protocols from `reflectlog/core/` at layer boundaries. Runtime implementations expose `from_config()` factories.
- `Config` is immutable and environment-derived; use adapters rather than coupling application code to it.
- Docstrings and strings use double quotes; Ruff enforces `docstring-quotes = "double"`.
- Sync C-library operations cross `asyncify` thread boundaries. Use `threading.Lock`/`RLock`, not `asyncio.Lock`.
- Preserve lock order: acquire `_write_lock` before `_lock`.
- Wrap external failures in project exceptions and preserve causes with `raise ... from e`.
- Use structured logging with redacted `extra` fields; never log memory content or credentials casually.

## ANTI-PATTERNS (THIS PROJECT)

- Do not suppress type errors with `type: ignore`, `@ts-expect-error`, or `as any`.
- Do not use bare `except:` or empty exception handlers.
- Do not access search engines directly from MCP tools; route operations through the memory pipelines/manager.
- Do not treat USearch writes as thread-safe.
- Do not normalize reranker scores one item at a time or apply recency decay before normalization.
- Do not compact Tantivy on the delete path; `compact()` is maintenance only.
- Do not pad short embed batches with `[]`; fail closed.
- Do not min-max a single fusion list (drops near-ties at threshold 0.8).
- Do not treat MagicMock auto-attrs as batch APIs; production uses `type(obj).__dict__.get(...)`.
- Do not expose secrets, tokens, or API keys in logs, exceptions, or tests.

## UNIQUE STYLES

- Add flow: parallel dedup/replace, then embed outside `_write_lock`, then sequential persist.
- Search flow: parallel backends, fusion, threshold filter, optional CE rerank (skip if ≤1 hit).
- Tantivy delete is tombstone+commit only; compact when ratio/count thresholds fire.
- The source of truth for `get_all()` is the USearch semantic backend; maintain backend consistency.
- Plugin support covers entry points, directory scanning, and static registration.
- Test configuration treats warnings as errors; coverage is reported by the wrapper but has no enforced fail-under gate.

## COMMANDS

```bash
uv sync
./start-type-check.sh
./start-lint.sh --check
./start-lint.sh --all
./start-unittest.sh
./start-unittest.sh --coverage
./start-unittest.sh --parallel
uv run reflectlog --transport http --port 9103
```

## NOTES

- `project.scripts.reflectlog` targets `reflectlog.server:main`.
- `ty` checks both `reflectlog` and `tests`, with `stubs/` on its extra path.
- `pytest` is async-auto; shared fixtures are in `tests/conftest.py`.
- Memory and application-utility test fixtures disable Numba JIT for coverage; integration tests use real engines.
- Hooks are versioned in `scripts/git-hooks/`; edit there, then install with `scripts/setup-git-hooks.sh`.

## GUIDANCE HIERARCHY

Child guides: `reflectlog/{application,core,infrastructure,plugins,utility}` and test mirrors. Deepest: `memory/fusion`, `memory/reranking` (pointer), `utility/platforms`, `infrastructure/search` (marker). No guides on empty `infrastructure/{embeddings,llm,memory,reranking}` markers.
