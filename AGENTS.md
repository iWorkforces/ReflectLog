# ReflectLog Knowledge Base

**Generated:** 2026-09-03
**Commit:** e401dbc
**Branch:** develop

## OVERVIEW

Python 3.14 MCP memory server. USearch + SQLite identity + Tantivy FTS + raw RRF, optional local FlagReranker. LLM is for smart replacement only.

## STRUCTURE

```
./
├── reflectlog/          # Flat hatch package (not src/)
│   ├── server.py        # Installed CLI: reflectlog.server:main
│   ├── application/     # Config, MCP, add/search pipelines, tools
│   ├── core/            # Protocols, StrEnums, adapters, leases
│   ├── infrastructure/  # Engines FLAT; embeddings/ is the only full child
│   ├── plugins/         # Present; not wired at startup
│   └── utility/         # HTTP pool, Numba scoring, OS credentials
├── tests/               # unit/ mirrors package; integration uses real engines
├── stubs/               # ty extra-paths + pyright stubPath
├── scripts/             # Copied git hooks + platform gates + ad-hoc
└── start-*.sh           # Local lint / typecheck / pytest wrappers
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| CLI + signals | `reflectlog/server.py` | Warmup; persist on SIGINT/SIGTERM/SIGBREAK |
| MCP + auth | `application/mcp_server.py` | Tools via `MemoryManager` only |
| Memory facade | `application/memory/manager.py` | Locks, journal, public API |
| Add / search | `application/memory/` | 3-phase add; 4-step hybrid search |
| Journal replay | `application/memory/replacement_recovery.py` | Restart converge |
| Workspace lease | `infrastructure/storage_coordinator.py` | Portalocker + generation sidecar |
| Config | `application/config/settings.py` | Frozen env `Config` |
| Protocols | `core/` | `ISemanticSearchEngine`, `IStorageCoordinator` |
| Backends | `infrastructure/` | USearch, Tantivy, SQLite, CE |
| Embeddings | `infrastructure/embeddings/` | Qwen client + LRU cache |
| HTTP | `utility/http.py` | `HttpClientFactory` |
| Score math | `utility/scoring.py` | Numba RRF / min-max / recency |

## CODE MAP

| Symbol | Type | Location | Refs | Role |
|--------|------|----------|------|------|
| `main` | Function | `server.py` | CLI | Installed entry |
| `FastMCPServer` | Class | `application/mcp_server.py` | High | Tool registry + transports |
| `MemoryManager` | Class | `application/memory/manager.py` | ~139 | Add/search/delete facade |
| `Config` | Dataclass | `application/config/settings.py` | High | Frozen env settings |
| `ConfigAdapter` | Class | `core/config_adapters.py` | High | `Config` → protocols |
| `USearchEngine` | Class | `infrastructure/usearch_engine.py` | High | HNSW + SQLite SoT |
| `TantivyEngine` | Class | `infrastructure/tantivy_engine.py` | High | FTS + tombstones |
| `MemoryStore` | Class | `infrastructure/memory_store.py` | High | Identity + journal |
| `PortalockerStorageCoordinator` | Class | `storage_coordinator.py` | High | Exclusive/shared lease |
| `RanxFusionEngine` | Class | `memory/fusion/ranx_fusion.py` | Med | RRF Numba; other ranx |
| `CrossEncoderReranker` | Class | `cross_encoder_reranker.py` | High | Search Step 4 |
| `delete_memories` | Method | `manager.py` | High | Returns deleted `list[str]` |

## CONVENTIONS

- 3.14+ native unions; Ruff `UP` + `ANN`/`TC`/`PYI`; double quotes including docstrings.
- Protocols at layer boundaries; factories are `from_config()`.
- Lock order: Portalocker lease → `_write_lock` → `_lock`. `threading.RLock`, not `asyncio.Lock`.
- Embed add/batch **outside** exclusive. Search: no manager SHARED across embed/CE.
- `raise ... from e`. Never log memory text or secrets.
- Dual typecheck: `ty` + pyright. Both must pass. No `type: ignore`.

## ANTI-PATTERNS (THIS PROJECT)

- No `getattr`, `optional_attr()`, `invoke_if_callable()`, or `type(obj).__dict__`.
- No MagicMock auto-attrs as APIs; put the method on a protocol.
- Tools never touch engines; go through `MemoryManager`.
- No compact-on-delete. No embed-batch pad with `[]`. No single-list fusion min-max.
- No recency before CE normalize/threshold. Skip CE if ≤1 hit (pipeline owns skip).
- No HNSW load when SQLite is missing/empty/unreadable.
- No first-create `Index.save(live)`. First durable write is temp+replace on commit.

## UNIQUE STYLES

- Add: dedup/replace → embed outside lease → persist NEW then OLD.
- Search: parallel backends → raw RRF (threshold 0.0) → CE if >1 hit.
- Identity: unique `(workspace_id, content)` in SQLite. Tantivy is not exact match.
- `get_all()` / `count()` SoT is USearch/SQLite.
- Journal kinds `add|delete|replace`; later-write-wins.
- Publish generation after engines converge, then leftover ADD complete, then replace complete.
- Coverage fail-under 90%. Pytest warnings are errors.

## COMMANDS

```bash
uv sync
./start-type-check.sh
./start-lint.sh --check
./start-unittest.sh --coverage
uv run reflectlog --transport http --port 9103
```

## NOTES

- Installed script: `reflectlog = reflectlog.server:main`. `mcp_server.main` is a thinner second entry.
- `ty` + pyright include `tests`; tests execution env relaxes mock noise.
- Default pytest paths omit `tests/load/` and root `tests/test_*.py`.
- Hooks: edit `scripts/git-hooks/`, install via `scripts/setup-git-hooks.sh`. Pre-push = typecheck + lint `--all` (writes). No pytest in hooks.
- Focused CI: `.github/workflows/platform-storage.yml` (storage/USearch/Tantivy gates). Not full lint/type/coverage.
- `pyproject.toml` version and `reflectlog/version.py` can diverge; CLI prints the latter.

## GUIDANCE HIERARCHY

Children: `reflectlog/{application,core,infrastructure,plugins,utility}` and `tests/` mirrors.
Deep: `memory/fusion`, `memory/reranking` (pointer), `utility/platforms`, `infrastructure/embeddings`, `infrastructure/search` (marker).
No guides on empty `infrastructure/{llm,memory,reranking}`.
