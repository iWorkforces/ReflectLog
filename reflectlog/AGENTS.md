# reflectlog Package

**Generated:** 2026-09-03
**Commit:** e401dbc
**Branch:** develop

## OVERVIEW
Installed hatch package. CLI is `server.py:main`. Plugins present; not imported at startup.

## STRUCTURE

```
reflectlog/
├── server.py            # CLI, Numba warmup, SIGINT/SIGTERM persist
├── version.py           # CLI version; may diverge from pyproject
├── application/         # MCP, Config, MemoryManager, tools
│   └── utils/           # logging, SecretString, validation, unused SIGHUP
├── core/                # Protocols + StrEnums + adapters + leases
├── infrastructure/      # Engines FLAT; embeddings/ is the only full child
│   ├── embeddings/      # Qwen client + LRU cache
│   └── storage_coordinator.py  # Portalocker + generation sidecar
├── plugins/             # discovery/registry/loading; unwired
└── utility/             # HttpClientFactory, Numba scoring, retry, OS credentials
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Startup | `server.py` | Warmup; no plugin load; no `setup_config_reload` |
| MCP tools | `application/mcp_server.py` | `AVAILABLE_TOOL_CLASSES`; injects `MemoryManager` |
| Engines | `application/memory/manager.py` | Inlines `from_config()`; `EngineFactory` unused at runtime |
| Workspace lease | `infrastructure/storage_coordinator.py` | Lives here, not in `core/` |
| Lease protocol | `core/storage_coordination.py` | `IStorageCoordinator`, `LeaseMode` |
| Embeddings | `infrastructure/embeddings/` | `LangchainQwenEmbeddings`, `CachedEmbeddings` |
| HTTP | `utility/http.py` | `HttpClientFactory`; no leftover `http_client.py` |

## CONVENTIONS

- Two utility layers: `utility/` = HTTP/scoring/retry/credentials; `application/utils/` = logging/`SecretString`/validation.
- Factories: `from_config()` (not `from_app_config()`).
- Lock order: Portalocker lease → `_write_lock` → `_lock`. `threading.RLock`.
- `access.py` deleted. No `getattr` / `optional_attr` / `invoke_if_callable` / `type(obj).__dict__`.

## ANTI-PATTERNS

- Do not treat `plugins/` as live at startup.
- Do not construct engines via `EngineFactory` in production paths.
- Do not put embedding modules back on `infrastructure/` root.
- Do not move `storage_coordinator.py` out of `infrastructure/`.
- Do not resurrect `http_client.py`, `metrics.py`, or `circuit_breaker.py` as APIs.
