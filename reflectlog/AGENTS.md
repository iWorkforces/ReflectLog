# reflectlog Package

**Generated:** 2026-08-30  **Commit:** 062b44f  **Branch:** develop

## OVERVIEW
Installed hatch package. CLI is `server.py:main`. Children own pipelines, protocols, engines.

## STRUCTURE

```
reflectlog/
├── server.py            # CLI, Numba warmup, SIGINT/SIGTERM persist
├── version.py           # CLI version; may diverge from pyproject
├── application/         # MCP, Config, MemoryManager, tools
│   └── utils/           # logging, SecretString, validation, unused SIGHUP
├── core/                # Protocols + StrEnums + adapters
├── infrastructure/      # Engines at root; embeddings/ is the only full child
│   └── embeddings/      # Qwen client + LRU cache (not infrastructure root)
├── plugins/             # discovery/registry/loading; not imported at startup
└── utility/             # HttpClientFactory, Numba scoring, retry, OS credentials
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Startup | `server.py` | Warmup; no plugin load; no `setup_config_reload` |
| MCP tools | `application/mcp_server.py` | `AVAILABLE_TOOL_CLASSES`; injects `MemoryManager` |
| Engines | `application/memory/manager.py` | Inlines `from_config()`; `EngineFactory` unused at runtime |
| Embeddings | `infrastructure/embeddings/` | `LangchainQwenEmbeddings`, `CachedEmbeddings` |
| HTTP | `utility/http.py` | `HttpClientFactory`; no leftover `http_client.py` |
| Enums | `core/enums.py` | `RerankerEngine` StrEnum, not Literal |
| Types | `core/types.py` | `IStoredMemory`; `ISemanticSearchEngine.embedder` |

## CONVENTIONS

- Factories: `from_config()` (not `from_app_config()`).
- Two utility layers: `utility/` = HTTP/scoring/retry/credentials; `application/utils/` = logging/`SecretString`/validation.
- `access.py` deleted. No `getattr` / `optional_attr` / `invoke_if_callable` / `type(obj).__dict__`.

## ANTI-PATTERNS

- Do not treat `plugins/` as live at startup.
- Do not construct engines via `EngineFactory` in production paths.
- Do not put embedding modules back on `infrastructure/` root.
- Do not resurrect `http_client.py`, `metrics.py`, or `circuit_breaker.py` as APIs.
