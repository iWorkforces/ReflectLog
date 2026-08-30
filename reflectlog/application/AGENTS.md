# ReflectLog Application Layer

**Generated:** 2026-08-30  **Commit:** 062b44f  **Branch:** develop

## OVERVIEW
Orchestration: MCP + `MemoryManager`. Tools never touch engines. Pipeline math lives in `memory/` and `utility/scoring.py`.

## STRUCTURE

```
application/
├── mcp_server.py        # FastMCPServer, AVAILABLE_TOOL_CLASSES, HTTP bearer
├── constants.py         # MIN_OVERFETCH_LIMIT, log truncate
├── config/              # Frozen Config; settings.setup_config_reload unused
├── memory/
│   ├── manager.py       # Public API; inlines USearch/Tantivy/fusion
│   ├── engine_factory.py # EngineFactory exists; unused at runtime
│   ├── add_phases.py    # AddPipeline + DuplicateDetection/SmartReplacement/Storage
│   ├── search_strategies.py # SearchPipeline / SearchContext / SearchResult
│   └── replacement_recovery.py
├── tools/               # BaseTool subclasses
└── utils/               # logging, security.SecretString, validation
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Tool wiring | `mcp_server.py` | Injects `MemoryManager`; `ALLOWED_TOOLS` filter |
| Facade | `memory/manager.py` | `_write_lock` then `_lock`; lazy CE/replacer |
| Add | `memory/add_phases.py` | Phase classes above; embed outside write lock |
| Search | `memory/search_strategies.py` | 4-step; skip CE if ≤1 hit |
| Config reload | `config/settings.py` `setup_config_reload` | SIGHUP helper; not called at startup |
| App utils | `utils/` | Not HTTP/scoring — those are `utility/` |

## CONVENTIONS

- Engines: manager calls `USearchConfig.from_config(ConfigAdapter(config))` (CE/replacer same). Do not route startup through `EngineFactory`.
- Tools: `AddTool` / `SearchTool` / `GetAllTool` / `RemoveTool` / `HealthCheckTool` only.
- `application/utils/` has no `http_client.py` / `metrics.py` / `retry.py` / `circuit_breaker.py`.
- `access.py` deleted. No `getattr` / `optional_attr` / `invoke_if_callable` / `type(obj).__dict__`.

## ANTI-PATTERNS

- Tools must not import engines or fusion.
- Do not call `setup_config_reload()` from `server.py` / `FastMCPServer`.
- Do not treat leftover `utils/*.pyc` (`http_client`, `metrics`, `circuit_breaker`) as APIs.
