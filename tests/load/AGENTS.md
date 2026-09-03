# Load Tests

**Generated:** 2026-09-03
**Commit:** e401dbc
**Branch:** develop

## OVERVIEW

Locust scenarios against a running MCP server. **Locust is not a project dependency.** `tests/load/` is **not** pytest.

## STRUCTURE

```
tests/load/
└── locustfile.py    # ReflectLogUser — add / search / get_all / health
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Add | `ReflectLogUser.add_memory` → `POST /mcp/add` |
| Search | `ReflectLogUser.search_memory` → `POST /mcp/search` |
| Get all | `ReflectLogUser.get_all_memories` → `GET /mcp/get_all` |
| Health | `ReflectLogUser.health_check` → `GET /mcp/health_check` |

## CONVENTIONS

Locust imported only when not type-checking. Install locust yourself:

```bash
# not `uv run` — locust is not in the lockfile
locust -f tests/load/locustfile.py --host http://localhost:9103
locust -f tests/load/locustfile.py --headless -u 100 -r 10 -t 60s
```

## ANTI-PATTERNS

- Never add locust to `pyproject.toml` just to run this file.
- Never put `tests/load/` on pytest paths.
- Never run against production.
- Never drop `wait_time`.

## NOTES

Start the server separately (`uv run reflectlog --transport http --port 9103`). Focused CI does not run Locust. Pre-push does not run load tests.
