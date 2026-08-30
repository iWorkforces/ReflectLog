# Agent Guidelines for reflectlog/utility/platforms/

**Generated:** 2026-08-30  **Commit:** 062b44f  **Branch:** develop

## OVERVIEW
OS credential retrievers for Claude Code tokens. Factory returns a platform subclass or `None`. Failures return `None`; never raise or log tokens.

## STRUCTURE
```
platforms/
├── __init__.py      # get_platform_retriever() — lazy import by platform.system()
├── base.py          # CredentialRetriever ABC + parse_credential()
├── darwin.py        # macOS Keychain via `security` (timeout 10s)
├── linux.py         # Config files first (inline parse), then secret-tool (10s)
└── windows.py       # Credential Manager via PowerShell (timeout 30s)
```

## WHERE TO LOOK
| Task | Location | Notes |
|-------|----------|-------|
| Factory | `__init__.py` | Darwin / Windows / Linux; unknown OS → `None` |
| Parse chain | `base.py` | OAuth JSON → legacy `apiKey` → raw `sk-ant-` token |
| Linux files | `linux.py` `_check_config_files` | Inline JSON/raw parse; **skips** `parse_credential()` |
| Linux keyring | `linux.py` `_get_from_secret_tool` | Uses `parse_credential()` |
| Windows | `windows.py` | `timeout=30` (not 10s) |

## CONVENTIONS
- `get_credential()` always returns `str | None`. Catch `SubprocessError` / `OSError`.
- Darwin/Linux subprocess timeout **10s**; Windows **30s**.
- Linux config paths: `~/.claude/credentials`, `~/.config/claude-code/credentials`, `$XDG_CONFIG_HOME/claude-code/credentials`.
- Prefix check uses `TOKEN_PREFIX` / `SERVICE_NAME` from `utility/types.py`.
- Lazy import inside the factory `match` — no platform modules at package import.

## ANTI-PATTERNS
- Never raise or log tokens, partial tokens, or raw store output.
- Never treat Windows timeout as 10s.
- Never route Linux config-file reads through `parse_credential()`.
- Never call `get_credential()` without a subprocess timeout.
- Ban `getattr`, `optional_attr()`, and `type(obj).__dict__`.

## NOTES
Parent `utility/` owns `get_anthropic_api_key()` / `init_credentials()`. This package is OS I/O only.
