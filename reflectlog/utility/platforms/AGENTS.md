# Agent Guidelines for reflectlog/utility/platforms/

**Generated:** 2026-08-29  **Commit:** 7df1375  **Branch:** develop

## OVERVIEW
Platform abstraction for secure credential retrieval from OS-specific stores.

## STRUCTURE
```
platforms/
├── __init__.py      # Factory function (get_platform_retriever)
├── base.py          # CredentialRetriever ABC with parse_credential()
├── darwin.py        # macOS Keychain via security CLI
├── linux.py         # Linux secret-tool + env fallback
└── windows.py       # Windows Credential Manager
```

## WHERE TO LOOK
| Task | Location | Notes |
|-------|----------|-------|
| Platform factory | platforms/__init__.py | get_platform_retriever() |
| Credential parsing | platforms/base.py | 3-format fallback chain |

## CONVENTIONS
- Factory returns platform subclass or `None` on unknown OS. All `get_credential()` return None on errors.
- Darwin/Linux subprocess timeout 10s; **Windows 30s**. Lazy import in factory.
- Linux reads config files first (inline parse, skips `parse_credential()`), then `secret-tool`.

## ANTI-PATTERNS
- Never raise or log credential values
- Never call `get_credential()` without a subprocess timeout
