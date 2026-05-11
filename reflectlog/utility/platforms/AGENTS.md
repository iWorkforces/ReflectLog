# Agent Guidelines for reflectlog/utility/platforms/

**Generated:** 2026-04-11  **Commit:** 6f2b0f8  **Branch:** develop

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
- Factory returns platform subclass, all get_credential() return None on errors
- 10s subprocess timeout, lazy import in factory (no module-level platform imports)

## ANTI-PATTERNS
- Never raise exceptions on credential retrieval - return None
- Never log credential values or partial tokens
- Never bypass parse_credential() validation
- Never call get_credential() without subprocess timeout
