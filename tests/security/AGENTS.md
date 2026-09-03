# Security Tests

**Generated:** 2026-09-03
**Commit:** e401dbc
**Branch:** develop

## OVERVIEW

Injection / traversal / XSS. In default pytest `testpaths`.

## STRUCTURE

```
tests/security/
└── test_security.py
```

## WHERE TO LOOK

| Test | Purpose |
|------|---------|
| `test_sql_injection_basic` | Search payload must not execute SQL |
| `test_command_injection` | Shell metacharacters |
| `test_xss_attempt` | Stored content not treated as code |
| `test_path_traversal_workspace_id` | `WORKSPACE_ID` charset |
| `test_auth_bypass_attempt` | No token/header bypass |
| `test_rate_limit_respected` / `test_concurrent_burst` | Load-adjacent safety |

## CONVENTIONS

- Manager is a partial `MemoryManager.__new__` with mocked engines — not a live store.
- `WORKSPACE_ID` pattern: `^[A-Za-z0-9_.-]{1,64}$`.
- Never log tokens or memory text.

## ANTI-PATTERNS

- Never skip this file before release.
- Never test only sanitized inputs.

## NOTES

Focused CI (`platform-storage.yml`) does **not** run this path. Local: `./start-unittest.sh`. Hooks do **not** run pytest.
