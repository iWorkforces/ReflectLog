# Config Unit Tests

**Generated:** 2026-09-03
**Commit:** e401dbc
**Branch:** develop

## OVERVIEW
Unit tests for presets, frozen `Config`, and `ConfigurationValidator`. Env is monkeypatched.

## STRUCTURE
```
tests/unit/application/config/
├── test_presets.py       # simple / balanced / performance / quality
├── test_settings.py      # Config.from_environment, get_config, SecretString
└── test_validation.py    # env bounds, SQL / path / workspace_id
```

## WHERE TO LOOK
| Test | Purpose |
|------|---------|
| `test_presets.py` | `REFLECTLOG_PROFILE` + `apply_preset_to_env` |
| `test_settings.py` | Frozen dataclass, lazy singleton, `_parse_optional_bool` |
| `test_validation.py` | `ConfigurationValidator` / `validate_config` |

## CONVENTIONS
- Monkeypatch env; reset Config singleton between tests.
- Secrets stay in `SecretString`; never assert raw tokens in logs.
- Native 3.14 unions. Double quotes.

## ANTI-PATTERNS
- Never leak real environment into tests.
- Never skip validation edge cases (SQL, `../`, overlong workspace_id).
- Ban `getattr`, `optional_attr()`, and `type(obj).__dict__`.
- No `type: ignore`.

## NOTES
NUMBA may be disabled in sibling utils tests for coverage; config tests should not depend on JIT. Local: env-isolated frozen Config; no process singleton.
