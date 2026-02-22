# Config Unit Tests

**Generated:** 2026-02-22
**Commit:** 6c2d6fa
**Branch:** develop

## OVERVIEW

Unit tests for configuration presets and validation logic. Mocked environment, no real config loading.

## STRUCTURE

```
tests/unit/application/config/
├── test_presets.py         # Preset profiles (simple/balanced/performance/quality)
└── test_validation.py      # 60+ env var validation, SQL injection, path traversal
```

## WHERE TO LOOK

| Test | Purpose |
|------|---------|
| test_presets.py | Config preset switching via `REFLECTLOG_PROFILE` |
| test_validation.py | `ConfigurationValidator` with edge cases |

## KEY PATTERNS

### Preset Override
```python
def test_preset_override(monkeypatch):
    monkeypatch.setenv("REFLECTLOG_PROFILE", "performance")
    config = Config.from_environment()
    assert config.search_limit == 10  # performance preset
```

### SQL Injection Tests
```python
injection_patterns = [
    "'; DROP TABLE memories; --",
    "' OR '1' = '1",
    "1; DELETE FROM messages",
]
```

## ANTI-PATTERNS

- Never test with real environment variables
- Never skip validation edge cases

## NOTES

- **NUMBA disabled**: Required for coverage compatibility
- **Config reset**: autouse fixture clears singleton between tests
