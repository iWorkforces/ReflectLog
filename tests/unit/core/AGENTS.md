# Core Unit Tests

**Generated:** 2026-08-29  **Commit:** 7df1375  **Branch:** develop

## OVERVIEW

Unit tests for core protocol adapters. Verifies Config-to-protocol mapping correctness.

## STRUCTURE

```
tests/unit/core/
├── test_config_adapters.py   # ConfigAdapter protocol compliance
└── test_prompts.py           # Replacement Template interpolation
```

## WHERE TO LOOK

| Test | Purpose |
|------|---------|
| test_config_adapters.py | ConfigAdapter wraps Config; `RerankerEngine` is `cross_encoder`\|`none` |
| test_prompts.py | Live `$old_memory`/`$new_memory`; no `{{` doubling |

## ANTI-PATTERNS

- Never test with real Config singleton
- Never skip protocol compliance checks

## NOTES

- **Protocol verification**: Ensures ConfigAdapter satisfies all fine-grained protocols
- **Large test file**: Covers 10+ protocol interfaces
