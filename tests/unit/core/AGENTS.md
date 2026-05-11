# Core Unit Tests

**Generated:** 2026-04-11  **Commit:** 6f2b0f8  **Branch:** develop

## OVERVIEW

Unit tests for core protocol adapters. Verifies Config-to-protocol mapping correctness.

## STRUCTURE

```
tests/unit/core/
└── test_config_adapters.py   # ConfigAdapter protocol compliance
```

## WHERE TO LOOK

| Test | Purpose |
|------|---------|
| test_config_adapters.py | ConfigAdapter wraps Config to satisfy IServerConfig, ISearchConfig, etc. |

## ANTI-PATTERNS

- Never test with real Config singleton
- Never skip protocol compliance checks

## NOTES

- **Protocol verification**: Ensures ConfigAdapter satisfies all fine-grained protocols
- **Large test file**: Covers 10+ protocol interfaces
