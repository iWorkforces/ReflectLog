# Core Unit Tests

**Generated:** 2026-09-03
**Commit:** e401dbc
**Branch:** develop

## OVERVIEW
Unit tests for `ConfigAdapter` protocol mapping and replacement prompt interpolation.

## STRUCTURE
```
tests/unit/core/
├── test_config_adapters.py   # ConfigAdapter + fine-grained adapters
└── test_prompts.py           # format_replacement_detection_prompt
```

## WHERE TO LOOK
| Test | Purpose |
|------|---------|
| `test_config_adapters.py` | `Config` → `ISearchConfig` / `IRerankerConfig` / etc. `RerankerEngine` is `cross_encoder` or `none` |
| `test_prompts.py` | Live `$old_memory` / `$new_memory`; single-brace JSON; no `{{` doubling |

## CONVENTIONS
- Build `Config` in-test; do not use the process singleton.
- Protocol checks are explicit (`isinstance` / adapter factories), not reflection.
- Native 3.14 unions. Double-quoted strings.

## ANTI-PATTERNS
- Never test against the real Config singleton.
- Never skip protocol compliance.
- Ban `getattr`, `optional_attr()`, and `type(obj).__dict__`.
- No `type: ignore`. No MagicMock auto-attrs as APIs.

## NOTES
`create_*_config_adapter` factories live in `reflectlog/core/config_adapters.py`. `_coerce_reranker_engine` is covered here. Local: protocol adapters + prompt interpolation only.
