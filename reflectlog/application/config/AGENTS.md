# Application Config

**Generated:** 2026-08-30
**Commit:** 062b44f
**Branch:** develop

## OVERVIEW

Frozen env-backed `Config`. Enum fields are `StrEnum` from `core.enums` (parsed via `parse_str_enum`). LLM prompt templates are in `core/prompts.py`, not here.

## STRUCTURE

```
config/
├── settings.py      # @dataclass(frozen=True) Config; get_config() singleton
├── validation.py    # ConfigurationValidator + WORKSPACE_ID_PATTERN
├── presets.py       # REFLECTLOG_PROFILE = simple|balanced|performance|quality
└── __init__.py      # Re-exports
```

Adapters (`Config` → protocol configs) live in `core/config_adapters.py` (`from_config()`).

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Env load | `settings.py` `Config.from_environment` | Fail-fast `ConfigurationError` |
| Singleton | `get_config()` / `_config_lock` | Tests reset via `reset_env_after_test` |
| Reranker | `RERANKER_ENGINE` | `cross_encoder` (default) or `none`. `llm` is invalid |
| Presets | `presets.py` | Env vars already set win over profile |
| Workspace | regex `^[A-Za-z0-9_.-]{1,64}$` | Path `..` / leading `/` rejected |

## CONVENTIONS

- `Config` is frozen. Mutate env + clear `_config`, do not assign fields.
- Typed enum fields: `TransportMode`, `RerankerEngine`, `FusionMethod`, `LlmProvider`, `EmbedderProvider`, `CrossEncoderDevice`.
- API keys are `SecretString`; use `.get_secret_value()`. Never log them.
- `setup_config_reload()` exists but is not registered at process startup.

## ANTI-PATTERNS

- Never treat `RERANKER_ENGINE=llm` as valid.
- Never use `str.format` on prompts with user text (`core/prompts.py` uses `string.Template`).
- Never log raw `OPENROUTER_API_KEY`.
- Never skip `__post_init__` / `ConfigurationValidator` when adding fields.
- Never allow path-like `WORKSPACE_ID`.
