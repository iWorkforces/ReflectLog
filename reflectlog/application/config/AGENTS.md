# Agent Guidelines for reflectlog/application/config/

**Generated:** 2026-08-26  **Commit:** 95567fa  **Branch:** develop

## OVERVIEW

Centralized configuration management with environment variable loading and comprehensive validation. LLM prompt templates live in `core/prompts.py`.

## STRUCTURE

```
config/
├── settings.py        # Config dataclass (60+ env vars, frozen)
├── validation.py      # ConfigurationValidator class
├── presets.py         # Config presets (simple/balanced/performance/quality)
└── __init__.py        # Public API exports
```

## WHERE TO LOOK

| Task | File | Notes |
|-------|------|--------|
| Config dataclass | settings.py | Frozen dataclass, environment variable loading |
| Validation logic | validation.py | Type, range, SQL injection prevention |
| Prompt templates | `../../core/prompts.py` | Smart replacement detection |
| Presets | presets.py | Pre-configured profiles for different use cases |
| Factory methods | settings.py | create_*_config() for engine configs |

## CONVENTIONS

**Frozen Config** - `@dataclass(frozen=True)` prevents runtime modifications to Config.

**Secret Wrapping** - API keys wrapped in `SecretString` (use `.get_secret_value()`).

**Jailbreak Protection** - Use `Template.safe_substitute()` and escape braces `{` `}` in user input.

**Validation in __post_init__** - Raise `ConfigurationError` for invalid values immediately.

**Regex Patterns** - `WORKSPACE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")` prevents path traversal.

**Preset Override** - Individual env vars override preset values via `REFLECTLOG_PROFILE`.

**Factory Pattern** - Static methods create specialized configs from base Config.

## ANTI-PATTERNS

- Never modify Config at runtime - it's frozen for safety
- Never use `.format()` on prompts with user input - use `Template.safe_substitute()`
- Never log or expose API keys - SecretString prevents accidental leaks
- Never skip validation - always validate in `__post_init__` or via ConfigurationValidator
- Never allow path traversal - validate WORKSPACE_ID against regex
