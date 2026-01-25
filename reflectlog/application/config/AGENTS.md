# Agent Guidelines for reflectlog/application/config/

This directory contains configuration management and system prompts for ReflectLogMCP. It centralizes environment variable handling, configuration validation, and LLM prompt templates.

## Directory Structure

```
config/
├── __init__.py          # Package exports and public API
├── settings.py          # Config dataclass from environment variables
├── prompts.py           # MCP instructions and LLM prompt templates
└── validation.py        # Configuration validation utilities
```

## Core Responsibilities

### Configuration Management

The `settings.py` module provides the `Config` dataclass that loads all settings from environment variables:

```python
@dataclass
class Config:
    '''Centralized configuration from environment variables.'''

    # Required configuration
    project_id: str

    # Optional configuration with sensible defaults
    transport: str = "stdio"
    port: int = 9103
    host: str = "127.0.0.1"

    # LLM configuration
    llm_model: str = "x-ai/grok-4.1-fast"
    embedding_model: str = "openai/text-embedding-3-large"
```

### Prompt Templates

The `prompts.py` module contains all LLM prompt templates:

- `MCP_INSTRUCTIONS`: System instructions for the MCP server
- `SCORING_PROMPT`: Prompt for relevance scoring without recency
- `SCORING_PROMPT_WITH_AGE`: Prompt for recency-aware scoring
- `REPLACEMENT_DETECTION_PROMPT`: Prompt for smart replacement detection

### Validation

The `validation.py` module provides utilities for validating configuration:

- Type checking for configuration values
- Range validation for numeric settings
- Dependency validation (e.g., API keys required for certain features)

## Key Patterns

### Environment Variable Loading

Use Pydantic-like patterns for environment variable loading:

```python
import os
from dataclasses import field

def get_env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes")

def get_env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))
```

### Configuration Factory

Provide factory methods for creating configurations:

```python
@classmethod
def from_app_config(cls, app_config: Config) -> "USearchConfig":
    '''Create USearchConfig from application Config.'''
    return cls(
        index_path=app_config.usearch_index_path_template.format(
            project_id=app_config.project_id
        ),
        metric="cosine",
        dimensions=app_config.embedding_dims,
    )
```

### Prompt Template Management

Store prompts as module-level constants with clear documentation:

```python
SCORING_PROMPT = '''You are a relevance scoring assistant.
Given a query and a memory, score how relevant the memory is to the query.

Query: {query}
Memory: {memory}

Score the relevance from 0.0 to 1.0, where 1.0 means highly relevant.
'''
```

## Configuration Categories

### Server Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PROJECT_ID` | str | required | Unique project identifier |
| `MCP_TRANSPORT` | str | "stdio" | Transport mode |
| `MCP_PORT` | int | 9103 | HTTP port |
| `MCP_HOST` | str | "127.0.0.1" | HTTP host |
| `LOG_LEVEL` | str | "INFO" | Logging level |

### LLM Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LLM_MODEL` | str | "x-ai/grok-4.1-fast" | LLM model for reranking |
| `LLM_PROVIDER` | str | "anthropic" | LLM provider (openai, anthropic) |
| `OPENROUTER_API_KEY` | str | None | OpenRouter API key |
| `EMBEDDING_MODEL` | str | "openai/text-embedding-3-large" | Embedding model |

### Search Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_HYBRID_SEARCH` | bool | true | Enable full-text search |
| `ENABLE_RRF_FUSION` | bool | true | Enable RRF fusion |
| `RERANKER_ENGINE` | str | "llm" | Reranking engine |
| `SEARCH_SCORE_THRESHOLD` | float | 0.5 | Minimum relevance score |
| `SEARCH_LIMIT` | int | 5 | Maximum results |

### Memory Management Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_SMART_REPLACE` | bool | true | Enable smart replacement |
| `SMART_REPLACE_THRESHOLD` | float | 0.7 | Replacement confidence |
| `DEDUPLICATE_MESSAGES` | bool | true | Skip exact duplicates |

### Performance Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `EAGER_INITIALIZATION` | bool | true | Pre-warm engines |
| `EMBEDDING_CACHE_ENABLED` | bool | true | Cache query embeddings |
| `EMBEDDING_CACHE_SIZE` | int | 100 | Cache size |
| `ADD_MAX_CONCURRENCY` | int | 4 | Parallel add limit |

## Validation Patterns

### Type Validation

Validate configuration types at initialization:

```python
def __post_init__(self) -> None:
    if self.port < 1 or self.port > 65535:
        raise ConfigurationError(f"Invalid port: {self.port}")
    if self.search_limit < 1:
        raise ConfigurationError("Search limit must be positive")
```

### Dependency Validation

Validate required dependencies:

```python
def __post_init__(self) -> None:
    if self.reranker_engine == "llm" and not self.openrouter_api_key:
        raise ConfigurationError(
            "OPENROUTER_API_KEY required for LLM reranking"
        )
```

### Range Validation

Validate numeric ranges:

```python
def __post_init__(self) -> None:
    if not 0.0 <= self.smart_replace_threshold <= 1.0:
        raise ConfigurationError(
            "SMART_REPLACE_THRESHOLD must be between 0.0 and 1.0"
        )
```

## Prompt Engineering

### Prompt Structure

Follow consistent structure for LLM prompts:

```python
PROMPT_TEMPLATE = '''## Role
You are a {role}.

## Task
{description}

## Context
{context}

## Input
Query: {query}
Memory: {memory}

## Output Format
Respond with a JSON object containing:
- "score": relevance score (0.0 to 1.0)
- "reasoning": brief explanation
'''
```

### Prompt Variables

Use clear variable placeholders:

```python
SCORING_PROMPT = '''Query: {query}
Memory: {memory}

Score relevance (0.0 to 1.0):'''
```

### Prompt Testing

Test prompts with various inputs:

```python
def test_scoring_prompt():
    prompt = SCORING_PROMPT.format(
        query="Python web development",
        memory="I prefer Python for web APIs"
    )
    assert "Python web development" in prompt
    assert "Python for web APIs" in prompt
```

## Common Operations

### Loading Configuration

```python
config = Config(
    project_id=os.getenv("PROJECT_ID"),
    openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
    transport=os.getenv("MCP_TRANSPORT", "stdio"),
    port=int(os.getenv("MCP_PORT", "9103")),
)
```

### Creating Engine Configurations

```python
usearch_config = USearchConfig.from_app_config(config)
tantivy_config = TantivyConfig.from_app_config(config)
```

### Validating Configuration

```python
validator = ConfigurationValidator(config)
if not validator.is_valid():
    errors = validator.get_errors()
    raise ConfigurationError(f"Invalid config: {errors}")
```

## Error Handling

### Configuration Errors

Use custom exception types:

```python
from reflectlog.application.exceptions import ConfigurationError

raise ConfigurationError(f"Missing required config: {key}")
```

### Validation Errors

Provide detailed validation messages:

```python
@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str]
    warnings: list[str]
```

## Dependencies

### Internal Dependencies

- `application/utils/`: Logging and validation utilities
- `application/exceptions.py`: Exception classes

### External Dependencies

- `pydantic`: Configuration validation (if used)
- `python-dotenv`: Environment variable loading (if used)

## Important Notes

### Sensitive Data

Never log or expose sensitive configuration:

- API keys should not appear in logs
- Use secret redaction utilities
- Mask configuration in error messages

### Defaults

Provide sensible defaults for all optional configuration:

- Avoid requiring explicit configuration for common use cases
- Document all defaults clearly
- Test with default configuration

### Backward Compatibility

Maintain backward compatibility when adding configuration:

- Never remove required configuration without deprecation period
- Support old configuration names via aliases
- Provide migration paths for configuration changes
