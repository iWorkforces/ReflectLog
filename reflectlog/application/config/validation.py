"""Configuration validation for ReflectLog Server.

This module provides comprehensive validation for configuration values,
ensuring that all settings are valid and consistent before the server starts.
"""

from dataclasses import dataclass
import re
from typing import Any, ClassVar, cast

from reflectlog.core.enums import (
    CrossEncoderDevice,
    FusionMethod,
    LlmProvider,
    RerankerEngine,
    TransportMode,
)


@dataclass
class ValidationError:
    """A single validation error."""

    field: str
    value: str | int | float | bool | None
    message: str

    def __str__(self) -> str:
        """String representation of the error."""
        return f"{self.field}: {self.message} (got: {self.value!r})"


class ConfigurationError(Exception):
    """Configuration validation error."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class ConfigurationValidator:
    """Validates configuration values.

    This class provides methods to validate various configuration settings,
    checking for type correctness, value ranges, and logical consistency.
    """

    # Regex patterns
    WORKSPACE_ID_PATTERN: ClassVar = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

    # Valid values for enums
    VALID_TRANSPORTS: ClassVar = frozenset(TransportMode)
    VALID_RERANKER_ENGINES: ClassVar = frozenset(RerankerEngine)
    VALID_LLM_PROVIDERS: ClassVar = frozenset(LlmProvider)
    VALID_CROSS_ENCODER_DEVICES: ClassVar = frozenset(CrossEncoderDevice)
    VALID_FUSION_METHODS: ClassVar = frozenset(FusionMethod)

    def __init__(self) -> None:
        """Initialize validator with an empty list of errors."""
        super().__init__()
        self.errors: list[ValidationError] = []

    def reset(self) -> None:
        """Clear all accumulated errors."""
        self.errors.clear()

    def add_error(self, field: str, value: Any, message: str) -> None:
        """Add a validation error.

        Args:
            field: The configuration field name
            value: The invalid value
            message: Description of why the value is invalid
        """
        self.errors.append(ValidationError(field=field, value=value, message=message))

    def validate_workspace_id(self, workspace_id: str) -> bool:
        """Validate WORKSPACE_ID format.

        Args:
            workspace_id: The workspace ID to validate

        Returns:
            True if valid, False otherwise
        """
        if not workspace_id:
            self.add_error("WORKSPACE_ID", workspace_id, "Cannot be empty")
            return False

        if not self.WORKSPACE_ID_PATTERN.match(workspace_id):
            self.add_error(
                "WORKSPACE_ID",
                workspace_id,
                "Must contain only A-Za-z0-9_.- and be 1-64 characters",
            )
            return False

        if ".." in workspace_id or workspace_id.startswith("/"):
            self.add_error(
                "WORKSPACE_ID",
                workspace_id,
                "Path traversal patterns not allowed",
            )
            return False

        return True

    def validate_transport(self, transport: str) -> bool:
        """Validate MCP_TRANSPORT value.

        Args:
            transport: The transport mode

        Returns:
            True if valid, False otherwise
        """
        if transport not in self.VALID_TRANSPORTS:
            self.add_error(
                "MCP_TRANSPORT",
                transport,
                f"Must be one of: {', '.join(sorted(self.VALID_TRANSPORTS))}",
            )
            return False
        return True

    def validate_port(self, port: int) -> bool:
        """Validate MCP_PORT value.

        Args:
            port: The port number

        Returns:
            True if valid, False otherwise
        """
        if not (1 <= port <= 65535):
            self.add_error("MCP_PORT", port, "Must be between 1 and 65535")
            return False
        return True

    def validate_percentage(
        self,
        field: str,
        value: float,
        min_value: float = 0.0,
        max_value: float = 1.0,
    ) -> bool:
        """Validate a percentage-based configuration value.

        Args:
            field: The configuration field name
            value: The value to validate
            min_value: Minimum allowed value (default: 0.0)
            max_value: Maximum allowed value (default: 1.0)

        Returns:
            True if valid, False otherwise
        """
        if not (min_value <= value <= max_value):
            self.add_error(
                field,
                value,
                f"Must be between {min_value} and {max_value}",
            )
            return False
        return True

    def validate_positive_int(
        self,
        field: str,
        value: int,
        min_value: int = 1,
    ) -> bool:
        """Validate a positive integer configuration value.

        Args:
            field: The configuration field name
            value: The value to validate
            min_value: Minimum allowed value (default: 1)

        Returns:
            True if valid, False otherwise
        """
        if value < min_value:
            self.add_error(
                field,
                value,
                f"Must be at least {min_value}",
            )
            return False
        return True

    def validate_positive_float(
        self,
        field: str,
        value: float,
        min_value: float = 0.0,
    ) -> bool:
        """Validate a positive float configuration value.

        Args:
            field: The configuration field name
            value: The value to validate
            min_value: Minimum allowed value (default: 0.0)

        Returns:
            True if valid, False otherwise
        """
        if value < min_value:
            self.add_error(
                field,
                value,
                f"Must be at least {min_value}",
            )
            return False
        return True

    def validate_reranker_engine(self, engine: str) -> bool:
        """Validate RERANKER_ENGINE value.

        Args:
            engine: The reranker engine type

        Returns:
            True if valid, False otherwise
        """
        if engine not in self.VALID_RERANKER_ENGINES:
            self.add_error(
                "RERANKER_ENGINE",
                engine,
                f"Must be one of: {', '.join(sorted(self.VALID_RERANKER_ENGINES))}",
            )
            return False
        return True

    def validate_llm_provider(self, provider: str) -> bool:
        """Validate LLM_PROVIDER value.

        Args:
            provider: The LLM provider

        Returns:
            True if valid, False otherwise
        """
        if provider not in self.VALID_LLM_PROVIDERS:
            self.add_error(
                "LLM_PROVIDER",
                provider,
                f"Must be one of: {', '.join(sorted(self.VALID_LLM_PROVIDERS))}",
            )
            return False
        return True

    def validate_cross_encoder_device(self, device: str) -> bool:
        """Validate CROSS_ENCODER_DEVICE value.

        Args:
            device: The cross-encoder device

        Returns:
            True if valid, False otherwise
        """
        if device not in self.VALID_CROSS_ENCODER_DEVICES:
            self.add_error(
                "CROSS_ENCODER_DEVICE",
                device,
                f"Must be one of: {', '.join(sorted(self.VALID_CROSS_ENCODER_DEVICES))}",
            )
            return False
        return True

    def validate_fusion_method(self, method: str) -> bool:
        """Validate FUSION_METHOD value.

        Args:
            method: The fusion method

        Returns:
            True if valid, False otherwise
        """
        if method not in self.VALID_FUSION_METHODS:
            self.add_error(
                "FUSION_METHOD",
                method,
                f"Must be one of: {', '.join(sorted(self.VALID_FUSION_METHODS))}",
            )
            return False
        return True

    def validate_dependencies(
        self,
        enable_hybrid_search: bool,
        enable_rrf_fusion: bool,
        reranker_engine: str,
    ) -> bool:
        """Validate logical dependencies between configuration options.

        Args:
            enable_hybrid_search: Whether hybrid search is enabled
            enable_rrf_fusion: Whether RRF fusion is enabled
            reranker_engine: The reranker engine type

        Returns:
            True if all dependencies are valid, False otherwise
        """
        valid = True

        # RRF fusion requires hybrid search (though we allow it for flexibility)
        # if enable_rrf_fusion and not enable_hybrid_search:
        #     self.add_error(
        #         "ENABLE_RRF_FUSION",
        #         enable_rrf_fusion,
        #         "RRF fusion requires hybrid search to be enabled",
        #     )
        #     valid = False

        # Reranker requires at least one search engine
        # (this is always true since we always have semantic search)

        return valid

    def validate_embedding_settings(
        self,
        embedder_provider: str,
        embedding_dims: int,
        qwen_embedding_dims: int,
    ) -> bool:
        """Validate embedding-related settings.

        Args:
            embedder_provider: The embedder provider
            embedding_dims: OpenAI embedding dimensions
            qwen_embedding_dims: Qwen embedding dimensions

        Returns:
            True if valid, False otherwise
        """
        valid = True

        # Validate dimensions are positive
        if not self.validate_positive_int("EMBEDDING_DIMS", embedding_dims):
            valid = False

        if not self.validate_positive_int("QWEN_EMBEDDING_DIMS", qwen_embedding_dims):
            valid = False

        return valid

    def validate_circuit_breaker_settings(
        self,
        enabled: bool,
        failure_threshold: int,
        timeout: float,
        success_threshold: int,
    ) -> bool:
        """Validate circuit breaker settings.

        Args:
            enabled: Whether circuit breaker is enabled
            failure_threshold: Number of failures before opening
            timeout: Seconds before attempting recovery
            success_threshold: Successes needed to close circuit

        Returns:
            True if valid, False otherwise
        """
        if not enabled:
            return True  # Skip validation if disabled

        valid = True

        if not self.validate_positive_int(
            "CIRCUIT_BREAKER_FAILURE_THRESHOLD",
            failure_threshold,
            min_value=1,
        ):
            valid = False

        if not self.validate_positive_float(
            "CIRCUIT_BREAKER_TIMEOUT",
            timeout,
            min_value=1.0,
        ):
            valid = False

        if not self.validate_positive_int(
            "CIRCUIT_BREAKER_SUCCESS_THRESHOLD",
            success_threshold,
            min_value=1,
        ):
            valid = False

        return valid

    def validate_memory_lengths(
        self,
        min_length: int,
        max_length: int,
    ) -> bool:
        """Validate memory length settings.

        Args:
            min_length: Minimum memory length
            max_length: Maximum memory length

        Returns:
            True if valid, False otherwise
        """
        valid = True

        if not self.validate_positive_int("MIN_MESSAGE_LENGTH", min_length):
            valid = False

        if not self.validate_positive_int("MAX_MESSAGE_LENGTH", max_length):
            valid = False

        if min_length >= max_length:
            self.add_error(
                "MIN_MESSAGE_LENGTH / MAX_MESSAGE_LENGTH",
                (min_length, max_length),
                "MIN_MESSAGE_LENGTH must be less than MAX_MESSAGE_LENGTH",
            )
            valid = False

        return valid

    def validate_query(
        self: ConfigurationValidator,
        query: str,
        max_length: int = 1000,
    ) -> bool:
        """Validate search query for security and content.

        Args:
            self: The validator instance
            query: The search query to validate
            max_length: Maximum allowed query length (default: 1000)

        Returns:
            True if valid, False otherwise
        """
        if not query:
            self.add_error("query", query, "Query cannot be empty")
            return False

        if len(query) > max_length:
            self.add_error(
                "query",
                query,
                f"Query length ({len(query)}) exceeds maximum ({max_length})",
            )
            return False

        return True

    def sanitize_query(
        self: ConfigurationValidator,
        query: str,
        max_length: int = 1000,
    ) -> str:
        """Sanitize search query to prevent injection attacks.

        Args:
            self: The validator instance
            query: The search query to sanitize
            max_length: Maximum allowed query length (default: 1000)

        Returns:
            Sanitized query string

        This function removes or escapes potentially dangerous characters:
        - SQL/NoSQL injection patterns
        - Command injection patterns
        - Excess whitespace
        """
        if not query:
            return ""

        # Limit length
        if len(query) > max_length:
            query = query[:max_length]

        # Remove null bytes and control characters
        sanitized = "".join(char for char in query if ord(char) >= 32)

        # Remove common SQL injection patterns
        injection_patterns = [
            r"';\s*--",
            r"'\s+or\s+",
            r"\s+and\s+",
            r";\s*drop\s+",
            r";\s*delete\s+",
            r";\s*insert\s+",
            r";\s*update\s+",
            r";\s*exec\s*",
            r"union\s+select",
            r"\|\s*select",
        ]

        for pattern in injection_patterns:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)

        # Remove multiple consecutive spaces
        sanitized = re.sub(r"\s{2,}", " ", sanitized)

        # Trim leading/trailing whitespace
        sanitized = sanitized.strip()

        return sanitized

    def validate_memory(
        self: ConfigurationValidator,
        content: str,
        min_length: int = 1,
        max_length: int = 30720,
    ) -> bool:
        """Validate memory content for security and content.

        Args:
            self: The validator instance
            content: The memory content to validate
            min_length: Minimum allowed length (default: 1)
            max_length: Maximum allowed length (default: 30720)

        Returns:
            True if valid, False otherwise
        """
        if not content:
            self.add_error("memory", content, "Memory cannot be empty")
            return False

        if len(content) < min_length:
            self.add_error(
                "memory",
                content,
                f"Memory length ({len(content)}) below minimum ({min_length})",
            )
            return False

        if len(content) > max_length:
            self.add_error(
                "memory",
                content,
                f"Memory length ({len(content)}) exceeds maximum ({max_length})",
            )
            return False

        # Check for null bytes and control characters (except newline, tab)
        for char in content:
            char_code = ord(char)
            if char_code < 9 or (char_code >= 11 and char_code <= 12):
                self.add_error(
                    "memory",
                    content,
                    "Memory contains invalid control characters",
                )
                return False

        return True

    def validate_openrouter_api_key_format(
        self: ConfigurationValidator,
        api_key: str,
    ) -> bool:
        """Validate OpenRouter API key format.

        Args:
            self: The validator instance
            api_key: The API key to validate

        Returns:
            True if valid format, False otherwise

        Expected format: sk-or-v1-XXXXXXXXXXXXXXXX
        """
        if not api_key:
            self.add_error("OPENROUTER_API_KEY", api_key, "API key cannot be empty")
            return False

        # OpenRouter keys start with 'sk-or-v1-'
        if not api_key.startswith("sk-or-v1-"):
            self.add_error(
                "OPENROUTER_API_KEY",
                api_key,
                "API key must start with 'sk-or-v1-'",
            )
            return False

        # Check length (typical OpenRouter keys are 51 characters: sk-or-v1- + 39 chars)
        if len(api_key) < 10 or len(api_key) > 100:
            self.add_error(
                "OPENROUTER_API_KEY",
                api_key,
                "API key length must be between 10 and 100 characters",
            )
            return False

        return True

    def validate_openrouter_api_key(
        self: ConfigurationValidator,
        api_key: str,
    ) -> bool:
        """Validate OpenRouter API key format.

        Args:
            self: The validator instance
            api_key: The API key to validate

        Returns:
            True if valid format, False otherwise

        Expected format: sk-or-v1-XXXXXXXXXXXXXXXX
        """
        if not api_key:
            self.add_error("OPENROUTER_API_KEY", api_key, "API key cannot be empty")
            return False

        # OpenRouter keys start with 'sk-or-v1-'
        if not api_key.startswith("sk-or-v1-"):
            self.add_error(
                "OPENROUTER_API_KEY",
                api_key,
                "API key must start with 'sk-or-v1-'",
            )
            return False

        # Check length (typical OpenRouter keys are 51 characters: sk-or-v1- + 39 chars)
        if len(api_key) < 10 or len(api_key) > 100:
            self.add_error(
                "OPENROUTER_API_KEY",
                api_key,
                "API key length must be between 10 and 100 characters",
            )
            return False

        return True

    def has_errors(self) -> bool:
        """Check if any validation errors have been recorded.

        Returns:
            True if there are errors, False otherwise
        """
        return len(self.errors) > 0

    def get_error_message(self) -> str:
        """Get a formatted error message with all validation errors.

        Returns:
            A string containing all validation errors, one per line
        """
        if not self.errors:
            return "No validation errors"

        lines = ["Configuration validation failed:"]
        for error in self.errors:
            lines.append(f"  - {error}")
        return "\n".join(lines)


def _get_attr(config: object, name: str) -> object | None:
    """Get an attribute from a config object, returning None if not found."""
    from reflectlog.core.access import optional_attr

    return optional_attr(config, name)


def _validate_server_config(
    validator: ConfigurationValidator,
    config: object,
) -> None:
    """Validate server transport, port, and workspace ID."""
    workspace_id = _get_attr(config, "workspace_id")
    if isinstance(workspace_id, str) and workspace_id:
        _ = validator.validate_workspace_id(workspace_id)

    transport = _get_attr(config, "transport")
    if isinstance(transport, str) and transport:
        _ = validator.validate_transport(transport)

    port = _get_attr(config, "port")
    if isinstance(port, int):
        _ = validator.validate_port(port)


def _validate_search_config(
    validator: ConfigurationValidator,
    config: object,
) -> None:
    """Validate search limit, threshold, and percentage fields."""
    percentage_fields = [
        ("search_score_threshold", "SEARCH_SCORE_THRESHOLD"),
        ("fusion_ranking_threshold", "FUSION_RANKING_THRESHOLD"),
        ("cross_encoder_score_threshold", "CROSS_ENCODER_SCORE_THRESHOLD"),
        ("smart_replace_threshold", "SMART_REPLACE_THRESHOLD"),
        ("smart_replace_min_similarity", "SMART_REPLACE_MIN_SIMILARITY"),
        (
            "tantivy_compaction_threshold_ratio",
            "TANTIVY_COMPACTION_THRESHOLD_RATIO",
        ),
        ("recency_decay_rate", "RECENCY_DECAY_RATE"),
    ]

    for attr_name, field_name in percentage_fields:
        value = _get_attr(config, attr_name)
        if isinstance(value, (int, float)):
            _ = validator.validate_percentage(field_name, float(value))

    positive_int_fields = [
        ("search_limit", "SEARCH_LIMIT", 1),
        ("remove_search_limit", "REMOVE_SEARCH_LIMIT", 1),
        ("fusion_rrf_k", "FUSION_RRF_K", 1),
        ("rerank_max_concurrency", "RERANK_MAX_CONCURRENCY", 1),
        ("cross_encoder_batch_size", "CROSS_ENCODER_BATCH_SIZE", 1),
        ("cross_encoder_max_length", "CROSS_ENCODER_MAX_LENGTH", 1),
        ("add_max_concurrency", "ADD_MAX_CONCURRENCY", 1),
        ("embedding_batch_size", "EMBEDDING_BATCH_SIZE", 1),
        ("embedding_cache_size", "EMBEDDING_CACHE_SIZE", 1),
    ]

    for attr_name, field_name, min_val in positive_int_fields:
        value = _get_attr(config, attr_name)
        if isinstance(value, int):
            _ = validator.validate_positive_int(field_name, value, min_val)


def _validate_storage_config(
    validator: ConfigurationValidator,
    config: object,
) -> None:
    """Validate memory lengths and logical dependencies."""
    min_length = _get_attr(config, "min_message_length")
    max_length = _get_attr(config, "max_message_length")
    if isinstance(min_length, int) and isinstance(max_length, int):
        _ = validator.validate_memory_lengths(min_length, max_length)

    enable_hybrid_search = _get_attr(config, "enable_hybrid_search")
    enable_rrf_fusion = _get_attr(config, "enable_rrf_fusion")
    reranker_engine = _get_attr(config, "reranker_engine")
    _ = validator.validate_dependencies(
        enable_hybrid_search if isinstance(enable_hybrid_search, bool) else True,
        enable_rrf_fusion if isinstance(enable_rrf_fusion, bool) else True,
        reranker_engine
        if isinstance(reranker_engine, str) and reranker_engine
        else RerankerEngine.CROSS_ENCODER,
    )


def _validate_embedder_config(
    validator: ConfigurationValidator,
    config: object,
) -> None:
    """Validate fusion method and fusion weights."""
    fusion_method = _get_attr(config, "fusion_method")
    if isinstance(fusion_method, str) and fusion_method:
        _ = validator.validate_fusion_method(fusion_method)

    fusion_weights = _get_attr(config, "fusion_weights")
    if fusion_weights is not None:
        if not isinstance(fusion_weights, list):
            raise ConfigurationError(
                f"fusion_weights must be a list, got {type(fusion_weights).__name__}"
            )
        typed_weights: list[object] = cast(list[object], fusion_weights)
        if len(typed_weights) < 2:
            raise ConfigurationError(
                "fusion_weights must have at least 2 elements for weighted RRF"
            )
        for i, w in enumerate(typed_weights):
            if not isinstance(w, (int, float)):
                raise ConfigurationError(
                    f"fusion_weights[{i}] must be a number, got {type(w).__name__}"
                )
            if w < 0:
                raise ConfigurationError(
                    f"fusion_weights[{i}] must be non-negative, got {w}"
                )


def _validate_reranker_config(
    validator: ConfigurationValidator,
    config: object,
) -> None:
    """Validate reranker engine, LLM provider, cross-encoder, and circuit breaker."""
    reranker_engine = _get_attr(config, "reranker_engine")
    if isinstance(reranker_engine, str) and reranker_engine:
        _ = validator.validate_reranker_engine(reranker_engine)

    llm_provider = _get_attr(config, "llm_provider")
    if isinstance(llm_provider, str) and llm_provider:
        _ = validator.validate_llm_provider(llm_provider)

    cross_encoder_device = _get_attr(config, "cross_encoder_device")
    if isinstance(cross_encoder_device, str) and cross_encoder_device:
        _ = validator.validate_cross_encoder_device(cross_encoder_device)

    circuit_breaker_enabled = _get_attr(config, "circuit_breaker_enabled")
    if isinstance(circuit_breaker_enabled, bool) and circuit_breaker_enabled:
        failure_threshold = _get_attr(config, "circuit_breaker_failure_threshold")
        timeout = _get_attr(config, "circuit_breaker_timeout")
        success_threshold = _get_attr(config, "circuit_breaker_success_threshold")

        _ = validator.validate_circuit_breaker_settings(
            circuit_breaker_enabled,
            failure_threshold if isinstance(failure_threshold, int) else 5,
            timeout if isinstance(timeout, (int, float)) else 60.0,
            success_threshold if isinstance(success_threshold, int) else 2,
        )


def _validate_security_fields(
    validator: ConfigurationValidator,
    config: object,
) -> None:
    """Validate API key formats and security-sensitive fields."""
    openrouter_api_key = _get_attr(config, "openrouter_api_key")
    if isinstance(openrouter_api_key, str) and openrouter_api_key:
        _ = validator.validate_openrouter_api_key_format(openrouter_api_key)


def validate_config(config: object) -> list[ValidationError]:
    """Validate a configuration object.

    This is a convenience function that creates a validator and runs
    all validation checks on the given configuration object.

    Args:
        config: A Config object (or object with similar attributes)

    Returns:
        List of validation errors (empty if valid)
    """
    validator = ConfigurationValidator()

    _validate_server_config(validator, config)
    _validate_search_config(validator, config)
    _validate_storage_config(validator, config)
    _validate_embedder_config(validator, config)
    _validate_reranker_config(validator, config)
    _validate_security_fields(validator, config)

    return validator.errors


__all__ = [
    "ConfigurationValidator",
    "ValidationError",
    "validate_config",
]
