"""Configuration validation for ReflectLogMCP Server.

This module provides comprehensive validation for configuration values,
ensuring that all settings are valid and consistent before the server starts.
"""

from dataclasses import dataclass
import re
from typing import Any, ClassVar


@dataclass
class ValidationError:
    """A single validation error."""

    field: str
    value: Any
    message: str

    def __str__(self) -> str:
        """String representation of the error."""
        return f"{self.field}: {self.message} (got: {self.value!r})"


class ConfigurationValidator:
    """Validates configuration values.

    This class provides methods to validate various configuration settings,
    checking for type correctness, value ranges, and logical consistency.
    """

    # Regex patterns
    PROJECT_ID_PATTERN: ClassVar = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

    # Valid values for enums
    VALID_TRANSPORTS: ClassVar = {"stdio", "http", "sse", "streamable-http"}
    VALID_RERANKER_ENGINES: ClassVar = {"llm", "cross_encoder", "none"}
    VALID_LLM_PROVIDERS: ClassVar = {"openai", "anthropic"}
    VALID_CROSS_ENCODER_DEVICES: ClassVar = {"cpu", "cuda", "mps"}
    VALID_FUSION_METHODS: ClassVar = {"rrf", "sum", "mnz", "max", "bordafuse"}

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

    def validate_project_id(self, project_id: str) -> bool:
        """Validate PROJECT_ID format.

        Args:
            project_id: The project ID to validate

        Returns:
            True if valid, False otherwise
        """
        if not project_id:
            self.add_error("PROJECT_ID", project_id, "Cannot be empty")
            return False

        if not self.PROJECT_ID_PATTERN.match(project_id):
            self.add_error(
                "PROJECT_ID",
                project_id,
                "Must contain only A-Za-z0-9_.- and be 1-64 characters",
            )
            return False

        if ".." in project_id or project_id.startswith("/"):
            self.add_error(
                "PROJECT_ID",
                project_id,
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

    def validate_message_lengths(
        self,
        min_length: int,
        max_length: int,
    ) -> bool:
        """Validate message length settings.

        Args:
            min_length: Minimum message length
            max_length: Maximum message length

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

    def validate_message(
        self: ConfigurationValidator,
        message: str,
        min_length: int = 1,
        max_length: int = 30720,
    ) -> bool:
        """Validate message content for security and content.

        Args:
            self: The validator instance
            message: The message to validate
            min_length: Minimum allowed length (default: 1)
            max_length: Maximum allowed length (default: 30720)

        Returns:
            True if valid, False otherwise
        """
        if not message:
            self.add_error("message", message, "Message cannot be empty")
            return False

        if len(message) < min_length:
            self.add_error(
                "message",
                message,
                f"Message length ({len(message)}) below minimum ({min_length})",
            )
            return False

        if len(message) > max_length:
            self.add_error(
                "message",
                message,
                f"Message length ({len(message)}) exceeds maximum ({max_length})",
            )
            return False

        # Check for null bytes and control characters (except newline, tab)
        for char in message:
            char_code = ord(char)
            if char_code < 9 or (char_code >= 11 and char_code <= 12):
                self.add_error(
                    "message",
                    message,
                    "Message contains invalid control characters",
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

    def get_attr(name: str) -> Any:
        return getattr(config, name, None)

    # Validate project_id
    project_id = get_attr("project_id")
    if project_id:
        _ = validator.validate_project_id(project_id)

    # Validate transport
    transport = get_attr("transport")
    if transport:
        _ = validator.validate_transport(transport)

    # Validate port
    port = get_attr("port")
    if port is not None:
        _ = validator.validate_port(port)

    # Validate percentage values
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
        value = get_attr(attr_name)
        if value is not None:
            _ = validator.validate_percentage(field_name, value)

    # Validate positive integers
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
        value = get_attr(attr_name)
        if value is not None:
            _ = validator.validate_positive_int(field_name, value, min_val)

    # Validate reranker engine
    reranker_engine = get_attr("reranker_engine")
    if reranker_engine:
        _ = validator.validate_reranker_engine(reranker_engine)

    # Validate LLM provider
    llm_provider = get_attr("llm_provider")
    if llm_provider:
        _ = validator.validate_llm_provider(llm_provider)

    # Validate cross-encoder device
    cross_encoder_device = get_attr("cross_encoder_device")
    if cross_encoder_device:
        _ = validator.validate_cross_encoder_device(cross_encoder_device)

    # Validate fusion method
    fusion_method = get_attr("fusion_method")
    if fusion_method:
        _ = validator.validate_fusion_method(fusion_method)

    # Validate message lengths
    min_length = get_attr("min_message_length")
    max_length = get_attr("max_message_length")
    if min_length is not None and max_length is not None:
        _ = validator.validate_message_lengths(min_length, max_length)

    # Validate dependencies
    enable_hybrid_search = get_attr("enable_hybrid_search")
    enable_rrf_fusion = get_attr("enable_rrf_fusion")
    reranker_engine = get_attr("reranker_engine")
    _ = validator.validate_dependencies(
        enable_hybrid_search if enable_hybrid_search is not None else True,
        enable_rrf_fusion if enable_rrf_fusion is not None else True,
        reranker_engine if reranker_engine else "llm",
    )

    # Validate circuit breaker settings
    circuit_breaker_enabled = get_attr("circuit_breaker_enabled")
    if circuit_breaker_enabled:
        _ = validator.validate_circuit_breaker_settings(
            circuit_breaker_enabled,
            get_attr("circuit_breaker_failure_threshold") or 5,
            get_attr("circuit_breaker_timeout") or 60.0,
            get_attr("circuit_breaker_success_threshold") or 2,
        )

    # Validate OpenRouter API key format
    openrouter_api_key = get_attr("openrouter_api_key")
    if openrouter_api_key:
        _ = validator.validate_openrouter_api_key_format(openrouter_api_key)

    return validator.errors


__all__ = [
    "ConfigurationValidator",
    "ValidationError",
    "validate_config",
]
