"""Utilities for ReflectLogMCP Server."""

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitState,
    circuit_breaker_decorator,
)
from .logging import (
    StructuredLogger,
    create_logger,
    format_fusion_score_status,
)
from .metrics import MetricsRegistry, MetricValue, timed
from .numba_utils import (
    compute_rrf_scores_batch,
    distance_to_similarity_cosine,
    filter_scores_by_threshold,
    normalize_scores_minmax,
    warmup_numba_functions,
)
from .security import (
    SecretString,
    redact_dict_secrets,
    sanitize_for_logging,
)
from .validation import truncate_message, validate_messages

__all__ = [
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpenError",
    "CircuitState",
    "circuit_breaker_decorator",
    # Metrics
    "MetricValue",
    "MetricsRegistry",
    "timed",
    # Numba-accelerated utilities
    "compute_rrf_scores_batch",
    "distance_to_similarity_cosine",
    "filter_scores_by_threshold",
    "normalize_scores_minmax",
    "warmup_numba_functions",
    # Security
    "SecretString",
    "redact_dict_secrets",
    "sanitize_for_logging",
    # Validation and logging
    "create_logger",
    "format_fusion_score_status",
    "StructuredLogger",
    "truncate_message",
    "validate_messages",
]
