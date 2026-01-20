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
    "validate_messages",
    "truncate_message",
    "StructuredLogger",
    "create_logger",
    "format_fusion_score_status",
    "SecretString",
    "sanitize_for_logging",
    "redact_dict_secrets",
    # Numba-accelerated utilities
    "normalize_scores_minmax",
    "filter_scores_by_threshold",
    "compute_rrf_scores_batch",
    "distance_to_similarity_cosine",
    "warmup_numba_functions",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpenError",
    "CircuitState",
    "circuit_breaker_decorator",
    # Metrics
    "MetricsRegistry",
    "MetricValue",
    "timed",
]
