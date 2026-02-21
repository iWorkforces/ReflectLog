'''Utilities for ReflectLogMCP Server.'''

from collections.abc import Callable
from typing import Any, cast

from . import circuit_breaker as _circuit_breaker
from . import metrics as _metrics
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitState,
)
from .config_reload import ConfigReloadManager, setup_signal_handler
from .logging import (
    StructuredLogger,
    create_logger,
    format_fusion_score_status,
)
from .metrics import MetricsRegistry, MetricValue
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
from .validation import truncate_memory, validate_memories

_CIRCUIT_BREAKER_DECORATOR_NAME = 'circuit_breaker_decorator'
_TIMED_NAME = 'timed'

circuit_breaker_decorator = cast(
    Callable[..., Any],
    getattr(_circuit_breaker, _CIRCUIT_BREAKER_DECORATOR_NAME),
)
timed = cast(Callable[..., Any], getattr(_metrics, _TIMED_NAME))

__all__ = [
    # CamelCase
    'CircuitBreaker',
    'CircuitBreakerConfig',
    'CircuitBreakerOpenError',
    'CircuitState',
    'ConfigReloadManager',
    'MetricValue',
    'MetricsRegistry',
    'SecretString',
    'StructuredLogger',
    # snake_case
    'circuit_breaker_decorator',
    'compute_rrf_scores_batch',
    'create_logger',
    'distance_to_similarity_cosine',
    'filter_scores_by_threshold',
    'format_fusion_score_status',
    'normalize_scores_minmax',
    'redact_dict_secrets',
    'sanitize_for_logging',
    'setup_signal_handler',
    'timed',
    'truncate_memory',
    'validate_memories',
    'warmup_numba_functions',
]
