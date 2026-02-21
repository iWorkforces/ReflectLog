'''Fusion strategies for hybrid search.'''

from typing import Any

from .base import FusionEngine
from .ranx_fusion import (
    DEFAULT_NORMALIZATIONS,
    SUPPORTED_METHODS,
    SUPPORTED_NORMALIZATIONS,
    RanxFusionEngine,
)


def create_fusion_engine(
    method: str = 'rrf',
    normalization: str | None = None,
    rrf_k: int = 60,
    logger: Any | None = None,
) -> FusionEngine:
    '''Factory function to create a fusion engine.

    Args:
        method: Fusion algorithm to use. One of: rrf, sum, mnz,
               max, bordafuse. Defaults to 'rrf'.
        normalization: Score normalization strategy. One of: min-max, max,
                      sum, zmuv, rank, borda. Defaults to auto-select
                      based on method.
        rrf_k: RRF k parameter (only used when method='rrf'). Lower values
              give more weight to top ranks. Defaults to 60.
        logger: Optional StructuredLogger instance for debug logging.

    Returns:
        FusionEngine instance configured with the specified parameters.

    Raises:
        ValueError: If method or normalization is not supported.
    '''
    return RanxFusionEngine(
        method=method,
        normalization=normalization,
        rrf_k=rrf_k,
        logger=logger,
    )


__all__ = [
    'DEFAULT_NORMALIZATIONS',
    'SUPPORTED_METHODS',
    'SUPPORTED_NORMALIZATIONS',
    'FusionEngine',
    'RanxFusionEngine',
    'create_fusion_engine',
]
