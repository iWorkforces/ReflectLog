"""Fusion strategies for hybrid search."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reflectlog.core.logging import IStructuredLogger

from reflectlog.core.enums import FusionMethod, FusionNormalization

from .base import FusionEngine
from .ranx_fusion import RanxFusionEngine


def create_fusion_engine(
    method: FusionMethod | str = FusionMethod.RRF,
    normalization: FusionNormalization | str | None = None,
    rrf_k: int = 60,
    weights: list[float] | None = None,
    logger: IStructuredLogger | None = None,
) -> FusionEngine:
    """Factory function to create a fusion engine.

    Args:
        method: Fusion algorithm to use. One of: rrf, sum, mnz,
               max, bordafuse. Defaults to 'rrf'.
        normalization: Score normalization strategy. One of: min-max, max,
                      sum, zmuv, rank, borda. Defaults to auto-select
                      based on method.
        rrf_k: RRF k parameter (only used when method='rrf'). Lower values
              give more weight to top ranks. Defaults to 60.
        weights: Optional list of weights for weighted RRF fusion. Must have
                at least 2 elements if provided. Defaults to None (equal weights).
        logger: Optional StructuredLogger instance for debug logging.

    Returns:
        FusionEngine instance configured with the specified parameters.

    Raises:
        ValueError: If method or normalization is not supported.
    """
    return RanxFusionEngine(
        method=method,
        normalization=normalization,
        rrf_k=rrf_k,
        weights=weights,
        logger=logger,
    )


__all__ = [
    "create_fusion_engine",
]
