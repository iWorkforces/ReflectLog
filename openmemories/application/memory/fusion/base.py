"""Base protocol for fusion engines."""

from typing import List, Optional, Protocol, Tuple


class FusionEngine(Protocol):
    """Protocol for fusion engines.

    All fusion engines must implement this interface to be used
    interchangeably with the MemoryManager.
    """

    @property
    def method(self) -> str:
        """Return the fusion method name (e.g., 'rrf', 'combsum')."""
        ...

    @property
    def normalization(self) -> Optional[str]:
        """Return the normalization strategy (e.g., 'min-max', 'rank')."""
        ...

    def fuse(
        self,
        *result_sets: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """Fuse multiple ranked lists into a single ranking.

        Args:
            *result_sets: Variable number of (message, score) tuple lists.
                         Each list should be sorted by score descending.

        Returns:
            List of (message, fused_score) tuples sorted by score descending.
        """
        ...
