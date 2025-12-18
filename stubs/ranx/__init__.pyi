"""Type stubs for ranx library."""

from typing import Any, Dict, List, Optional

class Run:
    """A ranked run containing query-document scores."""

    run: Dict[str, Dict[str, float]]
    name: str

    def __init__(
        self,
        run_dict: Dict[str, Dict[str, float]],
        name: Optional[str] = None,
    ) -> None: ...

def fuse(
    runs: List[Run],
    norm: Optional[str] = None,
    method: str = "rrf",
    params: Optional[Dict[str, Any]] = None,
) -> Run:
    """Fuse multiple runs into a single combined run.

    Args:
        runs: List of Run objects to fuse.
        norm: Normalization strategy (e.g., 'min-max', 'rank').
        method: Fusion method (e.g., 'rrf', 'combsum').
        params: Method-specific parameters (e.g., {'k': 60} for RRF).

    Returns:
        Combined Run object with fused scores.
    """
    ...
