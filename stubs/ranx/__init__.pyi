"""Type stubs for ranx library."""

from typing import Any

class Run:
    """A ranked run containing query-document scores."""

    run: dict[str, dict[str, float]]
    name: str

    def __init__(
        self,
        run_dict: dict[str, dict[str, float]],
        name: str | None = None,
    ) -> None: ...

def fuse(
    runs: list[Run],
    norm: str | None = None,
    method: str = "rrf",
    params: dict[str, Any] | None = None,
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
