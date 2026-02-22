"""Type stubs for ranx library."""

from typing import NotRequired, TypedDict

class FuseParams(TypedDict, total=False):
    """Parameters supported by fusion algorithms."""

    k: NotRequired[int | float]
    phi: NotRequired[int | float]
    alpha: NotRequired[float]
    beta: NotRequired[float]
    gamma: NotRequired[float]
    weights: NotRequired[dict[str, float]]
    min_norm: NotRequired[float]
    max_norm: NotRequired[float]

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
    params: FuseParams | dict[str, object] | None = None,
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
