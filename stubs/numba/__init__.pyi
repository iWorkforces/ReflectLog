"""Type stubs for numba library.

This provides type hints for the numba JIT compiler decorators and functions
used in reflectlog.application.utils.numba_utils.
"""

from typing import Any, Callable, TypeVar, overload

# Type variable for generic function signatures
F = TypeVar("F", bound=Callable[..., Any])

class config:
    """Numba configuration namespace."""

    THREADING_LAYER: str
    NUMBA_DEFAULT_NUM_THREADS: int
    CACHE_DIR: str

@overload
def jit(
    func: F,
) -> F: ...
@overload
def jit(
    *,
    nopython: bool = False,
    cache: bool = False,
    fastmath: bool = False,
    parallel: bool = False,
    nogil: bool = False,
    forceobj: bool = False,
) -> Callable[[F], F]: ...
def jit(
    func: F | None = None,
    *,
    nopython: bool = False,
    cache: bool = False,
    fastmath: bool = False,
    parallel: bool = False,
    nogil: bool = False,
    forceobj: bool = False,
) -> F | Callable[[F], F]:
    """Just-In-Time compiler decorator.

    Args:
        func: Function to compile (if used without parentheses).
        nopython: If True, compile in nopython mode (no Python objects).
        cache: If True, cache compiled function to disk.
        fastmath: If True, enable fast-math optimizations.
        parallel: If True, enable automatic parallelization.
        nogil: If True, release GIL during execution.
        forceobj: If True, compile in object mode.

    Returns:
        Compiled function or decorator.
    """
    ...

@overload
def njit(
    func: F,
) -> F: ...
@overload
def njit(
    *,
    cache: bool = False,
    fastmath: bool = False,
    parallel: bool = False,
    nogil: bool = False,
) -> Callable[[F], F]: ...
def njit(
    func: F | None = None,
    *,
    cache: bool = False,
    fastmath: bool = False,
    parallel: bool = False,
    nogil: bool = False,
) -> F | Callable[[F], F]:
    """No-Python mode JIT compiler decorator.

    Equivalent to @jit(nopython=True).

    Args:
        func: Function to compile (if used without parentheses).
        cache: If True, cache compiled function to disk.
        fastmath: If True, enable fast-math optimizations.
        parallel: If True, enable automatic parallelization.
        nogil: If True, release GIL during execution.

    Returns:
        Compiled function or decorator.
    """
    ...

def prange(start: int, stop: int | None = None, step: int = 1) -> range:
    """Parallel range for use in numba parallel loops.

    Args:
        start: Start value (or stop if stop is None).
        stop: Stop value.
        step: Step value.

    Returns:
        Range-like object for parallel iteration.
    """
    ...

def vectorize(
    signatures: list[str] | None = None,
    *,
    nopython: bool = False,
    target: str = "cpu",
    cache: bool = False,
    fastmath: bool = False,
) -> Callable[[F], F]:
    """Create a NumPy ufunc from a scalar function.

    Args:
        signatures: List of type signatures.
        nopython: If True, compile in nopython mode.
        target: Compilation target ('cpu', 'parallel', 'cuda').
        cache: If True, cache compiled function.
        fastmath: If True, enable fast-math optimizations.

    Returns:
        Decorator that creates a ufunc.
    """
    ...

def guvectorize(
    signatures: list[str],
    layout: str,
    *,
    nopython: bool = False,
    target: str = "cpu",
    cache: bool = False,
    fastmath: bool = False,
) -> Callable[[F], F]:
    """Create a generalized NumPy ufunc.

    Args:
        signatures: List of type signatures.
        layout: Input/output layout specification.
        nopython: If True, compile in nopython mode.
        target: Compilation target ('cpu', 'parallel', 'cuda').
        cache: If True, cache compiled function.
        fastmath: If True, enable fast-math optimizations.

    Returns:
        Decorator that creates a gufunc.
    """
    ...
