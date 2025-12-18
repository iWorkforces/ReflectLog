"""Type stubs for numba.core.errors module."""

class NumbaWarning(Warning):
    """Base class for Numba warnings."""

    ...

class NumbaTypeSafetyWarning(NumbaWarning):
    """Warning for type safety issues in Numba.

    This warning is raised when Numba detects potential type safety
    issues in JIT-compiled code, particularly in the ranx library.
    """

    ...

class NumbaDeprecationWarning(NumbaWarning):
    """Warning for deprecated Numba features."""

    ...

class NumbaPerformanceWarning(NumbaWarning):
    """Warning for performance issues in Numba code."""

    ...

class TypingError(Exception):
    """Exception for type inference errors."""

    ...

class UnsupportedError(Exception):
    """Exception for unsupported features."""

    ...

class CompilerError(Exception):
    """Exception for compiler errors."""

    ...
