"""Typed attribute helpers that do not use ``getattr``.

``getattr`` produces ``Unknown`` under strict checkers and hides missing
API contracts. Prefer direct attribute access on protocols. Use these
helpers only for dynamic names or third-party objects without stubs.
"""


def optional_attr(obj: object, name: str) -> object | None:
    """Return ``obj.name`` or ``None`` when the attribute is absent."""
    try:
        return object.__getattribute__(obj, name)
    except AttributeError:
        return None
