"""Memory validation utilities for ReflectLog Server."""

from typing import TYPE_CHECKING
import unicodedata

if TYPE_CHECKING:
    from collections.abc import Sequence

# Allowed control characters (tab, newline, carriage return)
_ALLOWED_CONTROL_CHARS = {"\t", "\n", "\r"}


def validate_memories(
    memories: Sequence[object], min_length: int, max_length: int
) -> tuple[bool, str | None]:
    """Validate memory list for add/remove operations.

    Args:
        memories: List of memories to validate.
        min_length: Minimum allowed memory length in characters.
        max_length: Maximum allowed memory length in characters.

    Returns:
        Tuple of (is_valid, error_message).
        If valid, error_message is None.
        If invalid, error_message contains the validation error.
    """
    if not memories:
        # Empty list is a valid no-op for add/remove operations.
        return True, None

    for i, memory in enumerate(memories):
        # Check for None first (before type check)
        if memory is None:
            return False, f"Memory at index {i} is None"

        # Check type
        if not isinstance(memory, str):
            return False, f"Memory at index {i} is not a string: {type(memory)}"

        # Check for whitespace-only content (before length check)
        if not memory.strip():
            return False, f"Memory at index {i} contains only whitespace"

        # Check minimum length
        if len(memory) < min_length:
            return (
                False,
                f"Memory at index {i} is too short (min: {min_length} chars)",
            )

        # Check maximum length
        if len(memory) > max_length:
            return (
                False,
                f"Memory at index {i} is too long (max: {max_length} chars)",
            )

        # Check for disallowed control characters
        for char in memory:
            if (
                unicodedata.category(char) == "Cc"
                and char not in _ALLOWED_CONTROL_CHARS
            ):
                return (
                    False,
                    f"Memory at index {i} contains control characters at position {memory.index(char)}",
                )

    return True, None


def validate_add_batch(
    memories: list[str], max_batch: int, max_chars: int
) -> tuple[bool, str | None]:
    """Reject oversized add payloads before embed/persist."""
    if len(memories) > max_batch:
        return False, f"Too many memories in one add (max: {max_batch})"
    total_chars = sum(len(item) for item in memories)
    if total_chars > max_chars:
        return False, f"Add payload too large (max: {max_chars} characters)"
    return True, None


def validate_remove_batch(
    memories: list[str], max_batch: int, max_chars: int
) -> tuple[bool, str | None]:
    """Reject oversized remove payloads after uniqueness."""
    if len(memories) > max_batch:
        return False, f"Too many memories in one remove (max: {max_batch})"
    total_chars = sum(len(item) for item in memories)
    if total_chars > max_chars:
        return False, f"Remove payload too large (max: {max_chars} characters)"
    return True, None


def validate_remove_memories(memories: Sequence[object]) -> tuple[bool, str | None]:
    """Reject empty or non-string remove targets without add length/control rules.

    Stored rows may predate a later min/max or control-character policy.
    Whitespace-only strings are still rejected as they cannot identify a row.
    """
    if not memories:
        return True, None
    for i, memory in enumerate(memories):
        if memory is None:
            return False, f"Memory at index {i} is None"
        if not isinstance(memory, str):
            return False, f"Memory at index {i} is not a string: {type(memory)}"
        if not memory.strip():
            return False, f"Memory at index {i} contains only whitespace"
    return True, None
