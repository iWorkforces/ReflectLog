"""Memory validation utilities for ReflectLog Server."""

import re
from typing import Any
import unicodedata

# SQL injection patterns for basic detection
_SQL_INJECTION_PATTERNS = [
    r"(?i)\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION)\b.*\bFROM\b",
    r"(?i)\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION)\b.*\bWHERE\b",
    r"(?i)\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION)\b.*\bINTO\b",
    r"(?i);.*\b(DROP|DELETE|UPDATE|INSERT)\b",
    r"(?i)'.*OR.*'.*=.*'",
    r"(?i)'.*UNION.*SELECT",
    r"(?i)'.*;.*DROP",
    r"(?i)\bEXEC\b.*\bXP_CMD\b",
    r"(?i)\bEXEC\b.*\bSP_OA\b",
]

# Pre-compiled regex patterns for performance
_SQL_INJECTION_REGEX = [re.compile(pattern) for pattern in _SQL_INJECTION_PATTERNS]

# Allowed control characters (tab, newline, carriage return)
_ALLOWED_CONTROL_CHARS = {"\t", "\n", "\r"}


def validate_memories(
    memories: list[Any], min_length: int, max_length: int
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

        # Check for SQL injection patterns
        for pattern in _SQL_INJECTION_REGEX:
            if pattern.search(memory):
                return (
                    False,
                    f"Memory at index {i} contains potentially harmful content (SQL injection pattern)",
                )

    return True, None


def truncate_memory(content: str, max_length: int = 100) -> str:
    """Truncate a memory for display purposes (Unicode-aware).

    This function avoids splitting Unicode grapheme clusters (combining characters,
    emoji sequences, etc.) by checking if the truncation point would split a
    combining character from its base character.

    Args:
        content: The memory content to truncate.
        max_length: Maximum length for display (in code points).

    Returns:
        Truncated memory with ellipsis if needed.
    """
    import unicodedata

    if len(content) <= max_length:
        return content

    # Find a safe truncation point that won't split a grapheme cluster
    # Check if the character at max_length is a combining character
    if max_length < len(content):
        # If the next character is a combining mark, backtrack to avoid splitting
        next_char = content[max_length]
        if unicodedata.combining(next_char) != 0:
            # Backtrack to find a non-combining character
            i = max_length - 1
            while i > 0 and unicodedata.combining(content[i]) != 0:
                i -= 1
            if i > 0:  # Only truncate if we found a safe point
                return f"{content[:i]}..."

    return f"{content[:max_length]}..."
