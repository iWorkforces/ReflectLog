"""Message validation utilities for ReflectLogMCP Server."""

import re
import unicodedata
from typing import Any

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


def validate_messages(
    messages: list[Any], min_length: int, max_length: int
) -> tuple[bool, Optional[str]]:
    """Validate message list for add/remove operations.

    Args:
        messages: List of messages to validate.
        min_length: Minimum allowed message length in characters.
        max_length: Maximum allowed message length in characters.

    Returns:
        Tuple of (is_valid, error_message).
        If valid, error_message is None.
        If invalid, error_message contains the validation error.
    """
    if not messages:
        # Empty list is a valid no-op for add/remove operations.
        return True, None

    for i, msg in enumerate(messages):
        # Check for None first (before type check)
        if msg is None:
            return False, f"Message at index {i} is None"

        # Check type
        if not isinstance(msg, str):
            return False, f"Message at index {i} is not a string: {type(msg)}"

        # Check for whitespace-only content (before length check)
        if not msg.strip():
            return False, f"Message at index {i} contains only whitespace"

        # Check minimum length
        if len(msg) < min_length:
            return (
                False,
                f"Message at index {i} is too short (min: {min_length} chars)",
            )

        # Check maximum length
        if len(msg) > max_length:
            return (
                False,
                f"Message at index {i} is too long (max: {max_length} chars)",
            )

        # Check for disallowed control characters
        for char in msg:
            if (
                unicodedata.category(char) == "Cc"
                and char not in _ALLOWED_CONTROL_CHARS
            ):
                return (
                    False,
                    f"Message at index {i} contains control characters at position {msg.index(char)}",
                )

        # Check for SQL injection patterns
        for pattern in _SQL_INJECTION_REGEX:
            if pattern.search(msg):
                return (
                    False,
                    f"Message at index {i} contains potentially harmful content (SQL injection pattern)",
                )

    return True, None


def truncate_message(message: str, max_length: int = 100) -> str:
    """Truncate a message for display purposes (Unicode-aware).

    This function avoids splitting Unicode grapheme clusters (combining characters,
    emoji sequences, etc.) by checking if the truncation point would split a
    combining character from its base character.

    Args:
        message: The message to truncate.
        max_length: Maximum length for display (in code points).

    Returns:
        Truncated message with ellipsis if needed.
    """
    import unicodedata

    if len(message) <= max_length:
        return message

    # Find a safe truncation point that won't split a grapheme cluster
    # Check if the character at max_length is a combining character
    if max_length < len(message):
        # If the next character is a combining mark, backtrack to avoid splitting
        next_char = message[max_length]
        if unicodedata.combining(next_char) != 0:
            # Backtrack to find a non-combining character
            i = max_length - 1
            while i > 0 and unicodedata.combining(message[i]) != 0:
                i -= 1
            if i > 0:  # Only truncate if we found a safe point
                return f"{message[:i]}..."

    return f"{message[:max_length]}..."
