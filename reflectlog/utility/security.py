"""Security utilities for input validation.

This module provides security validation functions that are safe
to use across all layers (utility, infrastructure, application).
"""

import re


def validate_workspace_id(workspace_id: str) -> str:
    """Strip and lowercase a safe 1..64-character ASCII workspace identifier.

    Only A-Za-z0-9_.- are allowed; '.' and every '..' occurrence are rejected.
    Invalid identifiers raise core ValidationError without exposing raw input.
    """
    from reflectlog.core.exceptions import ValidationError

    normalized = workspace_id.strip()
    if not normalized:
        raise ValidationError("workspace_id cannot be empty")
    if len(normalized) > 64:
        raise ValidationError(
            "workspace_id too long (max 64 characters after stripping)"
        )
    if re.fullmatch(r"[A-Za-z0-9_.-]+", normalized, flags=re.ASCII) is None:
        raise ValidationError(
            "Invalid workspace_id: contains invalid characters. "
            "Only ASCII A-Za-z0-9_.- are allowed."
        )
    if normalized == "." or ".." in normalized:
        raise ValidationError(
            "Invalid workspace_id: Path traversal patterns '.' and '..' are not allowed"
        )
    return normalized.lower()
