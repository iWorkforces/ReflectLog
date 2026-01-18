"""Security utilities for ReflectLogMCP Server."""

import re
from typing import Any


class SecretString:
    """A wrapper for sensitive strings that prevents accidental exposure in logs.

    This class wraps sensitive values like API keys to ensure they are not
    accidentally logged or displayed. The actual value is only accessible
    via the `get_secret_value()` method.

    Example:
        >>> secret = SecretString("sk-abc123")
        >>> print(secret)
        ***REDACTED***
        >>> secret.get_secret_value()
        'sk-abc123'
    """

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        """Initialize with a secret value.

        Args:
            value: The sensitive string to protect.
        """
        self._value = value

    def get_secret_value(self) -> str:
        """Get the actual secret value.

        Returns:
            The underlying secret string.
        """
        return self._value

    def __str__(self) -> str:
        """Return redacted representation for printing."""
        return "***REDACTED***"

    def __repr__(self) -> str:
        """Return redacted representation for debugging."""
        return "SecretString('***REDACTED***')"

    def __bool__(self) -> bool:
        """Return True if the secret value is non-empty."""
        return bool(self._value)

    def __len__(self) -> int:
        """Return the length of the secret value."""
        return len(self._value)


# Patterns for sensitive data detection
_SENSITIVE_PATTERNS = [
    # API keys
    (re.compile(r"sk-[a-zA-Z0-9]{20,}", re.IGNORECASE), "[API_KEY_REDACTED]"),
    (
        re.compile(r"api[_-]?key[\"']?\s*[:=]\s*[\"']?[a-zA-Z0-9_-]+", re.IGNORECASE),
        "[API_KEY_REDACTED]",
    ),
    # Bearer tokens
    (re.compile(r"bearer\s+[a-zA-Z0-9_.-]+", re.IGNORECASE), "[BEARER_TOKEN_REDACTED]"),
    # Passwords
    (
        re.compile(r"password[\"']?\s*[:=]\s*[\"']?[^\s\"']+", re.IGNORECASE),
        "[PASSWORD_REDACTED]",
    ),
    # Email addresses
    (re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), "[EMAIL_REDACTED]"),
]


def sanitize_for_logging(
    value: Any,
    max_length: int = 200,
    redact_sensitive: bool = True,
) -> str:
    """Sanitize a value for safe logging.

    This function prepares values for logging by:
    1. Converting to string
    2. Truncating to max_length
    3. Optionally redacting sensitive patterns (API keys, passwords, etc.)

    Args:
        value: The value to sanitize (will be converted to string).
        max_length: Maximum length before truncation (default: 200).
        redact_sensitive: Whether to redact sensitive patterns (default: True).

    Returns:
        A sanitized string safe for logging.

    Example:
        >>> sanitize_for_logging("My API key is sk-abc123def456")
        'My API key is [API_KEY_REDACTED]'
        >>> sanitize_for_logging("x" * 500, max_length=100)
        'xxxx...xxxx (truncated, original length: 500)'
    """
    # Convert to string
    text = str(value) if value is not None else ""

    # Redact sensitive patterns
    if redact_sensitive:
        for pattern, replacement in _SENSITIVE_PATTERNS:
            text = pattern.sub(replacement, text)

    # Truncate if needed
    if len(text) > max_length:
        # Show beginning and end for context
        half = (max_length - 20) // 2
        text = f"{text[:half]}...{text[-half:]} (truncated, original length: {len(str(value))})"

    return text


def validate_project_id(project_id: str) -> str:
    """Validate project_id to prevent path traversal attacks.

    This function validates that a project_id contains only safe characters
    and does not contain path traversal patterns like "../" that could allow
    writing files outside the intended directory.

    Args:
        project_id: The project identifier to validate.

    Returns:
        The validated project_id (lowercased).

    Raises:
        ValidationError: If project_id contains invalid characters or path patterns.

    Example:
        >>> validate_project_id("my-project-123")
        'my-project-123'
        >>> validate_project_id("../../../etc")
        ValidationError: Invalid project_id: ../../../etc
    """
    from ..exceptions import ValidationError

    if not project_id:
        raise ValidationError("project_id cannot be empty")

    # Check for path traversal patterns
    if ".." in project_id or project_id.startswith("/"):
        raise ValidationError(f"Invalid project_id: {project_id}")

    # Allow only alphanumeric, hyphen, underscore, and dot
    if not re.match(r"^[a-zA-Z0-9_.-]+$", project_id):
        raise ValidationError(
            f"project_id contains invalid characters: {project_id}. "
            "Only alphanumeric, hyphens, underscores, and dots are allowed."
        )

    # Length limit (prevent excessively long project IDs)
    if len(project_id) > 128:
        raise ValidationError(
            f"project_id too long (max 128 characters): {len(project_id)}"
        )

    return project_id.lower()


def redact_dict_secrets(
    data: dict[str, Any],
    secret_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Create a copy of a dictionary with secret values redacted.

    This is useful for safely logging configuration dictionaries that may
    contain sensitive values.

    Args:
        data: The dictionary to redact.
        secret_keys: Set of key names to redact (case-insensitive).
            Defaults to common secret key names.

    Returns:
        A new dictionary with secret values replaced by "[REDACTED]".

    Example:
        >>> config = {"api_key": "sk-secret", "model": "gpt-4"}
        >>> redact_dict_secrets(config)
        {'api_key': '[REDACTED]', 'model': 'gpt-4'}
    """
    if secret_keys is None:
        secret_keys = {
            "api_key",
            "apikey",
            "api-key",
            "secret",
            "password",
            "token",
            "key",
            "openrouter_api_key",
            "openai_api_key",
        }

    secret_keys_lower = {k.lower() for k in secret_keys}

    result: dict[str, Any] = {}
    for key, value in data.items():
        if key.lower() in secret_keys_lower:
            result[key] = "[REDACTED]"
        elif isinstance(value, dict):
            result[key] = redact_dict_secrets(value, secret_keys)
        elif isinstance(value, SecretString):
            result[key] = str(value)  # Returns "***REDACTED***"
        else:
            result[key] = value

    return result
