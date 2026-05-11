"""Characterization tests for security module equivalence.

Proves that reflectlog.utility.security.validate_project_id and
reflectlog.application.utils.security.validate_project_id have identical
behavior before module consolidation.
"""

import inspect

import pytest

from reflectlog.application.utils.security import (
    SecretString,
    redact_dict_secrets,
    sanitize_for_logging,
    validate_project_id as app_validate,
)
from reflectlog.core.exceptions import ValidationError
from reflectlog.utility.security import validate_project_id as util_validate


class TestValidateProjectIdEquivalence:
    """Prove both validate_project_id implementations are identical."""

    def test_function_signature_matches(self) -> None:
        """Both functions have the same signature."""
        util_sig = inspect.signature(util_validate)
        app_sig = inspect.signature(app_validate)
        assert list(util_sig.parameters.keys()) == list(app_sig.parameters.keys())

    @pytest.mark.parametrize(
        "project_id,expected",
        [
            ("my-project", "my-project"),
            ("test_project", "test_project"),
            ("project.v2", "project.v2"),
            ("MyProject", "myproject"),
            ("ABC-123", "abc-123"),
            ("a", "a"),
            ("a" * 128, "a" * 128),
            ("simple123", "simple123"),
            ("dotted.name.here", "dotted.name.here"),
            ("under_score", "under_score"),
            ("MiXeD.CaSe-123_test", "mixed.case-123_test"),
        ],
        ids=[
            "hyphenated",
            "underscored",
            "dotted",
            "uppercase-lowered",
            "mixed-case-lowered",
            "single-char",
            "max-length-128",
            "simple-alphanumeric",
            "multi-dotted",
            "underscore-only",
            "all-allowed-chars",
        ],
    )
    def test_valid_ids_both_paths(self, project_id: str, expected: str) -> None:
        """Both functions accept valid IDs and return the same lowercased result."""
        assert util_validate(project_id) == expected
        assert app_validate(project_id) == expected

    @pytest.mark.parametrize(
        "project_id,error_match",
        [
            ("", "cannot be empty"),
            ("../etc", "Invalid project_id"),
            ("../../passwd", "Invalid project_id"),
            ("/absolute/path", "Invalid project_id"),
            ("has space", "invalid characters"),
            ("has@symbol", "invalid characters"),
            ("has!bang", "invalid characters"),
            ("path/slash", "invalid characters"),
            ("back\\slash", "invalid characters"),
            ("semi;colon", "invalid characters"),
            ("a" * 129, "too long"),
            ("has\nnewline", "invalid characters"),
            ("has\ttab", "invalid characters"),
        ],
        ids=[
            "empty",
            "path-traversal-simple",
            "path-traversal-deep",
            "absolute-path",
            "space",
            "at-symbol",
            "exclamation",
            "forward-slash",
            "backslash",
            "semicolon",
            "too-long-129",
            "newline",
            "tab",
        ],
    )
    def test_invalid_ids_both_paths(self, project_id: str, error_match: str) -> None:
        """Both functions reject invalid IDs with the same error type."""
        with pytest.raises(ValidationError, match=error_match):
            util_validate(project_id)

        with pytest.raises(ValidationError, match=error_match):
            app_validate(project_id)

    def test_raises_validation_error_type(self) -> None:
        """Both raise reflectlog.core.exceptions.ValidationError, not built-in."""
        for fn in (util_validate, app_validate):
            with pytest.raises(ValidationError) as exc_info:
                fn("")
            assert exc_info.type is ValidationError

    def test_path_traversal_after_lowercase(self) -> None:
        """Path traversal check happens after lowercasing."""
        for fn in (util_validate, app_validate):
            with pytest.raises(ValidationError, match="Invalid project_id"):
                fn("test..hidden")

    def test_double_dot_anywhere_rejected(self) -> None:
        """'..' anywhere in the string is rejected, not just at start."""
        for fn in (util_validate, app_validate):
            with pytest.raises(ValidationError, match="Invalid project_id"):
                fn("safe..notreally")


class TestSecretString:
    """Tests for SecretString in application/utils/security.py."""

    def test_str_redacted(self) -> None:
        """str() returns redacted string."""
        secret = SecretString("my-api-key")
        assert str(secret) == "***REDACTED***"

    def test_repr_redacted(self) -> None:
        """repr() returns redacted string."""
        secret = SecretString("my-api-key")
        assert repr(secret) == "***REDACTED***"

    def test_get_secret_value(self) -> None:
        """get_secret_value() returns the actual value."""
        secret = SecretString("sk-abc123")
        assert secret.get_secret_value() == "sk-abc123"

    def test_bool_nonempty(self) -> None:
        """Non-empty secret is truthy."""
        assert bool(SecretString("value"))

    def test_bool_empty(self) -> None:
        """Empty secret is falsy."""
        assert not bool(SecretString(""))

    def test_len(self) -> None:
        """len() returns length of the underlying value."""
        assert len(SecretString("hello")) == 5
        assert len(SecretString("")) == 0

    def test_equality(self) -> None:
        """Two SecretStrings with same value are equal."""
        a = SecretString("same")
        b = SecretString("same")
        assert a == b

    def test_inequality(self) -> None:
        """Two SecretStrings with different values are not equal."""
        a = SecretString("one")
        b = SecretString("two")
        assert a != b

    def test_not_equal_to_plain_string(self) -> None:
        """SecretString is not equal to a plain string."""
        secret = SecretString("value")
        assert secret != "value"

    def test_hashable(self) -> None:
        """SecretString can be used in sets/dicts."""
        s = SecretString("key")
        d = {s: "val"}
        assert d[s] == "val"


class TestSanitizeForLogging:
    """Tests for sanitize_for_logging in application/utils/security.py."""

    def test_basic_string(self) -> None:
        """Basic string passes through unchanged."""
        assert sanitize_for_logging("hello world") == "hello world"

    def test_none_becomes_empty(self) -> None:
        """None is converted to empty string."""
        assert sanitize_for_logging(None) == ""

    def test_truncation(self) -> None:
        """Long strings are truncated with ellipsis."""
        long_text = "x" * 500
        result = sanitize_for_logging(long_text, max_length=100)
        assert len(result) < 500
        assert "truncated" in result

    def test_api_key_redacted(self) -> None:
        """API keys matching sk-... pattern are redacted."""
        text = "key is sk-abcdefghijklmnopqrstuvwxyz"
        result = sanitize_for_logging(text)
        assert "sk-" not in result
        assert "REDACTED" in result

    def test_no_redaction_when_disabled(self) -> None:
        """Sensitive patterns pass through when redaction is disabled."""
        text = "key is sk-abcdefghijklmnopqrstuvwxyz"
        result = sanitize_for_logging(text, redact_sensitive=False)
        assert "sk-" in result


class TestRedactDictSecrets:
    """Tests for redact_dict_secrets in application/utils/security.py."""

    def test_redacts_api_key(self) -> None:
        """API key values are redacted."""
        data = {"api_key": "sk-secret123", "model": "gpt-4"}
        result = redact_dict_secrets(data)
        assert result["api_key"] == "[REDACTED]"
        assert result["model"] == "gpt-4"

    def test_case_insensitive_keys(self) -> None:
        """Key matching is case-insensitive."""
        data = {"API_KEY": "secret", "Password": "secret"}
        result = redact_dict_secrets(data)
        assert result["API_KEY"] == "[REDACTED]"
        assert result["Password"] == "[REDACTED]"

    def test_nested_dict_redaction(self) -> None:
        """Nested dictionaries are recursively redacted."""
        data = {"outer": {"api_key": "secret", "value": "ok"}}
        result = redact_dict_secrets(data)
        assert result["outer"]["api_key"] == "[REDACTED]"
        assert result["outer"]["value"] == "ok"

    def test_list_with_dicts(self) -> None:
        """Lists containing dicts have secrets redacted."""
        data = {"items": [{"token": "abc"}, {"name": "safe"}]}
        result = redact_dict_secrets(data)
        assert result["items"][0]["token"] == "[REDACTED]"
        assert result["items"][1]["name"] == "safe"

    def test_tuple_with_dicts(self) -> None:
        """Tuples containing dicts have secrets redacted and type preserved."""
        data = {"items": ({"secret": "hidden"},)}
        result = redact_dict_secrets(data)
        assert isinstance(result["items"], tuple)
        assert result["items"][0]["secret"] == "[REDACTED]"

    def test_secret_string_redacted(self) -> None:
        """SecretString values are converted to redacted form."""
        data = {"my_val": SecretString("hidden")}
        result = redact_dict_secrets(data)
        assert result["my_val"] == "***REDACTED***"

    def test_custom_secret_keys(self) -> None:
        """Custom secret key set overrides defaults."""
        data = {"custom_field": "sensitive", "api_key": "visible"}
        result = redact_dict_secrets(data, secret_keys={"custom_field"})
        assert result["custom_field"] == "[REDACTED]"
        assert result["api_key"] == "visible"

    def test_original_not_modified(self) -> None:
        """Original dictionary is not mutated."""
        data = {"api_key": "secret"}
        _ = redact_dict_secrets(data)
        assert data["api_key"] == "secret"
