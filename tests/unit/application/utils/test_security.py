'''Unit tests for reflectlog.application.utils.security module.'''

import pytest

from reflectlog.core.exceptions import ValidationError
from reflectlog.application.utils.security import (
    SecretString,
    redact_dict_secrets,
    sanitize_for_logging,
    validate_project_id,
)


class TestSecretString:
    '''Tests for SecretString class.'''

    def test_get_secret_value(self) -> None:
        '''Test get_secret_value returns the actual secret.'''
        secret = SecretString("my-secret-api-key")
        assert secret.get_secret_value() == "my-secret-api-key"

    def test_str_returns_redacted(self) -> None:
        '''Test __str__ returns redacted value.'''
        secret = SecretString("my-secret-api-key")
        assert str(secret) == "***REDACTED***"

    def test_repr_returns_redacted(self) -> None:
        '''Test __repr__ returns redacted representation.'''
        secret = SecretString("my-secret-api-key")
        assert repr(secret) == "***REDACTED***"

    def test_bool_true_for_non_empty(self) -> None:
        '''Test __bool__ returns True for non-empty secret.'''
        secret = SecretString("value")
        assert bool(secret) is True

    def test_bool_false_for_empty(self) -> None:
        '''Test __bool__ returns False for empty secret.'''
        secret = SecretString("")
        assert bool(secret) is False

    def test_len_returns_secret_length(self) -> None:
        '''Test __len__ returns the length of the secret value.'''
        secret = SecretString("12345")
        assert len(secret) == 5

    def test_len_empty_string(self) -> None:
        '''Test __len__ returns 0 for empty secret.'''
        secret = SecretString("")
        assert len(secret) == 0

    def test_slots_prevent_dict_creation(self) -> None:
        '''Test __slots__ prevents __dict__ attribute.'''
        secret = SecretString("test")
        assert not hasattr(secret, "__dict__")

    def test_print_does_not_expose_secret(self) -> None:
        '''Test that printing does not expose the secret.'''
        secret = SecretString("sk-abc123def456")
        output = f"API Key: {secret}"
        assert "sk-abc123def456" not in output
        assert "***REDACTED***" in output


class TestSanitizeForLogging:
    '''Tests for sanitize_for_logging function.'''

    def test_basic_string(self) -> None:
        '''Test basic string passthrough.'''
        result = sanitize_for_logging("Hello World")
        assert result == "Hello World"

    def test_redact_api_key_sk_prefix(self) -> None:
        '''Test redaction of sk- prefixed API keys.'''
        # API key pattern requires 20+ characters after sk-
        result = sanitize_for_logging("My key is sk-abc123def456ghij78901234")
        assert "sk-" not in result
        assert "[API_KEY_REDACTED]" in result

    def test_redact_api_key_assignment(self) -> None:
        '''Test redaction of api_key assignments.'''
        result = sanitize_for_logging("api_key = my-secret-key")
        assert "[API_KEY_REDACTED]" in result

    def test_redact_bearer_token(self) -> None:
        '''Test redaction of bearer tokens.'''
        # Bearer token pattern requires 20+ characters after "bearer "
        result = sanitize_for_logging("Authorization: Bearer abc123tokenxyz789def456")
        assert "[BEARER_TOKEN_REDACTED]" in result

    def test_redact_password(self) -> None:
        '''Test redaction of passwords.'''
        result = sanitize_for_logging("password='secret123'")
        assert "[PASSWORD_REDACTED]" in result

    def test_redact_email(self) -> None:
        '''Test redaction of email addresses.'''
        result = sanitize_for_logging("Contact: user@example.com")
        assert "[EMAIL_REDACTED]" in result

    def test_truncation(self) -> None:
        '''Test truncation of long strings.'''
        long_string = "x" * 500
        result = sanitize_for_logging(long_string, max_length=100)
        assert len(result) < len(long_string)
        assert "truncated" in result
        assert "500" in result  # Original length mentioned

    def test_truncation_shows_beginning_and_end(self) -> None:
        '''Test truncation shows beginning and end of string.'''
        # Create a string with distinct beginning and end
        long_string = "START" + "x" * 500 + "END"
        result = sanitize_for_logging(long_string, max_length=100)
        assert "START" in result
        assert "END" in result

    def test_disable_redaction(self) -> None:
        '''Test disabling sensitive data redaction.'''
        text = "api_key = secret123"
        result = sanitize_for_logging(text, redact_sensitive=False)
        assert "[API_KEY_REDACTED]" not in result
        assert "secret123" in result

    def test_none_value(self) -> None:
        '''Test handling of None value.'''
        result = sanitize_for_logging(None)
        assert result == ""

    def test_non_string_value(self) -> None:
        '''Test handling of non-string values.'''
        result = sanitize_for_logging(12345)
        assert result == "12345"

    def test_dict_value(self) -> None:
        '''Test handling of dict values.'''
        result = sanitize_for_logging({"key": "value"})
        assert "key" in result
        assert "value" in result


class TestRedactDictSecrets:
    '''Tests for redact_dict_secrets function.'''

    def test_redact_default_secret_keys(self) -> None:
        '''Test redaction of default secret keys.'''
        data = {
            "api_key": "sk-secret",
            "password": "mypassword",
            "token": "mytoken",
            "model": "gpt-4",
        }
        result = redact_dict_secrets(data)
        assert result["api_key"] == "[REDACTED]"
        assert result["password"] == "[REDACTED]"
        assert result["token"] == "[REDACTED]"
        assert result["model"] == "gpt-4"

    def test_case_insensitive_keys(self) -> None:
        '''Test redaction is case-insensitive.'''
        data = {
            "API_KEY": "secret",
            "Password": "secret",
            "TOKEN": "secret",
        }
        result = redact_dict_secrets(data)
        assert result["API_KEY"] == "[REDACTED]"
        assert result["Password"] == "[REDACTED]"
        assert result["TOKEN"] == "[REDACTED]"

    def test_nested_dict_redaction(self) -> None:
        '''Test redaction in nested dictionaries.'''
        data = {
            "config": {
                "api_key": "secret",
                "model": "gpt-4",
            },
            "name": "test",
        }
        result = redact_dict_secrets(data)
        assert result["config"]["api_key"] == "[REDACTED]"
        assert result["config"]["model"] == "gpt-4"
        assert result["name"] == "test"

    def test_custom_secret_keys(self) -> None:
        '''Test custom secret keys.'''
        data = {
            "custom_secret": "value",
            "normal_field": "value",
        }
        result = redact_dict_secrets(data, secret_keys={"custom_secret"})
        assert result["custom_secret"] == "[REDACTED]"
        assert result["normal_field"] == "value"

    def test_secret_string_value(self) -> None:
        '''Test redaction of SecretString values.'''
        # Use a non-secret key name to test SecretString handling specifically
        data = {
            "config_value": SecretString("my-secret"),
            "other": "value",
        }
        result = redact_dict_secrets(data)
        # SecretString's __str__ returns "***REDACTED***"
        assert result["config_value"] == "***REDACTED***"
        assert result["other"] == "value"

    def test_returns_new_dict(self) -> None:
        '''Test that redact_dict_secrets returns a new dict.'''
        original = {"api_key": "secret"}
        result = redact_dict_secrets(original)
        assert result is not original
        assert original["api_key"] == "secret"  # Original unchanged

    def test_empty_dict(self) -> None:
        '''Test handling of empty dict.'''
        result = redact_dict_secrets({})
        assert result == {}

    def test_openrouter_api_key(self) -> None:
        '''Test redaction of openrouter_api_key.'''
        data = {"openrouter_api_key": "sk-or-abc123"}
        result = redact_dict_secrets(data)
        assert result["openrouter_api_key"] == "[REDACTED]"

    def test_deeply_nested_dicts(self) -> None:
        '''Test redaction in deeply nested structures.'''
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "api_key": "deep_secret",
                    },
                },
            },
        }
        result = redact_dict_secrets(data)
        assert result["level1"]["level2"]["level3"]["api_key"] == "[REDACTED]"


class TestValidateProjectId:
    '''Tests for validate_project_id() function.'''

    def test_valid_alphanumeric_project_id(self) -> None:
        '''Test valid alphanumeric project_id passes validation.'''
        result = validate_project_id("my-project-123")
        assert result == "my-project-123"

    def test_valid_with_dots_project_id(self) -> None:
        '''Test valid project_id with dots passes validation.'''
        result = validate_project_id("my.project.name")
        assert result == "my.project.name"

    def test_lowercase_conversion(self) -> None:
        '''Test project_id is lowercased.'''
        result = validate_project_id("My-Project-123")
        assert result == "my-project-123"

    def test_empty_project_id_raises_error(self) -> None:
        '''Test empty project_id raises ValidationError.'''
        with pytest.raises(ValidationError, match="project_id cannot be empty"):
            validate_project_id("")

    def test_path_traversal_double_dot_raises_error(self) -> None:
        '''Test path traversal pattern (..) raises ValidationError.'''
        with pytest.raises(ValidationError, match="Invalid project_id"):
            validate_project_id("../../../etc")

    def test_path_traversal_leading_slash_raises_error(self) -> None:
        '''Test leading slash raises ValidationError.'''
        with pytest.raises(ValidationError, match="Invalid project_id"):
            validate_project_id("/etc/passwd")

    def test_invalid_characters_raises_error(self) -> None:
        '''Test invalid characters raise ValidationError.'''
        with pytest.raises(ValidationError, match="contains invalid characters"):
            validate_project_id("project@id")

    def test_max_length_enforced(self) -> None:
        '''Test maximum length constraint (128 characters).'''
        with pytest.raises(ValidationError, match="too long"):
            validate_project_id("a" * 129)

    def test_max_length_boundary(self) -> None:
        '''Test 128 character project_id passes validation.'''
        result = validate_project_id("a" * 128)
        assert result == "a" * 128

    def test_special_characters_not_allowed(self) -> None:
        '''Test special characters are not allowed.'''
        with pytest.raises(ValidationError, match="contains invalid characters"):
            validate_project_id("project$id")
