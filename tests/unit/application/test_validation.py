"""Tests for message validation logic."""

import pytest

from ccmemories.application.utils.validation import validate_messages


@pytest.mark.unit
class TestValidateMessages:
    """Test suite for validate_messages() function."""

    def test_empty_list_is_valid(self):
        """Test that empty list is valid (no-op case)."""
        is_valid, error_msg = validate_messages([], 1, 30720)
        assert is_valid is True
        assert error_msg is None

    def test_single_valid_message(self):
        """Test validation of single valid message."""
        is_valid, error_msg = validate_messages(["Hello, World!"], 1, 30720)
        assert is_valid is True
        assert error_msg is None

    def test_multiple_valid_messages(self, sample_messages):
        """Test validation of multiple valid messages."""
        is_valid, error_msg = validate_messages(sample_messages["multiple"], 1, 30720)
        assert is_valid is True
        assert error_msg is None

    def test_message_with_special_characters(self):
        """Test validation of messages with special characters."""
        messages = [
            "Message with special chars: !@#$%^&*()",
            "Unicode: 你好世界 🌍",
        ]
        is_valid, error_msg = validate_messages(messages, 1, 30720)
        assert is_valid is True
        assert error_msg is None

    def test_message_with_newlines_and_tabs(self):
        """Test validation of messages with newlines and tabs."""
        messages = ["Message with\nnewlines", "Message with\ttabs"]
        is_valid, error_msg = validate_messages(messages, 1, 30720)
        assert is_valid is True
        assert error_msg is None

    def test_message_at_min_length(self, sample_messages):
        """Test validation of message at minimum length (1 character)."""
        messages = [sample_messages["edge_cases"]["min_length"]]
        is_valid, error_msg = validate_messages(messages, 1, 30720)
        assert is_valid is True
        assert error_msg is None

    def test_message_at_max_length(self, sample_messages):
        """Test validation of message at maximum length (30720 characters)."""
        messages = [sample_messages["edge_cases"]["max_length"]]
        is_valid, error_msg = validate_messages(messages, 1, 30720)
        assert is_valid is True
        assert error_msg is None

    def test_message_with_leading_trailing_spaces(self):
        """Test validation of message with valid content and spaces."""
        messages = ["   valid content   "]
        is_valid, error_msg = validate_messages(messages, 1, 30720)
        assert is_valid is True
        assert error_msg is None

    def test_non_string_message_invalid(self):
        """Test that non-string message is invalid."""
        messages = [123]  # Integer instead of string
        is_valid, error_msg = validate_messages(messages, 1, 30720)
        assert is_valid is False
        assert error_msg is not None
        assert "not a string" in error_msg
        assert "index 0" in error_msg

    def test_empty_string_invalid(self, sample_messages):
        """Test that empty string is invalid."""
        messages = [sample_messages["invalid"]["empty"]]
        is_valid, error_msg = validate_messages(messages, 1, 30720)
        assert is_valid is False
        assert error_msg is not None
        assert "too short" in error_msg

    def test_whitespace_only_message_invalid(self, sample_messages):
        """Test that whitespace-only message is invalid."""
        messages = [sample_messages["invalid"]["whitespace_only"]]
        is_valid, error_msg = validate_messages(messages, 1, 30720)
        assert is_valid is False
        assert error_msg is not None
        assert "only whitespace" in error_msg

    def test_message_too_long_invalid(self, sample_messages):
        """Test that message exceeding max length is invalid."""
        messages = [sample_messages["invalid"]["too_long"]]
        is_valid, error_msg = validate_messages(messages, 1, 30720)
        assert is_valid is False
        assert error_msg is not None
        assert "too long" in error_msg

    def test_mixed_valid_and_invalid_messages(self):
        """Test that validation fails if any message is invalid."""
        messages = ["Valid message", ""]  # Second message is invalid
        is_valid, error_msg = validate_messages(messages, 1, 30720)
        assert is_valid is False
        assert error_msg is not None
        assert "index 1" in error_msg

    @pytest.mark.parametrize(
        "invalid_message,expected_error",
        [
            (123, "not a string"),
            (None, "not a string"),
            ([], "not a string"),
            ({}, "not a string"),
        ],
    )
    def test_various_non_string_types(self, invalid_message, expected_error):
        """Test validation with various non-string types."""
        messages = [invalid_message]
        is_valid, error_msg = validate_messages(messages, 1, 30720)
        assert is_valid is False
        assert error_msg is not None
        assert expected_error in error_msg

    @pytest.mark.parametrize(
        "invalid_message,expected_error",
        [
            ("", "too short"),
            ("   ", "only whitespace"),
            ("\n", "only whitespace"),
            ("\t", "only whitespace"),
            ("   \n\t   ", "only whitespace"),
        ],
    )
    def test_various_empty_messages(self, invalid_message, expected_error):
        """Test validation with various empty/whitespace messages."""
        messages = [invalid_message]
        is_valid, error_msg = validate_messages(messages, 1, 30720)
        assert is_valid is False
        assert error_msg is not None
        assert expected_error in error_msg

    def test_error_message_includes_index(self):
        """Test that error messages include the index of invalid message."""
        messages = ["Valid", "Also valid", 123, "Another valid"]
        is_valid, error_msg = validate_messages(messages, 1, 30720)
        assert is_valid is False
        assert error_msg is not None
        assert "index 2" in error_msg

    def test_multiple_messages_first_invalid(self):
        """Test validation fails at first invalid message."""
        messages = ["", "Valid message"]
        is_valid, error_msg = validate_messages(messages, 1, 30720)
        assert is_valid is False
        assert error_msg is not None
        assert "index 0" in error_msg

    def test_multiple_messages_last_invalid(self):
        """Test validation fails at last invalid message."""
        messages = ["Valid message", ""]
        is_valid, error_msg = validate_messages(messages, 1, 30720)
        assert is_valid is False
        assert error_msg is not None
        assert "index 1" in error_msg

    def test_message_exactly_one_over_max_length(self):
        """Test message that is exactly one character over max length."""
        messages = ["x" * 30721]  # One over the limit
        is_valid, error_msg = validate_messages(messages, 1, 30720)
        assert is_valid is False
        assert error_msg is not None
        assert "too long" in error_msg
        assert "30720" in error_msg

    def test_large_list_of_valid_messages(self):
        """Test validation of large list of valid messages."""
        messages = [f"Message {i}" for i in range(100)]
        is_valid, error_msg = validate_messages(messages, 1, 30720)
        assert is_valid is True
        assert error_msg is None

    def test_large_list_with_one_invalid(self):
        """Test validation of large list with one invalid message."""
        messages = [f"Message {i}" for i in range(50)]
        messages.append("")  # Invalid at index 50
        messages.extend([f"Message {i}" for i in range(51, 100)])

        is_valid, error_msg = validate_messages(messages, 1, 30720)
        assert is_valid is False
        assert error_msg is not None
        assert "index 50" in error_msg
