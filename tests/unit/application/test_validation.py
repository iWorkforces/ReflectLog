'''Tests for memory validation logic.'''

from typing import Any

import pytest

from reflectlog.application.utils.validation import validate_memories


@pytest.mark.unit
class TestValidateMemories:
    '''Test suite for validate_memories() function.'''

    def test_empty_list_is_valid(self):
        '''Test that empty list is valid (no-op case).'''
        is_valid, error_msg = validate_memories([], 1, 30720)
        assert is_valid is True
        assert error_msg is None

    def test_single_valid_memory(self):
        '''Test validation of single valid memory.'''
        is_valid, error_msg = validate_memories(["Hello, World!"], 1, 30720)
        assert is_valid is True
        assert error_msg is None

    def test_multiple_valid_memories(self, sample_memories: dict[str, Any]):
        '''Test validation of multiple valid memories.'''
        is_valid, error_msg = validate_memories(sample_memories["multiple"], 1, 30720)
        assert is_valid is True
        assert error_msg is None

    def test_memory_with_special_characters(self):
        '''Test validation of memories with special characters.'''
        memories = [
            "Memory with special chars: !@#$%^&*()",
            "Unicode: 你好世界 🌍",
        ]
        is_valid, error_msg = validate_memories(memories, 1, 30720)
        assert is_valid is True
        assert error_msg is None

    def test_memory_with_newlines_and_tabs(self):
        '''Test validation of memories with newlines and tabs.'''
        memories = ["Memory with\nnewlines", "Memory with\ttabs"]
        is_valid, error_msg = validate_memories(memories, 1, 30720)
        assert is_valid is True
        assert error_msg is None

    def test_memory_at_min_length(self, sample_memories: dict[str, Any]):
        '''Test validation of memory at minimum length (1 character).'''
        memories = [sample_memories["edge_cases"]["min_length"]]
        is_valid, error_msg = validate_memories(memories, 1, 30720)
        assert is_valid is True
        assert error_msg is None

    def test_memory_at_max_length(self, sample_memories: dict[str, Any]):
        '''Test validation of memory at maximum length (30720 characters).'''
        memories = [sample_memories["edge_cases"]["max_length"]]
        is_valid, error_msg = validate_memories(memories, 1, 30720)
        assert is_valid is True
        assert error_msg is None

    def test_memory_with_leading_trailing_spaces(self):
        '''Test validation of memory with valid content and spaces.'''
        memories = ["   valid content   "]
        is_valid, error_msg = validate_memories(memories, 1, 30720)
        assert is_valid is True
        assert error_msg is None

    def test_non_string_memory_invalid(self):
        '''Test that non-string memory is invalid.'''
        memories = [123]  # Integer instead of string
        is_valid, error_msg = validate_memories(memories, 1, 30720)
        assert is_valid is False
        assert error_msg is not None
        assert "not a string" in error_msg
        assert "index 0" in error_msg

    def test_empty_string_invalid(self, sample_memories: dict[str, Any]):
        '''Test that empty string is invalid.'''
        memories = [sample_memories["invalid"]["empty"]]
        is_valid, error_msg = validate_memories(memories, 1, 30720)
        assert is_valid is False
        assert error_msg is not None
        assert "contains only whitespace" in error_msg

    def test_whitespace_only_memory_invalid(self, sample_memories: dict[str, Any]):
        '''Test that whitespace-only memory is invalid.'''
        memories = [sample_memories["invalid"]["whitespace_only"]]
        is_valid, error_msg = validate_memories(memories, 1, 30720)
        assert is_valid is False
        assert error_msg is not None
        assert "only whitespace" in error_msg

    def test_memory_too_long_invalid(self, sample_memories: dict[str, Any]):
        '''Test that memory exceeding max length is invalid.'''
        memories = [sample_memories["invalid"]["too_long"]]
        is_valid, error_msg = validate_memories(memories, 1, 30720)
        assert is_valid is False
        assert error_msg is not None
        assert "too long" in error_msg

    def test_mixed_valid_and_invalid_memories(self):
        '''Test that validation fails if any memory is invalid.'''
        memories = ["Valid memory", ""]  # Second memory is invalid
        is_valid, error_msg = validate_memories(memories, 1, 30720)
        assert is_valid is False
        assert error_msg is not None
        assert "index 1" in error_msg

    @pytest.mark.parametrize(
        "invalid_memory,expected_error",
        [
            (123, "not a string"),
            (None, "is None"),
            ([], "not a string"),
            ({}, "not a string"),
        ],
    )
    def test_various_non_string_types(
        self, invalid_memory: object, expected_error: str
    ):
        '''Test validation with various non-string types.'''
        memories = [invalid_memory]
        is_valid, error_msg = validate_memories(memories, 1, 30720)
        assert is_valid is False
        assert error_msg is not None
        assert expected_error in error_msg

    @pytest.mark.parametrize(
        "invalid_memory,expected_error",
        [
            ("", "contains only whitespace"),
            ("   ", "only whitespace"),
            ("\n", "only whitespace"),
            ("\t", "only whitespace"),
            ("   \n\t   ", "only whitespace"),
        ],
    )
    def test_various_empty_memories(self, invalid_memory: str, expected_error: str):
        '''Test validation with various empty/whitespace memories.'''
        memories = [invalid_memory]
        is_valid, error_msg = validate_memories(memories, 1, 30720)
        assert is_valid is False
        assert error_msg is not None
        assert expected_error in error_msg

    def test_error_memory_includes_index(self):
        '''Test that error memories include the index of invalid memory.'''
        memories = ["Valid", "Also valid", 123, "Another valid"]
        is_valid, error_msg = validate_memories(memories, 1, 30720)
        assert is_valid is False
        assert error_msg is not None
        assert "index 2" in error_msg

    def test_multiple_memories_first_invalid(self):
        '''Test validation fails at first invalid memory.'''
        memories = ["", "Valid memory"]
        is_valid, error_msg = validate_memories(memories, 1, 30720)
        assert is_valid is False
        assert error_msg is not None
        assert "index 0" in error_msg

    def test_multiple_memories_last_invalid(self):
        '''Test validation fails at last invalid memory.'''
        memories = ["Valid memory", ""]
        is_valid, error_msg = validate_memories(memories, 1, 30720)
        assert is_valid is False
        assert error_msg is not None
        assert "index 1" in error_msg

    def test_memory_exactly_one_over_max_length(self):
        '''Test memory that is exactly one character over max length.'''
        memories = ["x" * 30721]  # One over the limit
        is_valid, error_msg = validate_memories(memories, 1, 30720)
        assert is_valid is False
        assert error_msg is not None
        assert "too long" in error_msg
        assert "30720" in error_msg

    def test_large_list_of_valid_memories(self):
        '''Test validation of large list of valid memories.'''
        memories = [f"Memory {i}" for i in range(100)]
        is_valid, error_msg = validate_memories(memories, 1, 30720)
        assert is_valid is True
        assert error_msg is None

    def test_large_list_with_one_invalid(self):
        '''Test validation of large list with one invalid memory.'''
        memories = [f"Memory {i}" for i in range(50)]
        memories.append("")  # Invalid at index 50
        memories.extend([f"Memory {i}" for i in range(51, 100)])

        is_valid, error_msg = validate_memories(memories, 1, 30720)
        assert is_valid is False
        assert error_msg is not None
        assert "index 50" in error_msg


@pytest.mark.unit
class TestValidateAddBatch:
    def test_rejects_too_many_items(self) -> None:
        from reflectlog.application.utils.validation import validate_add_batch

        ok, error = validate_add_batch(["x"] * 101, 100, 500_000)
        assert ok is False
        assert error is not None
        assert "Too many memories" in error

    def test_rejects_too_many_chars(self) -> None:
        from reflectlog.application.utils.validation import validate_add_batch

        ok, error = validate_add_batch(["a" * 500_001], 100, 500_000)
        assert ok is False
        assert error is not None
        assert "too large" in error

    def test_accepts_at_limits(self) -> None:
        from reflectlog.application.utils.validation import validate_add_batch

        ok, error = validate_add_batch(["a"] * 100, 100, 500_000)
        assert ok is True
        assert error is None
