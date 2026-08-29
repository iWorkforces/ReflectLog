"""Unit tests for live prompt interpolation."""

import pytest

from reflectlog.core.prompts import format_replacement_detection_prompt


@pytest.mark.unit
class TestPromptFormatters:
    """Template substitution must insert live text and keep single-brace JSON."""

    def test_replacement_prompt_includes_both_memories(self) -> None:
        prompt = format_replacement_detection_prompt(
            "old preference {cats}", "new preference {dogs}"
        )
        assert "old preference {cats}" in prompt
        assert "new preference {dogs}" in prompt
        assert '{"should_replace": true' in prompt
        assert '{{"should_replace"' not in prompt
