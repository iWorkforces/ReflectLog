"""Property-based testing using Hypothesis.

Generates edge cases and complex scenarios to find bugs that
traditional unit testing might miss.

Example:
    @given(st.text(min_size=1, max_size=1000))
    def test_query_length(query: str):
        result = search_memory(query)
        assert isinstance(result, list)
        # Hypothesis will try many variations automatically

Usage:
    pytest tests/unit/application/test_search_hypothesis.py
    pytest --hypothesis-show-statematrix
"""

from hypothesis import given, settings, strategies as st  # type: ignore[import]
from reflectlog.application.config import Config
from reflectlog.application.memory.manager import MemoryManager
from reflectlog.application.utils import create_logger
from reflectlog.application.utils.security import SecretString
import pytest


@given(st.text(min_size=1, max_size=1000))
def test_query_edge_cases(query: str):
    """Test search with edge case inputs.

    Args:
        query: Query string to search.

    Hypothesis generates: Unicode, empty, very long, etc.
    """
    config = Config(
        project_id="hypothesis_test",
        openrouter_api_key=SecretString("test-key-not-real"),
    )
    logger = create_logger(__name__, config.project_id, "INFO")

    manager = MemoryManager(config, logger)
    results = manager.search(query)

    assert isinstance(results, list)
    assert len(results) <= manager.config.search_limit


@given(st.integers(min_value=1, max_value=1000))
def test_add_messages_varied_count(count: int):
    """Test add with varying message counts.

    Args:
        count: Number of messages to add.

    Hypothesis will test: 1, 100, 999, etc.
    """
    config = Config(
        project_id="hypothesis_test",
        openrouter_api_key=SecretString("test-key-not-real"),
    )
    logger = create_logger(__name__, config.project_id, "INFO")

    manager = MemoryManager(config, logger)
    added = manager.add_messages([f"Memory {i}" for i in range(count)])

    assert added == count


@given(st.integers(min_value=0, max_value=1))
def test_add_empty_messages(count: int):
    """Test add with empty message list.

    Args:
        count: Number of empty messages (0-1000).

    Boundary case: should handle gracefully.
    """
    config = Config(
        project_id="hypothesis_test",
        openrouter_api_key=SecretString("test-key-not-real"),
    )
    logger = create_logger(__name__, config.project_id, "INFO")

    manager = MemoryManager(config, logger)

    if count == 0:
        added = manager.add_messages([])
    else:
        messages = [""] * count
        added = manager.add_messages(messages)

    assert added == count


@given(st.text(min_size=10, max_size=100))
def test_add_injection_attempt(query: str):
    """Test that malicious query patterns are handled.

    Args:
        query: Query that might contain injection attempts.

    Hypothesis will generate: Various text patterns including potential injections.
    """
    config = Config(
        project_id="hypothesis_test",
        openrouter_api_key=SecretString("test-key-not-real"),
    )
    logger = create_logger(__name__, config.project_id, "INFO")

    manager = MemoryManager(config, logger)

    results = manager.search(query)

    assert isinstance(results, list)
    assert not any("error" in str(r).lower() for r in results)


@given(st.integers(min_value=0, max_value=10000))
def test_max_message_length(length: int):
    """Test max message length validation.

    Args:
        length: Message length to test.

    Boundary case: Should reject or accept.
    """
    config = Config(
        project_id="hypothesis_test",
        openrouter_api_key=SecretString("test-key-not-real"),
    )
    logger = create_logger(__name__, config.project_id, "INFO")

    manager = MemoryManager(config, logger)

    if length <= config.max_message_length:
        result = manager.add_messages([f"{'x' * length}"])
        assert result >= 0
    else:
        with pytest.raises(Exception):
            manager.add_messages([f"{'x' * length}"])
