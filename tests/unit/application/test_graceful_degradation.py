"""Graceful degradation path tests.

These tests verify that the system degrades gracefully when external
dependencies (LLM providers, APIs) fail or are unavailable.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from reflectlog.infrastructure.llm_reranker import OpenAIRerankerProvider
from reflectlog.infrastructure.smart_replacer import (
    OpenAIReplacementProvider,
)


@pytest.mark.unit
class TestLLMRerankerDegradation:
    """Tests for LLM reranker graceful degradation."""

    @pytest.mark.asyncio
    async def test_llm_fallback_to_fusion_score(self):
        """Test that reranker returns fusion score when LLM call fails."""
        provider = OpenAIRerankerProvider(
            api_key="test-key",
            base_url="https://api.test.com",
            model="test-model",
        )

        # Mock the LLM client to raise an exception
        provider._client = AsyncMock()
        provider._client.chat.completions.create = AsyncMock(
            side_effect=Exception("API unavailable")
        )

        # Mock logger
        provider._logger = Mock()

        # Call score_document - should gracefully fallback to fusion score
        fallback_score = 0.75
        result = await provider.score_document(
            query="test query",
            document="test document",
            fallback_score=fallback_score,
        )

        # Should return the fallback score
        assert result == ("test document", fallback_score)

        # Should have logged a warning
        provider._logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_partial_llm_failure(self):
        """Test that partial LLM failures are handled gracefully."""
        provider = OpenAIRerankerProvider(
            api_key="test-key",
            base_url="https://api.test.com",
            model="test-model",
        )

        # Mock the LLM client to fail on second call
        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("API temporarily unavailable")
            # Return valid response for other calls
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = '{"score": 0.85}'
            return mock_response

        provider._client = AsyncMock()
        provider._client.chat.completions.create = AsyncMock(side_effect=mock_create)

        # Mock logger
        provider._logger = Mock()

        # Score multiple documents
        documents = ["doc 1", "doc 2", "doc 3"]
        results = []

        for doc in documents:
            result = await provider.score_document(
                query="test query",
                document=doc,
                fallback_score=0.5,
            )
            results.append(result)

        # Should have 3 results (no crashes)
        assert len(results) == 3

        # Second result should be the fallback score
        assert results[1][1] == 0.5

        # First and third should have LLM scores
        assert results[0][1] == 0.85
        assert results[2][1] == 0.85

    @pytest.mark.asyncio
    async def test_invalid_json_response_fallback(self):
        """Test that invalid JSON responses are handled gracefully."""
        provider = OpenAIRerankerProvider(
            api_key="test-key",
            base_url="https://api.test.com",
            model="test-model",
        )

        # Mock the LLM client to return invalid JSON
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is not valid JSON"

        provider._client = AsyncMock()
        provider._client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Mock logger
        provider._logger = Mock()

        # Call score_document - should fallback to fusion score
        fallback_score = 0.6
        result = await provider.score_document(
            query="test query",
            document="test document",
            fallback_score=fallback_score,
        )

        # Should return the fallback score
        assert result == ("test document", fallback_score)

        # Should have logged a warning
        provider._logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_missing_score_field_fallback(self):
        """Test that missing score field in JSON is handled gracefully."""
        provider = OpenAIRerankerProvider(
            api_key="test-key",
            base_url="https://api.test.com",
            model="test-model",
        )

        # Mock the LLM client to return JSON without score field
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"reason": "This is relevant"}'

        provider._client = AsyncMock()
        provider._client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Mock logger
        provider._logger = Mock()

        # Call score_document - should fallback to fusion score
        fallback_score = 0.7
        result = await provider.score_document(
            query="test query",
            document="test document",
            fallback_score=fallback_score,
        )

        # Should return the fallback score
        assert result == ("test document", fallback_score)

    @pytest.mark.asyncio
    async def test_score_out_of_range_clamping(self):
        """Test that out-of-range scores are clamped to [0, 1]."""
        provider = OpenAIRerankerProvider(
            api_key="test-key",
            base_url="https://api.test.com",
            model="test-model",
        )

        # Test cases for out-of-range values
        test_cases = [
            ('{"score": 1.5}', 1.0),  # Above 1.0
            ('{"score": -0.5}', 0.0),  # Below 0.0
            ('{"score": 999.0}', 1.0),  # Way above 1.0
            ('{"score": -999.0}', 0.0),  # Way below 0.0
        ]

        for json_response, expected_clamped in test_cases:
            # Mock the LLM client
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = json_response

            provider._client = AsyncMock()
            provider._client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            # Mock logger
            provider._logger = Mock()

            # Call score_document
            result = await provider.score_document(
                query="test query",
                document="test document",
                fallback_score=0.5,
            )

            # Score should be clamped to [0, 1]
            assert result[1] == expected_clamped


@pytest.mark.unit
class TestSmartReplacerDegradation:
    """Tests for SmartReplacer graceful degradation."""

    @pytest.mark.asyncio
    async def test_replacer_adds_normally_on_llm_failure(self):
        """Test that memory is added normally when LLM replacement check fails."""
        provider = OpenAIReplacementProvider(
            api_key="test-key",
            base_url="https://api.test.com",
            model="test-model",
        )

        # Mock the LLM client to raise an exception
        provider._client = AsyncMock()
        provider._client.chat.completions.create = AsyncMock(
            side_effect=Exception("API unavailable")
        )

        # Mock logger
        provider._logger = Mock()

        # Call replacement detection - should gracefully fallback to "no replacement"
        prompt = (
            "Should 'I don't like cats anymore, I like dogs' replace 'I like cats'?"
        )
        result = await provider.detect_replacement(
            prompt=prompt,
            max_retries=3,
            retry_delay=1.0,
        )

        # Should return (should_replace=False, confidence=0.0, reason)
        should_replace, confidence, reason = result
        assert should_replace is False
        assert confidence == 0.0
        assert "failed" in reason.lower() or "error" in reason.lower()

        # Should have logged a warning
        provider._logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_replacer_with_retry_success_after_transient_failure(self):
        """Test that transient failures are retried successfully."""
        provider = OpenAIReplacementProvider(
            api_key="test-key",
            base_url="https://api.test.com",
            model="test-model",
        )

        # Mock the LLM client to fail twice, then succeed
        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Temporary API error")
            # Return valid response on third attempt
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[
                0
            ].message.content = '{"should_replace": true, "confidence": 0.9, "reason": "User changed preference"}'
            return mock_response

        provider._client = AsyncMock()
        provider._client.chat.completions.create = AsyncMock(side_effect=mock_create)

        # Mock logger
        provider._logger = Mock()

        # Call replacement detection - should retry and succeed
        prompt = (
            "Should 'I don't like cats anymore, I like dogs' replace 'I like cats'?"
        )
        result = await provider.detect_replacement(
            prompt=prompt,
            max_retries=3,
            retry_delay=1.0,
        )

        # Should succeed after retries
        should_replace, confidence, reason = result
        assert should_replace is True
        assert confidence == 0.9
        assert "changed preference" in reason.lower()

        # Should have made 3 attempts (2 failures + 1 success)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_replacer_invalid_json_fallback(self):
        """Test that invalid JSON responses are handled gracefully."""
        provider = OpenAIReplacementProvider(
            api_key="test-key",
            base_url="https://api.test.com",
            model="test-model",
        )

        # Mock the LLM client to return invalid JSON
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is not valid JSON"

        provider._client = AsyncMock()
        provider._client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Mock logger
        provider._logger = Mock()

        # Call replacement detection - should fallback to no replacement
        prompt = "Should 'new' replace 'old'?"
        result = await provider.detect_replacement(
            prompt=prompt,
            max_retries=3,
            retry_delay=1.0,
        )

        # Should return safe defaults
        should_replace, confidence, _reason = result
        assert should_replace is False
        assert confidence == 0.0

    @pytest.mark.asyncio
    async def test_replacer_missing_fields_fallback(self):
        """Test that missing fields in JSON are handled gracefully."""
        provider = OpenAIReplacementProvider(
            api_key="test-key",
            base_url="https://api.test.com",
            model="test-model",
        )

        # Test cases for missing fields with expected results
        test_cases = [
            (
                '{"should_replace": true}',
                True,
                0.0,
            ),  # Missing confidence and reason - should_replace is True
            (
                '{"confidence": 0.9}',
                False,
                0.9,
            ),  # Missing should_replace - defaults to False
            (
                '{"reason": "User changed mind"}',
                False,
                0.0,
            ),  # Missing should_replace and confidence
            ("{}", False, 0.0),  # Empty JSON - all defaults
        ]

        for json_response, expected_should_replace, expected_confidence in test_cases:
            # Mock the LLM client
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = json_response

            provider._client = AsyncMock()
            provider._client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            # Mock logger
            provider._logger = Mock()

            # Call replacement detection
            prompt = "Should 'new' replace 'old'?"
            result = await provider.detect_replacement(
                prompt=prompt,
                max_retries=3,
                retry_delay=1.0,
            )

            # Should return expected values
            should_replace, confidence, _reason = result
            assert should_replace == expected_should_replace, (
                f"For JSON '{json_response}': expected {expected_should_replace}, got {should_replace}"
            )
            assert confidence == expected_confidence, (
                f"For JSON '{json_response}': expected confidence {expected_confidence}, got {confidence}"
            )

    @pytest.mark.asyncio
    async def test_replacer_confidence_clamping(self):
        """Test that confidence values are clamped to [0, 1]."""
        provider = OpenAIReplacementProvider(
            api_key="test-key",
            base_url="https://api.test.com",
            model="test-model",
        )

        # Test cases for out-of-range confidence values
        test_cases = [
            ('{"should_replace": true, "confidence": 1.5, "reason": "test"}', 1.0),
            ('{"should_replace": true, "confidence": -0.5, "reason": "test"}', 0.0),
            ('{"should_replace": true, "confidence": 999.0, "reason": "test"}', 1.0),
        ]

        for json_response, expected_clamped in test_cases:
            # Mock the LLM client
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = json_response

            provider._client = AsyncMock()
            provider._client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            # Mock logger
            provider._logger = Mock()

            # Call replacement detection
            prompt = "Should 'new' replace 'old'?"
            result = await provider.detect_replacement(
                prompt=prompt,
                max_retries=3,
                retry_delay=1.0,
            )

            # Confidence should be clamped
            assert result[1] == expected_clamped

    @pytest.mark.asyncio
    async def test_replacer_timeout_handling(self):
        """Test that timeouts are handled gracefully."""
        provider = OpenAIReplacementProvider(
            api_key="test-key",
            base_url="https://api.test.com",
            model="test-model",
            timeout=0.001,  # Very short timeout
        )

        # Mock the LLM client to take too long
        async def mock_create(*args, **kwargs):
            await asyncio.sleep(1)  # Sleep longer than timeout
            mock_response = Mock()
            mock_response.choices = [Mock()]
            return mock_response

        provider._client = AsyncMock()
        provider._client.chat.completions.create = AsyncMock(side_effect=mock_create)

        # Mock logger
        provider._logger = Mock()

        # Call replacement detection - should timeout and fallback
        prompt = "Should 'new' replace 'old'?"
        result = await provider.detect_replacement(
            prompt=prompt,
            max_retries=1,  # Only retry once to speed up test
            retry_delay=0.1,
        )

        # Should return safe defaults (likely due to timeout)
        should_replace, _confidence, _reason = result
        # Should default to False on error
        assert should_replace is False
