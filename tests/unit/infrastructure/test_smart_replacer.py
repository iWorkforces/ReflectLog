"""Unit tests for SmartReplacer."""

import json
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from openmemories.infrastructure.smart_replacer import (
    ReplacementDecision,
    SmartReplacer,
    SmartReplacerConfig,
)


class TestReplacementDecision:
    """Test ReplacementDecision Pydantic schema."""

    def test_valid_decision_should_replace(self) -> None:
        """Test valid replacement decision."""
        decision = ReplacementDecision(
            should_replace=True,
            confidence=0.85,
            reason="Same topic with updated preference",
        )
        assert decision.should_replace is True
        assert decision.confidence == 0.85
        assert decision.reason == "Same topic with updated preference"

    def test_valid_decision_no_replace(self) -> None:
        """Test valid no-replacement decision."""
        decision = ReplacementDecision(
            should_replace=False,
            confidence=0.3,
            reason="Different topics",
        )
        assert decision.should_replace is False
        assert decision.confidence == 0.3

    def test_valid_confidence_zero(self) -> None:
        """Test minimum valid confidence."""
        decision = ReplacementDecision(
            should_replace=False,
            confidence=0.0,
            reason="No match",
        )
        assert decision.confidence == 0.0

    def test_valid_confidence_one(self) -> None:
        """Test maximum valid confidence."""
        decision = ReplacementDecision(
            should_replace=True,
            confidence=1.0,
            reason="Exact replacement",
        )
        assert decision.confidence == 1.0

    def test_invalid_confidence_below_zero(self) -> None:
        """Test that confidence below 0.0 is rejected."""
        with pytest.raises(ValueError):
            ReplacementDecision(
                should_replace=False,
                confidence=-0.1,
                reason="Invalid",
            )

    def test_invalid_confidence_above_one(self) -> None:
        """Test that confidence above 1.0 is rejected."""
        with pytest.raises(ValueError):
            ReplacementDecision(
                should_replace=True,
                confidence=1.1,
                reason="Invalid",
            )

    def test_json_schema_generation(self) -> None:
        """Test that JSON schema is generated correctly."""
        schema = ReplacementDecision.model_json_schema()
        assert "should_replace" in schema.get("properties", {})
        assert "confidence" in schema.get("properties", {})
        assert "reason" in schema.get("properties", {})

    def test_json_serialization(self) -> None:
        """Test JSON serialization round-trip."""
        decision = ReplacementDecision(
            should_replace=True,
            confidence=0.75,
            reason="Updated preference",
        )
        json_str = decision.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["should_replace"] is True
        assert parsed["confidence"] == 0.75
        assert parsed["reason"] == "Updated preference"


class TestSmartReplacerConfig:
    """Test SmartReplacerConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SmartReplacerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="x-ai/grok-4.1-fast",
        )

        assert config.threshold == 0.7
        assert config.enabled is True
        assert config.timeout == 30.0

    def test_custom_values(self) -> None:
        """Test configuration with custom values."""
        config = SmartReplacerConfig(
            api_key="test-key",
            base_url="https://custom.url/api/v1",
            model="custom/model",
            threshold=0.8,
            enabled=False,
            timeout=60.0,
        )

        assert config.api_key == "test-key"
        assert config.base_url == "https://custom.url/api/v1"
        assert config.model == "custom/model"
        assert config.threshold == 0.8
        assert config.enabled is False
        assert config.timeout == 60.0

    def test_from_app_config(self) -> None:
        """Test factory method from application config."""
        mock_app_config = MagicMock()
        mock_app_config.openrouter_api_key.get_secret_value.return_value = "api-key"
        mock_app_config.openrouter_base_url = "https://openrouter.ai/api/v1"
        mock_app_config.llm_model = "x-ai/grok-4.1-fast"
        mock_app_config.smart_replace_threshold = 0.8
        mock_app_config.enable_smart_replace = True

        config = SmartReplacerConfig.from_app_config(mock_app_config)

        assert config.api_key == "api-key"
        assert config.base_url == "https://openrouter.ai/api/v1"
        assert config.model == "x-ai/grok-4.1-fast"
        assert config.threshold == 0.8
        assert config.enabled is True


class TestSmartReplacerInitialization:
    """Test SmartReplacer initialization."""

    def test_initialization_creates_async_client_when_enabled(self) -> None:
        """Test that initialization creates AsyncOpenAI client when enabled."""
        config = SmartReplacerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="x-ai/grok-4.1-fast",
            enabled=True,
        )

        with patch(
            "openmemories.infrastructure.smart_replacer.AsyncOpenAI"
        ) as mock_async_client:
            replacer = SmartReplacer(config=config)

            mock_async_client.assert_called_once_with(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                http_client=ANY,
                timeout=30.0,
            )
            assert replacer._client is not None

    def test_initialization_no_client_when_disabled(self) -> None:
        """Test that no client is created when disabled."""
        config = SmartReplacerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="x-ai/grok-4.1-fast",
            enabled=False,
        )

        with patch(
            "openmemories.infrastructure.smart_replacer.AsyncOpenAI"
        ) as mock_async_client:
            replacer = SmartReplacer(config=config)

            mock_async_client.assert_not_called()
            assert replacer._client is None


class TestCheckReplacement:
    """Test check_replacement method."""

    @pytest.fixture
    def mock_replacer(self) -> SmartReplacer:
        """Create a mocked SmartReplacer instance."""
        config = SmartReplacerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="x-ai/grok-4.1-fast",
            threshold=0.7,
        )

        with patch("openmemories.infrastructure.smart_replacer.AsyncOpenAI"):
            replacer = SmartReplacer(config=config)
            return replacer

    @pytest.mark.asyncio
    async def test_check_replacement_should_replace(
        self, mock_replacer: SmartReplacer
    ) -> None:
        """Test successful replacement detection."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "should_replace": True,
                "confidence": 0.9,
                "reason": "Same topic with updated preference",
            }
        )

        mock_replacer._client = AsyncMock()
        mock_replacer._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        should_replace, confidence, reason = await mock_replacer.check_replacement(
            new_memory="I don't like cats anymore",
            existing_memory="I like cats",
        )

        assert should_replace is True
        assert confidence == 0.9
        assert reason == "Same topic with updated preference"

    @pytest.mark.asyncio
    async def test_check_replacement_no_replace(
        self, mock_replacer: SmartReplacer
    ) -> None:
        """Test no replacement needed."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "should_replace": False,
                "confidence": 0.2,
                "reason": "Different topics entirely",
            }
        )

        mock_replacer._client = AsyncMock()
        mock_replacer._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        should_replace, confidence, reason = await mock_replacer.check_replacement(
            new_memory="I like dogs",
            existing_memory="The weather is nice today",
        )

        assert should_replace is False
        assert confidence == 0.2
        assert reason == "Different topics entirely"

    @pytest.mark.asyncio
    async def test_check_replacement_below_threshold(
        self, mock_replacer: SmartReplacer
    ) -> None:
        """Test that replacement is not triggered when confidence is below threshold."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "should_replace": True,
                "confidence": 0.5,  # Below threshold of 0.7
                "reason": "Possibly related",
            }
        )

        mock_replacer._client = AsyncMock()
        mock_replacer._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        should_replace, confidence, reason = await mock_replacer.check_replacement(
            new_memory="I like dogs",
            existing_memory="I like cats",
        )

        # should_replace should be False because confidence is below threshold
        assert should_replace is False
        assert confidence == 0.5

    @pytest.mark.asyncio
    async def test_check_replacement_clamps_confidence(
        self, mock_replacer: SmartReplacer
    ) -> None:
        """Test that out-of-range confidence is clamped to 0-1."""
        # Test confidence > 1.0
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "should_replace": True,
                "confidence": 1.5,
                "reason": "Clamped",
            }
        )

        mock_replacer._client = AsyncMock()
        mock_replacer._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        _, confidence, _ = await mock_replacer.check_replacement(
            new_memory="test", existing_memory="test2"
        )
        assert confidence == 1.0

        # Test confidence < 0.0
        mock_response.choices[0].message.content = json.dumps(
            {
                "should_replace": False,
                "confidence": -0.5,
                "reason": "Clamped",
            }
        )

        _, confidence, _ = await mock_replacer.check_replacement(
            new_memory="test", existing_memory="test2"
        )
        assert confidence == 0.0

    @pytest.mark.asyncio
    async def test_check_replacement_disabled(self) -> None:
        """Test that disabled replacer returns no replacement."""
        config = SmartReplacerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="x-ai/grok-4.1-fast",
            enabled=False,
        )

        with patch("openmemories.infrastructure.smart_replacer.AsyncOpenAI"):
            replacer = SmartReplacer(config=config)

        should_replace, confidence, reason = await replacer.check_replacement(
            new_memory="test",
            existing_memory="test2",
        )

        assert should_replace is False
        assert confidence == 0.0
        assert reason == "Smart replacement disabled"

    @pytest.mark.asyncio
    async def test_check_replacement_client_not_initialized(
        self, mock_replacer: SmartReplacer
    ) -> None:
        """Test handling when client is not initialized."""
        mock_replacer._client = None

        should_replace, confidence, reason = await mock_replacer.check_replacement(
            new_memory="test",
            existing_memory="test2",
        )

        assert should_replace is False
        assert confidence == 0.0
        assert reason == "Client not initialized"

    @pytest.mark.asyncio
    async def test_check_replacement_invalid_json(
        self, mock_replacer: SmartReplacer
    ) -> None:
        """Test handling of invalid JSON response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "not valid json"

        mock_replacer._client = AsyncMock()
        mock_replacer._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_logger = MagicMock()
        mock_replacer.logger = mock_logger

        should_replace, confidence, reason = await mock_replacer.check_replacement(
            new_memory="test", existing_memory="test2"
        )

        assert should_replace is False
        assert confidence == 0.0
        assert "JSON parse error" in reason
        mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_replacement_empty_response(
        self, mock_replacer: SmartReplacer
    ) -> None:
        """Test handling of empty response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None

        mock_replacer._client = AsyncMock()
        mock_replacer._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_logger = MagicMock()
        mock_replacer.logger = mock_logger

        should_replace, confidence, reason = await mock_replacer.check_replacement(
            new_memory="test", existing_memory="test2"
        )

        assert should_replace is False
        assert confidence == 0.0
        assert "Error" in reason

    @pytest.mark.asyncio
    async def test_check_replacement_api_error(
        self, mock_replacer: SmartReplacer
    ) -> None:
        """Test handling of API error with retry behavior."""
        mock_replacer._client = AsyncMock()
        mock_replacer._client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        mock_logger = MagicMock()
        mock_replacer.logger = mock_logger

        should_replace, confidence, reason = await mock_replacer.check_replacement(
            new_memory="test", existing_memory="test2"
        )

        assert should_replace is False
        assert confidence == 0.0
        assert "Error" in reason
        # With retry logic, warning is called once per retry attempt + final error
        # Default: 3 retries + 1 final = 4 warning calls
        assert mock_logger.warning.call_count >= 1  # At least one warning logged


class TestStructuredOutputFallback:
    """Test structured output with fallback to json_object mode."""

    @pytest.fixture
    def mock_replacer(self) -> SmartReplacer:
        """Create a mocked SmartReplacer instance."""
        config = SmartReplacerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="x-ai/grok-4.1-fast",
        )

        with patch("openmemories.infrastructure.smart_replacer.AsyncOpenAI"):
            replacer = SmartReplacer(config=config)
            return replacer

    @pytest.mark.asyncio
    async def test_structured_output_success(
        self, mock_replacer: SmartReplacer
    ) -> None:
        """Test that structured output call succeeds."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "should_replace": True,
                "confidence": 0.85,
                "reason": "Same topic",
            }
        )

        mock_replacer._client = AsyncMock()
        mock_replacer._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        await mock_replacer._call_llm_with_structured_output("test prompt")

        # Verify structured output format was used
        mock_replacer._client.chat.completions.create.assert_called_once()
        call_kwargs = mock_replacer._client.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"]["type"] == "json_schema"
        assert "json_schema" in call_kwargs["response_format"]
        assert call_kwargs["response_format"]["json_schema"]["strict"] is True

    @pytest.mark.asyncio
    async def test_fallback_on_json_schema_not_supported(
        self, mock_replacer: SmartReplacer
    ) -> None:
        """Test fallback to json_object mode when structured outputs not supported."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "should_replace": False,
                "confidence": 0.3,
                "reason": "Different topics",
            }
        )

        # First call fails with json_schema error, second succeeds
        mock_replacer._client = AsyncMock()
        mock_replacer._client.chat.completions.create = AsyncMock(
            side_effect=[
                Exception("json_schema is not supported for this model"),
                mock_response,
            ]
        )

        mock_logger = MagicMock()
        mock_replacer.logger = mock_logger

        await mock_replacer._call_llm_with_structured_output("test prompt")

        # Verify fallback was called
        assert mock_replacer._client.chat.completions.create.call_count == 2

        # Verify second call used json_object mode
        second_call_kwargs = (
            mock_replacer._client.chat.completions.create.call_args_list[1][1]
        )
        assert second_call_kwargs["response_format"]["type"] == "json_object"

        # Verify warning was logged
        mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_fallback_on_unrelated_error(
        self, mock_replacer: SmartReplacer
    ) -> None:
        """Test that unrelated errors are not caught for fallback."""
        mock_replacer._client = AsyncMock()
        mock_replacer._client.chat.completions.create = AsyncMock(
            side_effect=Exception("Network timeout error")
        )

        with pytest.raises(Exception, match="Network timeout"):
            await mock_replacer._call_llm_with_structured_output("test prompt")

    @pytest.mark.asyncio
    async def test_client_not_initialized_raises_error(
        self, mock_replacer: SmartReplacer
    ) -> None:
        """Test that RuntimeError is raised when client is not initialized."""
        mock_replacer._client = None

        with pytest.raises(RuntimeError, match="not initialized"):
            await mock_replacer._call_llm_with_structured_output("test prompt")


class TestSmartReplacerIntegration:
    """Integration-style tests for full replacement flow."""

    @pytest.mark.asyncio
    async def test_full_replacement_flow_replace(self) -> None:
        """Test complete replacement flow with replacement triggered."""
        config = SmartReplacerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="x-ai/grok-4.1-fast",
            threshold=0.7,
        )

        with patch("openmemories.infrastructure.smart_replacer.AsyncOpenAI"):
            replacer = SmartReplacer(config=config, logger=MagicMock())

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = json.dumps(
                {
                    "should_replace": True,
                    "confidence": 0.9,
                    "reason": "Updated preference about cats",
                }
            )

            replacer._client = AsyncMock()
            replacer._client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            should_replace, confidence, reason = await replacer.check_replacement(
                new_memory="I don't like cats anymore, I prefer dogs",
                existing_memory="I like cats",
            )

            assert should_replace is True
            assert confidence == 0.9
            assert "Updated preference" in reason

    @pytest.mark.asyncio
    async def test_full_replacement_flow_no_replace(self) -> None:
        """Test complete replacement flow with no replacement."""
        config = SmartReplacerConfig(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="x-ai/grok-4.1-fast",
            threshold=0.7,
        )

        with patch("openmemories.infrastructure.smart_replacer.AsyncOpenAI"):
            replacer = SmartReplacer(config=config, logger=MagicMock())

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = json.dumps(
                {
                    "should_replace": False,
                    "confidence": 0.2,
                    "reason": "Different topics - cats vs weather",
                }
            )

            replacer._client = AsyncMock()
            replacer._client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            should_replace, confidence, reason = await replacer.check_replacement(
                new_memory="The weather is nice today",
                existing_memory="I like cats",
            )

            assert should_replace is False
            assert confidence == 0.2
