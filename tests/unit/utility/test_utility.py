"""Unit tests for reflectlog/utility/utility.py module."""

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from reflectlog.utility.types import OAUTH_TOKEN_PREFIX


class TestGetClaudeCodeApiKey:
    """Tests for get_claude_code_api_key() function."""

    @patch("reflectlog.utility.utility.get_platform_retriever")
    def test_returns_credential_from_retriever(
        self, mock_get_retriever: MagicMock
    ) -> None:
        """Should return credential when retriever finds one."""
        mock_retriever = MagicMock()
        mock_retriever.get_credential.return_value = "sk-ant-test-key-123"
        mock_get_retriever.return_value = mock_retriever

        from reflectlog.utility.utility import get_claude_code_api_key

        result = get_claude_code_api_key()

        assert result == "sk-ant-test-key-123"
        mock_retriever.get_credential.assert_called_once()

    @patch("reflectlog.utility.utility.get_platform_retriever")
    def test_returns_none_when_no_retriever(
        self, mock_get_retriever: MagicMock
    ) -> None:
        """Should return None when get_platform_retriever returns None."""
        mock_get_retriever.return_value = None

        from reflectlog.utility.utility import get_claude_code_api_key

        result = get_claude_code_api_key()

        assert result is None

    @patch("reflectlog.utility.utility.get_platform_retriever")
    def test_returns_none_on_exception(self, mock_get_retriever: MagicMock) -> None:
        """Should return None silently on any exception."""
        mock_get_retriever.side_effect = RuntimeError("Platform error")

        from reflectlog.utility.utility import get_claude_code_api_key

        result = get_claude_code_api_key()

        assert result is None

    @patch("reflectlog.utility.utility.get_platform_retriever")
    def test_returns_none_when_retriever_raises(
        self, mock_get_retriever: MagicMock
    ) -> None:
        """Should return None when retriever.get_credential() raises."""
        mock_retriever = MagicMock()
        mock_retriever.get_credential.side_effect = OSError("Keychain error")
        mock_get_retriever.return_value = mock_retriever

        from reflectlog.utility.utility import get_claude_code_api_key

        result = get_claude_code_api_key()

        assert result is None

    @patch("reflectlog.utility.utility.get_platform_retriever")
    def test_verbose_mode_prints_to_stderr(
        self, mock_get_retriever: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Should print debug message to stderr when verbose=True."""
        mock_retriever = MagicMock()
        mock_retriever.get_credential.return_value = "sk-ant-test"
        mock_get_retriever.return_value = mock_retriever

        from reflectlog.utility.utility import get_claude_code_api_key

        _ = get_claude_code_api_key(verbose=True)

        captured = capsys.readouterr()
        assert "credential store" in captured.err

    @patch("reflectlog.utility.utility.get_platform_retriever")
    def test_verbose_false_no_stderr_output(
        self, mock_get_retriever: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Should not print to stderr when verbose=False."""
        mock_retriever = MagicMock()
        mock_retriever.get_credential.return_value = "sk-ant-test"
        mock_get_retriever.return_value = mock_retriever

        from reflectlog.utility.utility import get_claude_code_api_key

        _ = get_claude_code_api_key(verbose=False)

        captured = capsys.readouterr()
        assert captured.err == ""

    @patch("reflectlog.utility.utility.get_platform_retriever")
    def test_returns_none_when_credential_is_none(
        self, mock_get_retriever: MagicMock
    ) -> None:
        """Should return None when retriever returns None credential."""
        mock_retriever = MagicMock()
        mock_retriever.get_credential.return_value = None
        mock_get_retriever.return_value = mock_retriever

        from reflectlog.utility.utility import get_claude_code_api_key

        result = get_claude_code_api_key()

        assert result is None


class TestGetAnthropicApiKey:
    """Tests for get_anthropic_api_key() function."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-env-key"}, clear=False)
    @patch("reflectlog.utility.utility.get_claude_code_api_key")
    def test_returns_from_env_var(self, mock_get_key: MagicMock) -> None:
        """Should return API key from ANTHROPIC_API_KEY env var with source=env."""
        from reflectlog.utility.utility import get_anthropic_api_key

        result = get_anthropic_api_key()

        assert result is not None
        assert result["api_key"] == "sk-ant-env-key"
        assert result["source"] == "env"
        # Should not attempt keychain retrieval
        mock_get_key.assert_not_called()

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.utility.utility.get_claude_code_api_key")
    def test_falls_back_to_keychain(self, mock_get_key: MagicMock) -> None:
        """Should fall back to keychain when env var not set."""
        mock_get_key.return_value = "sk-ant-keychain-key"

        from reflectlog.utility.utility import get_anthropic_api_key

        result = get_anthropic_api_key()

        assert result is not None
        assert result["api_key"] == "sk-ant-keychain-key"
        assert result["source"] == "claude-code"
        mock_get_key.assert_called_once()

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.utility.utility.get_claude_code_api_key")
    def test_returns_none_when_neither_available(self, mock_get_key: MagicMock) -> None:
        """Should return None when no env var and no keychain credential."""
        mock_get_key.return_value = None

        from reflectlog.utility.utility import get_anthropic_api_key

        result = get_anthropic_api_key()

        assert result is None

    @patch.dict(
        os.environ,
        {"ANTHROPIC_API_KEY": "sk-ant-env-key"},
        clear=False,
    )
    @patch("reflectlog.utility.utility.get_claude_code_api_key")
    def test_env_var_takes_priority_over_keychain(
        self, mock_get_key: MagicMock
    ) -> None:
        """Env var should take priority even if keychain has a key."""
        mock_get_key.return_value = "sk-ant-keychain-key"

        from reflectlog.utility.utility import get_anthropic_api_key

        result = get_anthropic_api_key()

        assert result is not None
        assert result["api_key"] == "sk-ant-env-key"
        assert result["source"] == "env"
        mock_get_key.assert_not_called()


class TestInitCredentials:
    """Tests for init_credentials() function."""

    @patch.dict(
        os.environ,
        {"ANTHROPIC_AUTH_TOKEN": "sk-ant-oat01-existing-token-abc"},
        clear=False,
    )
    @patch("reflectlog.utility.utility.get_claude_code_api_key")
    def test_returns_existing_env_token(self, mock_get_key: MagicMock) -> None:
        """Should return existing ANTHROPIC_AUTH_TOKEN from env."""
        from reflectlog.utility.utility import init_credentials

        result = init_credentials(verbose=False)

        assert result == "sk-ant-oat01-existing-token-abc"
        mock_get_key.assert_not_called()

    @patch.dict(
        os.environ,
        {"ANTHROPIC_AUTH_TOKEN": "sk-ant-oat01-existing-token-abc"},
        clear=False,
    )
    @patch("reflectlog.utility.utility.get_claude_code_api_key")
    def test_verbose_prints_existing_token_prefix(
        self,
        mock_get_key: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Should print truncated token when verbose=True and token exists."""
        from reflectlog.utility.utility import init_credentials

        _ = init_credentials(verbose=True)

        captured = capsys.readouterr()
        assert "ANTHROPIC_AUTH_TOKEN:" in captured.out
        assert "sk-ant-oat01-existin" in captured.out

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.utility.utility.get_claude_code_api_key")
    def test_retrieves_oauth_from_keychain(self, mock_get_key: MagicMock) -> None:
        """Should retrieve OAuth token from keychain and set env var."""
        oauth_token = f"{OAUTH_TOKEN_PREFIX}keychain-oauth-token-xyz"
        mock_get_key.return_value = oauth_token

        from reflectlog.utility.utility import init_credentials

        result = init_credentials(verbose=False)

        assert result == oauth_token
        assert os.environ.get("ANTHROPIC_AUTH_TOKEN") == oauth_token

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.utility.utility.get_claude_code_api_key")
    def test_sets_env_var_when_oauth_found_in_keychain(
        self, mock_get_key: MagicMock
    ) -> None:
        """Should set ANTHROPIC_AUTH_TOKEN env var when OAuth token found."""
        oauth_token = f"{OAUTH_TOKEN_PREFIX}keychain-token-abc123"
        mock_get_key.return_value = oauth_token

        from reflectlog.utility.utility import init_credentials

        _ = init_credentials(verbose=False)

        assert os.environ["ANTHROPIC_AUTH_TOKEN"] == oauth_token

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.utility.utility.get_claude_code_api_key")
    def test_verbose_prints_keychain_oauth_token(
        self,
        mock_get_key: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Should print OAuth token info when verbose and found in keychain."""
        oauth_token = f"{OAUTH_TOKEN_PREFIX}keychain-oauth-token-xyz"
        mock_get_key.return_value = oauth_token

        from reflectlog.utility.utility import init_credentials

        _ = init_credentials(verbose=True)

        captured = capsys.readouterr()
        assert "ANTHROPIC_AUTH_TOKEN:" in captured.out
        assert "retrieved from keychain" in captured.out

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.utility.utility.get_claude_code_api_key")
    def test_handles_non_oauth_token(
        self,
        mock_get_key: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Should handle regular API key (not OAuth) from keychain."""
        mock_get_key.return_value = "sk-ant-api-regular-key-123"

        from reflectlog.utility.utility import init_credentials

        result = init_credentials(verbose=True)

        assert result is None
        captured = capsys.readouterr()
        assert "not an OAuth token" in captured.out
        assert "OAuth tokens are only available" in captured.out
        assert "SDK will use stored credentials" in captured.out

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.utility.utility.get_claude_code_api_key")
    def test_handles_non_oauth_token_verbose_false(
        self, mock_get_key: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Should not print when verbose=False and non-OAuth token found."""
        mock_get_key.return_value = "sk-ant-api-regular-key-123"

        from reflectlog.utility.utility import init_credentials

        result = init_credentials(verbose=False)

        assert result is None
        captured = capsys.readouterr()
        assert captured.out == ""

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.utility.utility.get_claude_code_api_key")
    def test_no_token_anywhere(
        self,
        mock_get_key: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Should return None and print status when no token found."""
        mock_get_key.return_value = None

        from reflectlog.utility.utility import init_credentials

        result = init_credentials(verbose=True)

        assert result is None
        captured = capsys.readouterr()
        assert "ANTHROPIC_AUTH_TOKEN: <not set>" in captured.out
        assert "SDK will use stored credentials" in captured.out

    @patch.dict(os.environ, {}, clear=True)
    @patch("reflectlog.utility.utility.get_claude_code_api_key")
    def test_no_token_verbose_false(
        self, mock_get_key: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Should not print when verbose=False and no token found."""
        mock_get_key.return_value = None

        from reflectlog.utility.utility import init_credentials

        result = init_credentials(verbose=False)

        assert result is None
        captured = capsys.readouterr()
        assert captured.out == ""


class TestGenerateContent:
    """Tests for generate_content() async function."""

    @patch("reflectlog.utility.utility.query")
    @patch("reflectlog.utility.utility.ClaudeAgentOptions")
    async def test_returns_text_from_assistant_message(
        self, mock_options_cls: MagicMock, mock_query: MagicMock
    ) -> None:
        """Should return concatenated text from TextBlock content."""
        from reflectlog.utility.utility import (
            AssistantMessage,
            TextBlock,
            generate_content,
        )

        # Create mock TextBlock
        mock_block = MagicMock(spec=TextBlock)
        mock_block.text = "Hello, world!"

        # Create mock AssistantMessage
        mock_message = MagicMock(spec=AssistantMessage)
        mock_message.content = [mock_block]

        # Make query return an async iterator
        async def mock_query_iter(**kwargs: Any) -> Any:
            yield mock_message

        mock_query.side_effect = mock_query_iter

        result = await generate_content("Say hello")

        assert result == "Hello, world!"

    @patch("reflectlog.utility.utility.query")
    @patch("reflectlog.utility.utility.ClaudeAgentOptions")
    async def test_concatenates_multiple_text_blocks(
        self, mock_options_cls: MagicMock, mock_query: MagicMock
    ) -> None:
        """Should concatenate text from multiple TextBlocks."""
        from reflectlog.utility.utility import (
            AssistantMessage,
            TextBlock,
            generate_content,
        )

        mock_block1 = MagicMock(spec=TextBlock)
        mock_block1.text = "Hello, "
        mock_block2 = MagicMock(spec=TextBlock)
        mock_block2.text = "world!"

        mock_message = MagicMock(spec=AssistantMessage)
        mock_message.content = [mock_block1, mock_block2]

        async def mock_query_iter(**kwargs: Any) -> Any:
            yield mock_message

        mock_query.side_effect = mock_query_iter

        result = await generate_content("Say hello")

        assert result == "Hello, world!"

    @patch("reflectlog.utility.utility.query")
    @patch("reflectlog.utility.utility.ClaudeAgentOptions")
    async def test_raises_runtime_error_on_error_message(
        self, mock_options_cls: MagicMock, mock_query: MagicMock
    ) -> None:
        """Should raise RuntimeError when ResultMessage is an error."""
        from reflectlog.utility.utility import ResultMessage, generate_content

        mock_error = MagicMock(spec=ResultMessage)
        mock_error.is_error = True
        mock_error.subtype = "api_error"

        async def mock_query_iter(**kwargs: Any) -> Any:
            yield mock_error

        mock_query.side_effect = mock_query_iter

        with pytest.raises(RuntimeError, match="Query failed: api_error"):
            _ = await generate_content("Will fail")

    @patch("reflectlog.utility.utility.query")
    @patch("reflectlog.utility.utility.ClaudeAgentOptions")
    async def test_ignores_non_error_result_message(
        self, mock_options_cls: MagicMock, mock_query: MagicMock
    ) -> None:
        """Should ignore ResultMessage that is not an error."""
        from reflectlog.utility.utility import (
            AssistantMessage,
            ResultMessage,
            TextBlock,
            generate_content,
        )

        mock_block = MagicMock(spec=TextBlock)
        mock_block.text = "Content"

        mock_assistant = MagicMock(spec=AssistantMessage)
        mock_assistant.content = [mock_block]

        mock_result = MagicMock(spec=ResultMessage)
        mock_result.is_error = False

        async def mock_query_iter(**kwargs: Any) -> Any:
            yield mock_assistant
            yield mock_result

        mock_query.side_effect = mock_query_iter

        result = await generate_content("Test prompt")

        assert result == "Content"

    @patch("reflectlog.utility.utility.query")
    @patch("reflectlog.utility.utility.ClaudeAgentOptions")
    async def test_ignores_non_text_blocks(
        self, mock_options_cls: MagicMock, mock_query: MagicMock
    ) -> None:
        """Should skip content blocks that are not TextBlock instances."""
        from reflectlog.utility.utility import (
            AssistantMessage,
            TextBlock,
            generate_content,
        )

        mock_text_block = MagicMock(spec=TextBlock)
        mock_text_block.text = "Valid text"

        # A non-TextBlock content item
        mock_other_block = MagicMock()

        mock_message = MagicMock(spec=AssistantMessage)
        mock_message.content = [mock_other_block, mock_text_block]

        async def mock_query_iter(**kwargs: Any) -> Any:
            yield mock_message

        mock_query.side_effect = mock_query_iter

        result = await generate_content("Test")

        assert result == "Valid text"

    @patch("reflectlog.utility.utility.query")
    @patch("reflectlog.utility.utility.ClaudeAgentOptions")
    async def test_strips_result_text(
        self, mock_options_cls: MagicMock, mock_query: MagicMock
    ) -> None:
        """Should strip leading/trailing whitespace from result."""
        from reflectlog.utility.utility import (
            AssistantMessage,
            TextBlock,
            generate_content,
        )

        mock_block = MagicMock(spec=TextBlock)
        mock_block.text = "  Padded content  "

        mock_message = MagicMock(spec=AssistantMessage)
        mock_message.content = [mock_block]

        async def mock_query_iter(**kwargs: Any) -> Any:
            yield mock_message

        mock_query.side_effect = mock_query_iter

        result = await generate_content("Test")

        assert result == "Padded content"

    @patch("reflectlog.utility.utility.query")
    @patch("reflectlog.utility.utility.ClaudeAgentOptions")
    async def test_empty_response_returns_empty_string(
        self, mock_options_cls: MagicMock, mock_query: MagicMock
    ) -> None:
        """Should return empty string when no messages received."""
        from reflectlog.utility.utility import generate_content

        async def mock_query_iter(**kwargs: Any) -> Any:
            return
            yield  # noqa: RET504  # Make it an async generator

        mock_query.side_effect = mock_query_iter

        result = await generate_content("Test")

        assert result == ""

    @patch("reflectlog.utility.utility.query")
    @patch("reflectlog.utility.utility.ClaudeAgentOptions")
    async def test_passes_options_correctly(
        self, mock_options_cls: MagicMock, mock_query: MagicMock
    ) -> None:
        """Should pass correct options to ClaudeAgentOptions."""
        from reflectlog.utility.utility import generate_content

        async def mock_query_iter(**kwargs: Any) -> Any:
            return
            yield  # noqa: RET504

        mock_query.side_effect = mock_query_iter

        _ = await generate_content(
            "Test prompt",
            model="claude-sonnet-4-5-20250929",
            system_prompt="Be helpful",
            allowed_tools=["read", "write"],
        )

        mock_options_cls.assert_called_once_with(
            model="claude-sonnet-4-5-20250929",
            system_prompt="Be helpful",
            allowed_tools=["read", "write"],
            permission_mode="bypassPermissions",
        )

    @patch("reflectlog.utility.utility.query")
    @patch("reflectlog.utility.utility.ClaudeAgentOptions")
    async def test_default_options_empty_tools(
        self, mock_options_cls: MagicMock, mock_query: MagicMock
    ) -> None:
        """Should default to empty allowed_tools list."""
        from reflectlog.utility.utility import generate_content

        async def mock_query_iter(**kwargs: Any) -> Any:
            return
            yield  # noqa: RET504

        mock_query.side_effect = mock_query_iter

        _ = await generate_content("Test")

        mock_options_cls.assert_called_once_with(
            model=None,
            system_prompt=None,
            allowed_tools=[],
            permission_mode="bypassPermissions",
        )

    @patch("reflectlog.utility.utility.query")
    @patch("reflectlog.utility.utility.ClaudeAgentOptions")
    async def test_multiple_assistant_messages(
        self, mock_options_cls: MagicMock, mock_query: MagicMock
    ) -> None:
        """Should concatenate text across multiple AssistantMessages."""
        from reflectlog.utility.utility import (
            AssistantMessage,
            TextBlock,
            generate_content,
        )

        mock_block1 = MagicMock(spec=TextBlock)
        mock_block1.text = "Part 1. "

        mock_msg1 = MagicMock(spec=AssistantMessage)
        mock_msg1.content = [mock_block1]

        mock_block2 = MagicMock(spec=TextBlock)
        mock_block2.text = "Part 2."

        mock_msg2 = MagicMock(spec=AssistantMessage)
        mock_msg2.content = [mock_block2]

        async def mock_query_iter(**kwargs: Any) -> Any:
            yield mock_msg1
            yield mock_msg2

        mock_query.side_effect = mock_query_iter

        result = await generate_content("Multi-part")

        assert result == "Part 1. Part 2."

    @patch("reflectlog.utility.utility.query")
    @patch("reflectlog.utility.utility.ClaudeAgentOptions")
    async def test_passes_prompt_to_query(
        self, mock_options_cls: MagicMock, mock_query: MagicMock
    ) -> None:
        """Should pass prompt to query function."""
        from reflectlog.utility.utility import generate_content

        mock_options_instance = MagicMock()
        mock_options_cls.return_value = mock_options_instance

        async def mock_query_iter(**kwargs: Any) -> Any:
            return
            yield  # noqa: RET504

        mock_query.side_effect = mock_query_iter

        _ = await generate_content("My test prompt")

        mock_query.assert_called_once_with(
            prompt="My test prompt",
            options=mock_options_instance,
        )
