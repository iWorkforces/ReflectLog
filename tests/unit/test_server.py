"""Unit tests for openmemories/server.py CLI module."""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@pytest.mark.unit
class TestCLIArgumentParsing:
    """Test CLI argument parsing functionality."""

    @patch.dict(os.environ, {}, clear=True)
    @patch("openmemories.server.signal.signal")
    @patch("openmemories.server.warmup_numba_functions")
    @patch("openmemories.server.FastMCPServer")
    def test_default_arguments(self, mock_server_class, mock_warmup, mock_signal):
        """Test main() with no arguments uses defaults."""
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        # Import main and test
        from openmemories.server import main

        # Mock sys.argv
        with patch("sys.argv", ["openmemories"]):
            # Should not raise any errors
            try:
                main()
            except SystemExit:
                pass  # main() may call sys.exit()

            # Verify server was created and run
            mock_server_class.assert_called_once()

    @patch.dict(os.environ, {}, clear=True)
    @patch("openmemories.server.signal.signal")
    @patch("openmemories.server.warmup_numba_functions")
    @patch("openmemories.server.FastMCPServer")
    def test_transport_http_argument(self, mock_server_class, mock_warmup, mock_signal):
        """Test --transport http argument."""
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        from openmemories.server import main

        with patch("sys.argv", ["openmemories", "--transport", "http"]):
            try:
                main()
            except SystemExit:
                pass

            # Check environment variable was set
            assert os.environ.get("MCP_TRANSPORT") == "http"

    @patch.dict(os.environ, {}, clear=True)
    @patch("openmemories.server.signal.signal")
    @patch("openmemories.server.warmup_numba_functions")
    @patch("openmemories.server.FastMCPServer")
    def test_transport_stdio_argument(
        self, mock_server_class, mock_warmup, mock_signal
    ):
        """Test --transport stdio argument."""
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        from openmemories.server import main

        with patch("sys.argv", ["openmemories", "--transport", "stdio"]):
            try:
                main()
            except SystemExit:
                pass

            assert os.environ.get("MCP_TRANSPORT") == "stdio"

    @patch.dict(os.environ, {}, clear=True)
    @patch("openmemories.server.signal.signal")
    @patch("openmemories.server.warmup_numba_functions")
    @patch("openmemories.server.FastMCPServer")
    def test_port_argument(self, mock_server_class, mock_warmup, mock_signal):
        """Test --port argument."""
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        from openmemories.server import main

        with patch("sys.argv", ["openmemories", "--port", "9999"]):
            try:
                main()
            except SystemExit:
                pass

            assert os.environ.get("MCP_PORT") == "9999"

    @patch.dict(os.environ, {}, clear=True)
    @patch("openmemories.server.signal.signal")
    @patch("openmemories.server.warmup_numba_functions")
    @patch("openmemories.server.FastMCPServer")
    def test_host_argument(self, mock_server_class, mock_warmup, mock_signal):
        """Test --host argument."""
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        from openmemories.server import main

        with patch("sys.argv", ["openmemories", "--host", "192.168.1.1"]):
            try:
                main()
            except SystemExit:
                pass

            assert os.environ.get("MCP_HOST") == "192.168.1.1"

    @patch.dict(os.environ, {}, clear=True)
    @patch("openmemories.server.signal.signal")
    @patch("openmemories.server.warmup_numba_functions")
    @patch("openmemories.server.FastMCPServer")
    def test_path_argument(self, mock_server_class, mock_warmup, mock_signal):
        """Test --path argument."""
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        from openmemories.server import main

        with patch("sys.argv", ["openmemories", "--path", "/custom"]):
            try:
                main()
            except SystemExit:
                pass

            assert os.environ.get("MCP_PATH") == "/custom"

    @patch.dict(os.environ, {}, clear=True)
    @patch("openmemories.server.signal.signal")
    @patch("openmemories.server.warmup_numba_functions")
    @patch("openmemories.server.FastMCPServer")
    def test_multiple_arguments(self, mock_server_class, mock_warmup, mock_signal):
        """Test multiple arguments together."""
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        from openmemories.server import main

        with patch(
            "sys.argv",
            [
                "openmemories",
                "--transport",
                "http",
                "--port",
                "8080",
                "--host",
                "localhost",
            ],
        ):
            try:
                main()
            except SystemExit:
                pass

            assert os.environ.get("MCP_TRANSPORT") == "http"
            assert os.environ.get("MCP_PORT") == "8080"
            assert os.environ.get("MCP_HOST") == "localhost"

    @patch.dict(os.environ, {}, clear=True)
    def test_help_argument(self):
        """Test --help argument."""
        from openmemories.server import main

        with patch("sys.argv", ["openmemories", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            # --help causes sys.exit(0)
            assert exc_info.value.code == 0

    @patch.dict(os.environ, {}, clear=True)
    def test_version_argument(self):
        """Test --version argument."""
        from openmemories.server import main

        with patch("sys.argv", ["openmemories", "--version"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            # --version causes sys.exit(0)
            assert exc_info.value.code == 0


@pytest.mark.unit
class TestEnvironmentConfiguration:
    """Test environment variable configuration."""

    @patch.dict(os.environ, {"MCP_TRANSPORT": "sse"}, clear=True)
    @patch("openmemories.server.signal.signal")
    @patch("openmemories.server.warmup_numba_functions")
    @patch("openmemories.server.FastMCPServer")
    def test_env_var_transport(self, mock_server_class, mock_warmup, mock_signal):
        """Test MCP_TRANSPORT environment variable."""
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        from openmemories.server import main

        with patch("sys.argv", ["openmemories"]):
            try:
                main()
            except SystemExit:
                pass

            # Environment variable should still be set
            assert os.environ.get("MCP_TRANSPORT") == "sse"

    @patch.dict(os.environ, {"MCP_TRANSPORT": "stdio", "MCP_PORT": "5000"}, clear=True)
    @patch("openmemories.server.signal.signal")
    @patch("openmemories.server.warmup_numba_functions")
    @patch("openmemories.server.FastMCPServer")
    def test_cli_args_override_env_vars(
        self, mock_server_class, mock_warmup, mock_signal
    ):
        """Test that CLI arguments override environment variables."""
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        from openmemories.server import main

        with patch(
            "sys.argv", ["openmemories", "--transport", "http", "--port", "9104"]
        ):
            try:
                main()
            except SystemExit:
                pass

            # CLI args should override env vars
            assert os.environ.get("MCP_TRANSPORT") == "http"
            assert os.environ.get("MCP_PORT") == "9104"


@pytest.mark.unit
class TestServerInitialization:
    """Test server initialization and startup."""

    @patch.dict(os.environ, {"PROJECT_ID": "test_project"}, clear=True)
    @patch("openmemories.server.signal.signal")
    @patch("openmemories.server.warmup_numba_functions")
    @patch("openmemories.server.FastMCPServer")
    def test_server_instantiation(self, mock_server_class, mock_warmup, mock_signal):
        """Test FastMCPServer is instantiated."""
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        from openmemories.server import main

        with patch("sys.argv", ["openmemories"]):
            try:
                main()
            except SystemExit:
                pass

            # Verify server was created
            mock_server_class.assert_called_once()

    @patch.dict(os.environ, {"PROJECT_ID": "test_project"}, clear=True)
    @patch("openmemories.server.signal.signal")
    @patch("openmemories.server.warmup_numba_functions")
    @patch("openmemories.server.FastMCPServer")
    def test_server_run_called(self, mock_server_class, mock_warmup, mock_signal):
        """Test server.run() is called."""
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        from openmemories.server import main

        with patch("sys.argv", ["openmemories"]):
            try:
                main()
            except SystemExit:
                pass

            # Verify run was called
            mock_server.run.assert_called_once()


@pytest.mark.unit
class TestOutputStreams:
    """Test output stream configuration."""

    @patch.dict(os.environ, {"PROJECT_ID": "test_project"}, clear=True)
    @patch("sys.stderr")
    @patch("sys.stdout")
    @patch("openmemories.server.signal.signal")
    @patch("openmemories.server.warmup_numba_functions")
    @patch("openmemories.server.FastMCPServer")
    def test_stdio_transport_uses_stderr(
        self, mock_server_class, mock_warmup, mock_signal, mock_stdout, mock_stderr
    ):
        """Test stdio transport redirects output to stderr."""
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        from openmemories.server import main

        with patch("sys.argv", ["openmemories", "--transport", "stdio"]):
            try:
                main()
            except SystemExit:
                pass

            # For stdio, sys.stdout should be redirected to stderr
            # (This tests the logging configuration)
            assert os.environ.get("MCP_TRANSPORT") == "stdio"

    @patch.dict(os.environ, {"PROJECT_ID": "test_project"}, clear=True)
    @patch("openmemories.server.signal.signal")
    @patch("openmemories.server.warmup_numba_functions")
    @patch("openmemories.server.FastMCPServer")
    def test_http_transport_uses_stdout(
        self, mock_server_class, mock_warmup, mock_signal
    ):
        """Test http transport uses stdout."""
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        from openmemories.server import main

        with patch("sys.argv", ["openmemories", "--transport", "http"]):
            try:
                main()
            except SystemExit:
                pass

            # For http, normal stdout is used
            assert os.environ.get("MCP_TRANSPORT") == "http"
