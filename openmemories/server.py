import sys
import os
import argparse
import signal
from typing import Optional

# Add the current directory to Python path for direct execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure numba before any imports that use it
# Enable caching for JIT-compiled functions to avoid recompilation on restart
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.getcwd(), ".numba_cache"))
# Use threading backend for parallel execution
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
# Disable JIT debugging in production for better performance
os.environ.setdefault("NUMBA_DEBUG", "0")
# Enable fastmath for additional floating-point optimizations
os.environ.setdefault("NUMBA_FASTMATH", "1")

from openmemories import __version__
from openmemories.application.mcp_server import FastMCPServer
from openmemories.application.utils.numba_utils import warmup_numba_functions


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        prog="openmemories",
        description="MCP Server For Claude Code Project Memories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default stdio transport (for MCP clients)
  openmemories

  # Run as HTTP server
  openmemories --transport http --port 9104

  # Run with SSE transport
  openmemories --transport sse --port 8080 --host 0.0.0.0

Environment Variables:
  PROJECT_ID    The unique project name
  MCP_TRANSPORT   Override transport mode (stdio, http, sse, streamable-http)
  MCP_PORT        Override server port
  MCP_HOST        Override server host
  MCP_PATH        Override server path
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show version and exit",
    )

    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "http", "sse", "streamable-http"],
        help="Transport protocol (default: stdio for tool usage, or from settings.toml)",
    )

    parser.add_argument(
        "--port",
        type=int,
        help="Server port for non-stdio transports (default: 9104)",
    )

    parser.add_argument(
        "--host",
        type=str,
        help="Server host for non-stdio transports (default: 127.0.0.1)",
    )

    parser.add_argument(
        "--path",
        type=str,
        help="Server path for non-stdio transports (default: /mcp)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the MCP server.

    Supports both CLI argument parsing and environment variable configuration.
    When run as a tool (via `openmemories` command), defaults to stdio transport.

    Graceful shutdown is handled via signal handlers for SIGINT (Ctrl+C) and SIGTERM,
    ensuring all data from TantivyEngine and USearchEngine is persisted before exit.
    """
    # Parse command-line arguments
    args = parse_args()

    # Set environment variables based on CLI args or defaults
    # Priority: CLI args > existing env vars > default (stdio for tool usage)
    if args.transport:
        os.environ["MCP_TRANSPORT"] = args.transport
    elif "MCP_TRANSPORT" not in os.environ:
        # Default to stdio when run as installed tool
        os.environ["MCP_TRANSPORT"] = "stdio"

    if args.port:
        os.environ["MCP_PORT"] = str(args.port)

    if args.host:
        os.environ["MCP_HOST"] = args.host

    if args.path:
        os.environ["MCP_PATH"] = args.path

    # Get the transport mode to determine where to print messages
    transport_mode = os.environ.get("MCP_TRANSPORT", "stdio")

    # In stdio mode, ALL output must go to stderr to avoid corrupting JSON-RPC protocol
    # Only JSON-RPC messages should go to stdout in stdio mode
    output_stream = sys.stderr if transport_mode == "stdio" else sys.stdout

    print(
        "Starting OpenMemoriesMCP - Project-based AI Agent Memories...",
        file=output_stream,
    )
    print(f"Version: {__version__}", file=output_stream)
    print(f"Transport: {transport_mode}", file=output_stream)

    # Pre-compile numba JIT functions to avoid first-call latency
    print("Warming up numba JIT functions...", file=output_stream)
    warmup_numba_functions()
    print("Numba functions compiled and cached", file=output_stream)

    # Create server with dependency injection
    server: Optional[FastMCPServer] = None

    def graceful_shutdown(signum: int, frame: object) -> None:
        """Signal handler for graceful shutdown.

        Ensures all data from TantivyEngine and USearchEngine is persisted
        before the process exits.
        """
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        print(
            f"\nReceived {signal_name}, initiating graceful shutdown...",
            file=output_stream,
        )

        if server is not None:
            server.close()

        print("Shutdown complete.", file=output_stream)
        sys.exit(0)

    # Register signal handlers for graceful shutdown
    # SIGINT: Ctrl+C from terminal
    # SIGTERM: Standard termination signal (e.g., from process managers)
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    try:
        server = FastMCPServer()
        server.run()
    except KeyboardInterrupt:
        # KeyboardInterrupt may be raised before signal handler is fully set up
        print("\nServer shutdown requested...", file=output_stream)
        if server is not None:
            server.close()
    except Exception as e:
        print(f"Error during server operation: {e}", file=output_stream)
        if server is not None:
            server.close()
        raise


if __name__ == "__main__":
    main()
