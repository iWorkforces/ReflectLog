import argparse
import os
import signal
import sys
import threading
import time
from typing import TYPE_CHECKING

# Add parent directory to path for direct script execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure numba environment before imports
for key, value in [
    ("NUMBA_CACHE_DIR", os.path.join(os.getcwd(), ".numba_cache")),
    ("NUMBA_THREADING_LAYER", "workqueue"),
    ("NUMBA_DEBUG", "0"),
    ("NUMBA_FASTMATH", "1"),
]:
    _ = os.environ.setdefault(key, value)

if TYPE_CHECKING:
    from _typeshed import SupportsWrite

from reflectlog.application.mcp_server import FastMCPServer  # noqa: E402
from reflectlog.utility.scoring import (  # noqa: E402
    warmup_numba_functions,
)
from reflectlog.version import __version__  # noqa: E402


def warmup_numba_with_config(
    enabled: bool = True,
    mode: str = "sync",
    output_stream: SupportsWrite[str] | None = None,
) -> threading.Thread | None:
    """Warm up numba JIT functions with configurable execution mode.

    Args:
        enabled: Whether to perform JIT warmup at all.
        mode: Execution mode - "sync" (default), "async" (background thread),
            or "background" (daemon thread).
        output_stream: Stream to print progress (stderr for stdio, stdout otherwise).

    Returns:
        Thread object if mode is "async" or "background", None otherwise.

    Raises:
        ValueError: If mode is not one of "sync", "async", or "background".
    """
    if not enabled:
        if output_stream:
            print("Numba JIT warmup disabled (NUMBA_WARMUP=false)", file=output_stream)
        return None

    valid_modes = ("sync", "async", "background")
    if mode not in valid_modes:
        modes_str = ", ".join(valid_modes)
        raise ValueError(
            f"Invalid NUMBA_WARMUP_MODE: '{mode}'. Valid options: {modes_str}"
        )

    if mode == "sync":
        if output_stream:
            print("Warming up numba JIT functions (synchronous)...", file=output_stream)
        _ = warmup_numba_functions()
        if output_stream:
            print("Numba functions compiled and cached", file=output_stream)
        return None
    else:
        # async or background mode
        is_daemon = mode == "background"
        if output_stream:
            mode_desc = "background daemon thread" if is_daemon else "background thread"
            print(
                f"Warming up numba JIT functions ({mode_desc})...", file=output_stream
            )

        def warmup_worker():
            try:
                _ = warmup_numba_functions()
                if output_stream:
                    print(
                        "Numba functions compiled and cached (background complete)",
                        file=output_stream,
                    )
            except Exception as e:
                if output_stream:
                    print(f"Numba warmup warning: {e}", file=output_stream)

        thread = threading.Thread(target=warmup_worker, daemon=is_daemon)
        thread.start()
        return thread


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        prog="reflectlog",
        description="MCP Server For Claude Code Project Memories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default stdio transport (for MCP clients)
  reflectlog

  # Run as HTTP server
  reflectlog --transport http --port 9103

  # Run with SSE transport
  reflectlog --transport sse --port 8080 --host 0.0.0.0

Environment Variables:
  WORKSPACE_ID    The unique workspace name
  MCP_TRANSPORT   Override transport mode (stdio, http, sse, streamable-http)
  MCP_PORT        Override server port
  MCP_HOST        Override server host
  MCP_PATH        Override server path
        """,
    )

    _ = parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show version and exit",
    )

    _ = parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "http", "sse", "streamable-http"],
        help="Transport protocol (default: stdio for tool usage, or from settings)",
    )

    _ = parser.add_argument(
        "--port",
        type=int,
        help="Server port for non-stdio transports (default: 9103)",
    )

    _ = parser.add_argument(
        "--host",
        type=str,
        help="Server host for non-stdio transports (default: 127.0.0.1)",
    )

    _ = parser.add_argument(
        "--path",
        type=str,
        help="Server path for non-stdio transports (default: /mcp)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the MCP server.

    Supports both CLI argument parsing and environment variable configuration.
    When run as a tool (via `reflectlog` command), defaults to stdio transport.

    Graceful shutdown is handled via signal handlers for SIGINT (Ctrl+C) and SIGTERM,
    ensuring all data from TantivyEngine and USearchEngine is persisted before exit.
    """
    args = parse_args()
    transport_mode = _apply_cli_env_vars(args)
    output_stream = sys.stderr if transport_mode == "stdio" else sys.stdout

    print(
        "Starting ReflectLog - Project-based AI Agent Memories...",
        file=output_stream,
    )
    print(f"Version: {__version__}", file=output_stream)
    print(f"Transport: {transport_mode}", file=output_stream)

    startup_start_time = time.time()
    startup_phases = _run_numba_warmup(output_stream)

    server = _start_server(output_stream, startup_start_time, startup_phases)
    server.run()


def _apply_cli_env_vars(args: argparse.Namespace) -> str:
    """Set environment variables from CLI args and return transport mode."""
    if args.transport:
        os.environ["MCP_TRANSPORT"] = args.transport
    elif "MCP_TRANSPORT" not in os.environ:
        os.environ["MCP_TRANSPORT"] = "stdio"

    if args.port:
        os.environ["MCP_PORT"] = str(args.port)
    if args.host:
        os.environ["MCP_HOST"] = args.host
    if args.path:
        os.environ["MCP_PATH"] = args.path

    return os.environ.get("MCP_TRANSPORT", "stdio")


def _run_numba_warmup(
    output_stream: SupportsWrite[str],
) -> dict[str, float]:
    """Run numba JIT warmup phase and return startup phases dict."""
    numba_warmup_enabled = os.environ.get("NUMBA_WARMUP", "true").lower() == "true"
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    default_warmup_mode = "sync" if transport == "stdio" else "background"
    numba_warmup_mode = os.environ.get("NUMBA_WARMUP_MODE", default_warmup_mode).lower()

    startup_phases: dict[str, float] = {}
    numba_start = time.time()
    _ = warmup_numba_with_config(
        enabled=numba_warmup_enabled,
        mode=numba_warmup_mode,
        output_stream=output_stream,
    )
    startup_phases["numba_warmup"] = time.time() - numba_start
    return startup_phases


def _start_server(
    output_stream: SupportsWrite[str],
    startup_start_time: float,
    startup_phases: dict[str, float],
) -> FastMCPServer:
    """Initialize server with signal handlers and startup metrics.

    Registers SIGINT/SIGTERM handlers for graceful shutdown.
    """
    server: FastMCPServer | None = None

    def graceful_shutdown(signum: int, frame: object) -> None:
        """Signal handler for graceful shutdown."""
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        print(
            f"\nReceived {signal_name}, initiating graceful shutdown...",
            file=output_stream,
        )
        if server is not None:
            server.close()
        print("Shutdown complete.", file=output_stream)
        sys.exit(0)

    _ = signal.signal(signal.SIGINT, graceful_shutdown)
    _ = signal.signal(signal.SIGTERM, graceful_shutdown)

    try:
        phase_start = time.time()
        server = FastMCPServer()
        startup_phases["server_initialization"] = time.time() - phase_start

        total_startup_time = time.time() - startup_start_time
        startup_phases["total_startup"] = total_startup_time

        print(
            f"Server startup completed in {total_startup_time * 1000:.1f}ms",
            file=output_stream,
        )
        if os.environ.get("STARTUP_TIMING_VERBOSE", "false").lower() == "true":
            print("Startup timing breakdown:", file=output_stream)
            for phase, duration in startup_phases.items():
                print(f"  {phase}: {duration * 1000:.1f}ms", file=output_stream)

        server.set_startup_metrics(startup_phases)
        return server
    except KeyboardInterrupt:
        print("\nServer shutdown requested...", file=output_stream)
        if server is not None:
            server.close()
        raise
    except Exception as e:
        print(f"Error during server operation: {e}", file=output_stream)
        if server is not None:
            server.close()
        raise


if __name__ == "__main__":
    main()
