"""ReflectLogMCP Server - Refactored modular implementation."""

from typing import Dict, List, Tuple, Type

from fastmcp import FastMCP
from fastmcp.utilities.logging import get_logger

from .config import Config, build_instructions, config
from .memory import MemoryManager
from .tools import (
    AddTool,
    BaseTool,
    GetAllTool,
    HealthCheckTool,
    RemoveTool,
    SearchTool,
)
from .utils import create_logger

# Canonical registry of available MCP tool implementations.
AVAILABLE_TOOL_CLASSES: Dict[str, Type[BaseTool]] = {
    "add": AddTool,
    "get_all": GetAllTool,
    "search": SearchTool,
    "remove": RemoveTool,
    "health_check": HealthCheckTool,
}


class FastMCPServer:
    """Orchestrator for the ReflectLogMCP Server.

    This class coordinates the initialization and registration of all components:
    - Configuration management
    - Memory storage
    - MCP tools
    - Server transport
    """

    def __init__(self, server_config: Config = config):
        """Initialize the MCP server with all components.

        Args:
            server_config: Configuration instance (defaults to singleton).
        """
        super().__init__()
        self.config = server_config

        # Initialize structured logger
        self.logger = create_logger(
            __name__, self.config.project_id, self.config.log_level
        )

        # Log initialization
        self.logger.info(
            f"Initializing reflectlog MCP server [project_id={self.config.project_id}, "
            f"transport={self.config.transport}, port={self.config.port}, "
            f"log_level={self.config.log_level}, embedding_dims={self.config.qwen_embedding_dims if self.config.embedder_provider == 'langchain' else self.config.embedding_dims}]",
            extra={
                "transport": self.config.transport,
                "port": self.config.port,
                "embedding_dims": self.config.embedding_dims,
                "log_level": self.config.log_level,
            },
        )

        # Initialize memory manager
        self._memory_manager = MemoryManager(self.config, self.logger)

        # Initialize tools BEFORE creating FastMCP (to build dynamic instructions)
        self._initialize_tools()

        # Build dynamic instructions based on registered tools
        instructions = self._build_dynamic_instructions()

        # Initialize FastMCP with dynamic instructions
        self.mcp = FastMCP(name="reflectlog", instructions=instructions)

        # Register tools with FastMCP
        self._register_tools()

        self.logger.info("ReflectLogMCP Server initialized successfully")

    def _initialize_tools(self) -> None:
        """Initialize all MCP tool instances."""
        available_names = list(AVAILABLE_TOOL_CLASSES.keys())

        selected_names, invalid_names = self._determine_tool_selection(available_names)

        if invalid_names:
            self.logger.warning(
                "Ignoring unknown tool identifiers from ALLOWED_TOOLS",
                extra={"invalid_tools": sorted(invalid_names)},
            )

        if not selected_names:
            self.logger.warning(
                "No MCP tools selected for registration. Server will start without tools.",
                extra={"available_tools": available_names},
            )

        # Initialize each permitted tool with dependencies
        self.tools: List[BaseTool] = []
        for tool_name in selected_names:
            tool_class = AVAILABLE_TOOL_CLASSES[tool_name]
            tool = tool_class(
                config=self.config,
                memory_manager=self._memory_manager,
                logger=self.logger,
            )
            self.tools.append(tool)

            self.logger.info(
                f"Initialized tool: {tool.get_name()}",
                extra={"tool": tool.get_name()},
            )

        if selected_names:
            self.logger.info(
                "Tool initialization complete",
                extra={"registered_tools": selected_names},
            )

    def _register_tools(self) -> None:
        """Register all tools with the FastMCP instance."""
        for tool in self.tools:
            # Get the handler function from the tool
            handler = tool.get_handler()

            # Register with FastMCP using the decorator
            self.mcp.tool(handler)

            self.logger.info(
                f"Registered tool: {tool.get_name()}", extra={"tool": tool.get_name()}
            )

        self.logger.info(
            f"Registered {len(self.tools)} tools with FastMCP",
            extra={"tool_count": len(self.tools)},
        )

    def _build_dynamic_instructions(self) -> str:
        """Build MCP instructions dynamically from registered tools.

        Assembles instructions by collecting snippets from each initialized tool
        and passing them to the build_instructions() function.

        Returns:
            Complete MCP instructions string with only registered tools documented.
        """
        tool_snippets = [
            (tool.get_name(), tool.get_instruction_snippet()) for tool in self.tools
        ]

        instructions = build_instructions(tool_snippets)

        self.logger.info(
            f"Built dynamic instructions for {len(self.tools)} tool(s)",
            extra={
                "tool_count": len(self.tools),
                "tools": [tool.get_name() for tool in self.tools],
            },
        )

        return instructions

    def _determine_tool_selection(
        self, available_names: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Determine which tools should be initialized based on configuration.

        Args:
            available_names: List of canonical tool identifiers that ship with the server.

        Returns:
            Tuple of (selected tool names, invalid tool names).
        """
        allowed = self.config.allowed_tools

        if allowed is None:
            return available_names, []

        available_set = set(available_names)
        selected: List[str] = []
        invalid: List[str] = []

        for token in allowed:
            canonical = self._canonicalize_tool_token(token, available_set)
            if canonical:
                if canonical not in selected:
                    selected.append(canonical)
            else:
                invalid.append(token)

        return selected, invalid

    @staticmethod
    def _canonicalize_tool_token(token: str, available_set: set[str]) -> str | None:
        """Map a token from configuration to a canonical tool name."""
        normalized = token.strip().lower().replace("-", "_")
        if not normalized:
            return None

        if normalized in available_set:
            return normalized

        collapsed = normalized.replace("_", "")
        for name in available_set:
            if name.replace("_", "") == collapsed:
                return name

        suffixes = ("_tool", "tool")
        for suffix in suffixes:
            if normalized.endswith(suffix):
                base = normalized[: -len(suffix)]
                return FastMCPServer._canonicalize_tool_token(base, available_set)

        return None

    def run(self) -> None:
        """Start the FastMCP server with configured transport.

        The transport configuration is determined by:
        1. Environment variables (MCP_TRANSPORT, MCP_PORT, MCP_HOST, MCP_PATH)
        2. Config defaults
        """
        transport = self.config.transport

        if transport == "stdio":
            self.logger.info("Running MCP server with stdio transport")
            self.mcp.run(transport="stdio")
        else:
            self.logger.info(
                f"Running MCP server with {transport} transport",
                extra={
                    "host": self.config.host,
                    "port": self.config.port,
                    "path": self.config.path,
                },
            )
            self.mcp.run(
                transport=transport,
                port=self.config.port,
                host=self.config.host,
                path=self.config.path,
            )

    def close(self) -> None:
        """Gracefully shutdown the server and persist all data.

        This method ensures all data is properly saved before shutdown:
        1. Closes the MemoryManager (which persists USearch and Tantivy data)

        Should be called during graceful shutdown (e.g., on SIGINT/SIGTERM)
        to prevent data loss.
        """
        self.logger.info("Initiating graceful server shutdown...")

        try:
            self._memory_manager.close()
            self.logger.info("Server shutdown complete - all data persisted")
        except Exception as e:
            self.logger.error(
                f"Error during server shutdown: {e}",
                extra={"error": str(e)},
            )


def main() -> None:
    """Entry point for the ReflectLogMCP server.

    This function:
    1. Loads configuration from environment
    2. Initializes the server
    3. Starts the transport
    4. Handles graceful shutdown on SIGINT/SIGTERM
    """
    # Create fallback logger once for exception handling
    fallback_logger = get_logger(__name__)
    server: FastMCPServer | None = None

    try:
        # Configuration is loaded automatically via the singleton
        server = FastMCPServer()
        server.run()
    except RuntimeError as e:
        fallback_logger.error(f"Failed to start server: {e}")
        raise
    except KeyboardInterrupt:
        fallback_logger.info("Server shutdown requested (Ctrl+C)")
        if server is not None:
            server.close()
    except Exception as e:
        fallback_logger.error(f"Unexpected error: {e}")
        if server is not None:
            server.close()
        raise


if __name__ == "__main__":
    main()
