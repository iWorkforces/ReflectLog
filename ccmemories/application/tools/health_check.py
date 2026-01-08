"""Health check tool implementation for CCMemoriesMCP Server."""

from typing import Any, Dict

from .base import BaseTool


class HealthCheckTool(BaseTool):
    """Tool for health checking the server and its components."""

    def get_name(self) -> str:
        """Get the tool name."""
        return "health_check"

    def get_instruction_snippet(self) -> str:
        """Get the instruction snippet for MCP_INSTRUCTIONS."""
        return (
            "    • health_check() -> dict[str, Any]\n"
            "      Get server health status including component initialization state."
        )

    def get_handler(self):
        """Get the async tool handler function."""

        async def health_check() -> Dict[str, Any]:
            """Get server health status (async).

            Returns a health status dictionary with information about:
            - Overall server status
            - Project ID
            - Semantic engine initialization state
            - Tantivy full-text engine state
            - Configured reranker engine

            This tool is useful for monitoring and debugging server state.

            Returns:
                Dictionary containing health status information with keys:
                - status: Overall status string ("healthy")
                - project_id: The configured project identifier
                - semantic_engine: "initialized" if semantic engine is ready
                - tantivy_engine: "initialized" or "disabled" based on hybrid search
                - reranker_engine: The configured reranker type
                - hybrid_search_enabled: Whether full-text search is enabled
                - rrf_fusion_enabled: Whether RRF fusion is enabled
                - recency_boost_enabled: Whether recency boost is enabled
                - startup_metrics: Optional dict with startup timing data (milliseconds)

            Examples:
                >>> health_check()
                {
                    "status": "healthy",
                    "project_id": "my-project",
                    "semantic_engine": "initialized",
                    "tantivy_engine": "initialized",
                    "reranker_engine": "llm",
                    "hybrid_search_enabled": True,
                    "rrf_fusion_enabled": True,
                    "recency_boost_enabled": True,
                    "startup_metrics": {
                        "numba_warmup": 250.5,
                        "server_initialization": 150.3,
                        "total_startup": 400.8
                    }
                }
            """
            try:
                self.log_invocation("health_check")

                # Check semantic engine state
                semantic_engine_status = (
                    "initialized"
                    if self.memory._semantic_engine is not None
                    else "not_initialized"
                )

                # Check Tantivy engine state
                tantivy_engine_status = (
                    "initialized"
                    if self.memory._tantivy_engine is not None
                    else "disabled"
                )

                # Build health status response
                health_status = {
                    "status": "healthy",
                    "project_id": self.config.project_id,
                    "semantic_engine": semantic_engine_status,
                    "tantivy_engine": tantivy_engine_status,
                    "reranker_engine": self.config.reranker_engine,
                    "hybrid_search_enabled": self.config.enable_hybrid_search,
                    "rrf_fusion_enabled": self.config.enable_rrf_fusion,
                    "recency_boost_enabled": self.config.enable_recency_boost,
                }

                # Add startup metrics if available
                if self.memory._startup_metrics is not None:
                    # Convert seconds to milliseconds for better readability
                    startup_metrics_ms = {
                        phase: duration * 1000
                        for phase, duration in self.memory._startup_metrics.items()
                    }
                    health_status["startup_metrics"] = startup_metrics_ms

                self.log_completion("health_check", status=health_status["status"])

                return health_status

            except Exception as e:
                self.log_error("health_check", e)
                # Return unhealthy status on error
                return {
                    "status": "unhealthy",
                    "project_id": self.config.project_id,
                    "error": str(e),
                }

        return health_check
