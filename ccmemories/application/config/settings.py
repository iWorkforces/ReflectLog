"""Configuration management for CCMemoriesMCP Server."""

import os
import re
import threading
from dataclasses import dataclass
from typing import Literal, Optional, TypeAlias, cast

from ..exceptions import ConfigurationError
from ..utils.security import SecretString

# Note: LangchainQwenEmbeddings is imported lazily in MemoryManager
# to avoid unnecessary initialization when not using langchain provider

# Type definitions
TransportMode: TypeAlias = Literal["stdio", "http", "sse", "streamable-http"]


@dataclass(frozen=True)
class Config:
    """Configuration settings for CCMemoriesMCP Server.

    All settings are read from environment variables with sensible defaults.
    This class is immutable (frozen) to prevent accidental modifications at runtime.
    """

    # Core settings (required)
    project_id: str
    openrouter_api_key: SecretString  # Wrapped for security - use .get_secret_value()

    # Transport settings
    transport: TransportMode = "stdio"
    port: int = 9103
    host: str = "127.0.0.1"
    path: str = "/mcp"

    # API settings
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Embedding settings
    embedder_provider: str = "openai"
    embedding_model: str = "openai/text-embedding-3-large"
    embedding_dims: int = 3072
    qwen_embedding_dims: int = 4096

    # Embedding performance settings
    embedding_batch_size: int = 512  # Texts per API request for async batching
    embedding_max_concurrent_batches: int = 4  # Max parallel batch requests
    embedding_cache_enabled: bool = True  # Cache query embeddings (LRU)
    embedding_cache_size: int = 100  # Max cached embeddings

    # Search settings
    search_limit: int = 5
    remove_search_limit: int = 5

    # Hybrid search settings
    enable_hybrid_search: bool = True
    tantivy_index_path_template: str = "indexes/{project_id}/tantivy"
    overfetch_multiplier: int = 3  # Base multiplier (Fetch N * search_limit)
    overfetch_adaptive: bool = True  # Enable adaptive overfetch based on index size
    overfetch_min_multiplier: float = 1.5  # Min multiplier (for large indexes)
    overfetch_max_multiplier: float = 3.0  # Max multiplier (for small indexes)

    # Tantivy soft-delete settings (O(1) delete vs O(n) rebuild)
    tantivy_soft_delete_enabled: bool = True  # Use tombstone marking instead of rebuild
    tantivy_compaction_threshold_ratio: float = (
        0.2  # Compact when tombstones > 20% of docs
    )
    tantivy_compaction_max_tombstones: int = 10000  # Force compaction above this count
    tantivy_tombstone_ttl_days: int = 7  # Days before tombstones eligible for removal

    # Tantivy BM25 score normalization (batch min-max to 0-1 range)
    tantivy_normalize_scores: bool = (
        True  # Normalize BM25 scores for threshold filtering
    )

    # USearch exact search settings
    usearch_exact_search: bool = True  # Force exact brute-force search (bypasses HNSW)
    usearch_exact_search_threshold: int = (
        10000  # Auto-switch to exact when index < threshold
    )

    # Fusion settings (ranx-based)
    fusion_method: str = "rrf"  # rrf, sum, mnz, max, bordafuse
    fusion_normalization: Optional[str] = None  # min-max, max, sum, zmuv, rank, borda
    fusion_rrf_k: int = 60  # RRF k parameter (lower = more weight to top ranks)
    fusion_ranking_threshold: float = 0.8  # Min normalized RRF score to keep (0-1)
    enable_rrf_fusion: bool = True  # Enable RRF fusion (false = concatenate results)

    # Message validation settings
    max_message_length: int = 30720
    min_message_length: int = 1

    # LLM inference settings
    enable_llm_infer: bool = False

    # Reranker engine selection
    reranker_engine: str = "llm"  # "llm", "cross_encoder", or "none"

    # LLM reranking settings (used when reranker_engine="llm")
    llm_model: str = "x-ai/grok-4.1-fast"  # LLM model for reranking
    search_score_threshold: float = 0.5  # Min LLM relevance score to keep (0-1)
    rerank_max_concurrency: int = (
        10  # Max parallel LLM calls for reranking (increased from 5)
    )

    # Cross-encoder reranking settings (used when reranker_engine="cross_encoder")
    # Uses FlagEmbedding's FlagReranker for BGE reranker models
    cross_encoder_model: str = "BAAI/bge-reranker-v2-m3"  # HuggingFace model
    cross_encoder_top_k: int = 20  # Max results to return after reranking
    cross_encoder_device: str = "cpu"  # "cpu", "cuda", "mps"
    cross_encoder_batch_size: int = 32  # Batch size for inference
    cross_encoder_score_threshold: float = 0.5
    cross_encoder_use_fp16: bool = True  # Enable FP16 for faster inference
    cross_encoder_normalize: bool = True  # Normalize scores to 0-1 with sigmoid
    cross_encoder_max_length: int = 512  # Max token length for query-doc pairs

    # Unified reranker settings (apply to both LLM and CrossEncoder)
    reranker_min_results: int = 0  # Safety net: min results to return (0 = disabled)
    reranker_batch_normalize: bool = True  # Enable batch min-max normalization

    # Temporal-aware reranking settings (recency boost for memories)
    enable_recency_boost: bool = True  # Include memory age in reranking context
    recency_decay_rate: float = 0.01  # Decay rate per hour: exp(-rate * hours_old)
    recency_max_boost: float = 0.2  # Maximum recency boost (added to base score)

    # Memory behavior
    deduplicate_messages: bool = True

    # Smart memory replacement settings
    enable_smart_replace: bool = True  # Enable smart memory replacement detection
    smart_replace_threshold: float = 0.7  # Min LLM confidence to trigger replacement
    smart_replace_min_similarity: float = (
        0.9  # Min embedding similarity to trigger LLM check
    )
    smart_replace_candidate_limit: int = 3  # Max candidates to check for replacement
    smart_replace_archive_ttl_days: int = (
        30  # Days to keep archived memories (0 = permanent)
    )
    smart_replace_max_retries: int = 3  # Max LLM call retries
    smart_replace_retry_delay: float = (
        1.0  # Base delay (seconds) for exponential backoff
    )
    llm_provider: str = "anthropic"  # LLM provider: "openai" or "anthropic"

    # Concurrency settings
    add_max_concurrency: int = 4  # Max concurrent message additions

    # Initialization settings
    eager_initialization: bool = True  # Pre-warm engines during MemoryManager init

    # Logging settings
    log_level: str = "INFO"
    log_search_results_verbose: bool = False  # Log individual search results
    log_search_result_limit: int = 3  # Max results to log when verbose enabled

    # Tool registration settings
    allowed_tools: tuple[str, ...] | None = None

    @classmethod
    def from_environment(cls) -> "Config":
        """Create configuration from environment variables.

        This method centralizes environment parsing and performs strict validation
        for all critical settings. It fails fast with clear, non-secret-leaking
        error messages when configuration is invalid.

        Raises:
            ConfigurationError: If required environment variables are missing or invalid.
        """
        # Check required variables
        project_id = os.environ.get("PROJECT_ID")
        if not project_id:
            raise ConfigurationError(
                "The PROJECT_ID environment variable has not been configured."
            )

        # Enforce a safe PROJECT_ID format to avoid path traversal and
        # filesystem issues. Keep the rule intentionally strict.
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", project_id):
            raise ConfigurationError(
                "Invalid PROJECT_ID: only A-Za-z0-9_.- allowed, max length 64."
            )

        # Check for path traversal patterns (not caught by regex)
        if ".." in project_id or project_id.startswith("/"):
            raise ConfigurationError(
                f"Invalid PROJECT_ID: path traversal patterns not allowed: {project_id}"
            )

        openrouter_api_key_raw = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_api_key_raw:
            raise ConfigurationError(
                "The OPENROUTER_API_KEY environment variable has not been configured."
            )
        # Wrap in SecretString to prevent accidental logging
        openrouter_api_key = SecretString(openrouter_api_key_raw)

        # Get transport mode with validation
        transport_raw = os.environ.get("MCP_TRANSPORT", "stdio")
        valid_transports: tuple[TransportMode, ...] = (
            "stdio",
            "http",
            "sse",
            "streamable-http",
        )
        if transport_raw not in valid_transports:
            transport: TransportMode = "stdio"
        else:
            # We've validated it's in valid_transports, so cast is safe
            transport = cast(TransportMode, transport_raw)

        # Parse allowed tools configuration (comma-separated)
        def _normalize_tool_token(raw_token: str) -> str:
            """Normalize tool identifiers to snake_case."""
            token = raw_token.strip()
            if not token:
                return ""

            token = token.replace("-", "_")
            token = re.sub(r"\s+", "_", token)
            if token.isupper():
                token = token.lower()
            else:
                token = re.sub(r"(?<!^)(?=[A-Z])", "_", token).lower()
            token = re.sub(r"_+", "_", token)
            token = token.strip("_")
            return token

        allowed_tools_env = os.environ.get("ALLOWED_TOOLS")
        allowed_tools: tuple[str, ...] | None = None
        if allowed_tools_env is not None:
            entries = [
                entry.strip() for entry in allowed_tools_env.split(",") if entry.strip()
            ]

            normalized: list[str] = []
            allow_all = False

            for entry in entries:
                token = _normalize_tool_token(entry)
                if token in {"all", "*"}:
                    allow_all = True
                    break
                normalized.append(token)

            if allow_all:
                allowed_tools = None
            else:
                if normalized and all(
                    token in {"none", "disabled"} for token in normalized
                ):
                    allowed_tools = tuple()
                else:
                    # Preserve order while removing duplicates
                    seen: dict[str, None] = {}
                    for token in normalized:
                        if token not in seen:
                            seen[token] = None
                    allowed_tools = tuple(seen.keys())

            # If ALLOWED_TOOLS was set to an empty/blank value, respect it
            if allowed_tools_env.strip() == "":
                allowed_tools = tuple()

        # Determine reranker engine
        reranker_engine_raw = os.environ.get("RERANKER_ENGINE", "llm")
        reranker_engine = reranker_engine_raw.lower()
        valid_engines = ("llm", "cross_encoder", "none")
        if reranker_engine not in valid_engines:
            raise ConfigurationError(
                f"Invalid RERANKER_ENGINE: '{reranker_engine}'. "
                f"Valid options: {', '.join(valid_engines)}"
            )

        # Determine LLM provider (used by SmartReplacer and LLMReranker)
        llm_provider_raw = os.environ.get("LLM_PROVIDER", "anthropic")
        llm_provider = llm_provider_raw.lower()
        valid_llm_providers = ("openai", "anthropic")
        if llm_provider not in valid_llm_providers:
            raise ConfigurationError(
                f"Invalid LLM_PROVIDER: '{llm_provider}'. "
                f"Valid options: {', '.join(valid_llm_providers)}"
            )

        return cls(
            # Required
            project_id=project_id,
            openrouter_api_key=openrouter_api_key,
            # Transport
            transport=transport,
            port=int(os.environ.get("PORT", os.environ.get("MCP_PORT", "9103"))),
            host=os.environ.get("MCP_HOST", "127.0.0.1"),
            path=os.environ.get("MCP_PATH", "/mcp"),
            # API
            openrouter_base_url=os.environ.get(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            # Embeddings
            embedder_provider=os.environ.get("EMBEDDER_PROVIDER", "openai"),
            embedding_model=os.environ.get(
                "EMBEDDING_MODEL", "openai/text-embedding-3-large"
            ),
            embedding_dims=int(os.environ.get("EMBEDDING_DIMS", "3072")),
            qwen_embedding_dims=int(os.environ.get("QWEN_EMBEDDING_DIMS", "4096")),
            # Embedding performance
            embedding_batch_size=max(
                1, int(os.environ.get("EMBEDDING_BATCH_SIZE", "512"))
            ),
            embedding_max_concurrent_batches=max(
                1, int(os.environ.get("EMBEDDING_MAX_CONCURRENT_BATCHES", "4"))
            ),
            embedding_cache_enabled=os.environ.get(
                "EMBEDDING_CACHE_ENABLED", "true"
            ).lower()
            == "true",
            embedding_cache_size=max(
                1, int(os.environ.get("EMBEDDING_CACHE_SIZE", "100"))
            ),
            # Search
            search_limit=int(os.environ.get("SEARCH_LIMIT", "5")),
            remove_search_limit=int(os.environ.get("REMOVE_SEARCH_LIMIT", "5")),
            # Hybrid search settings
            enable_hybrid_search=os.environ.get("ENABLE_HYBRID_SEARCH", "true").lower()
            == "true",
            tantivy_index_path_template="indexes/{project_id}/tantivy",
            overfetch_multiplier=max(
                1, int(os.environ.get("OVERFETCH_MULTIPLIER", "3"))
            ),
            overfetch_adaptive=os.environ.get("OVERFETCH_ADAPTIVE", "true").lower()
            == "true",
            overfetch_min_multiplier=max(
                1.0, float(os.environ.get("OVERFETCH_MIN_MULTIPLIER", "1.5"))
            ),
            overfetch_max_multiplier=max(
                1.0, float(os.environ.get("OVERFETCH_MAX_MULTIPLIER", "3.0"))
            ),
            # Tantivy soft-delete settings
            tantivy_soft_delete_enabled=os.environ.get(
                "TANTIVY_SOFT_DELETE_ENABLED", "true"
            ).lower()
            == "true",
            tantivy_compaction_threshold_ratio=max(
                0.01,
                min(
                    1.0,
                    float(os.environ.get("TANTIVY_COMPACTION_THRESHOLD_RATIO", "0.2")),
                ),
            ),
            tantivy_compaction_max_tombstones=max(
                100, int(os.environ.get("TANTIVY_COMPACTION_MAX_TOMBSTONES", "10000"))
            ),
            tantivy_tombstone_ttl_days=max(
                0, int(os.environ.get("TANTIVY_TOMBSTONE_TTL_DAYS", "7"))
            ),
            # Tantivy BM25 score normalization
            tantivy_normalize_scores=os.environ.get(
                "TANTIVY_NORMALIZE_SCORES", "true"
            ).lower()
            == "true",
            # USearch exact search settings
            usearch_exact_search=os.environ.get("USEARCH_EXACT_SEARCH", "true").lower()
            == "true",
            usearch_exact_search_threshold=max(
                0, int(os.environ.get("USEARCH_EXACT_SEARCH_THRESHOLD", "10000"))
            ),
            # Fusion settings (ranx-based)
            fusion_method=os.environ.get("FUSION_METHOD", "rrf").lower(),
            fusion_normalization=os.environ.get("FUSION_NORMALIZATION") or None,
            fusion_rrf_k=int(os.environ.get("FUSION_RRF_K", "60")),
            fusion_ranking_threshold=float(
                os.environ.get("FUSION_RANKING_THRESHOLD", "0.8")
            ),
            enable_rrf_fusion=os.environ.get("ENABLE_RRF_FUSION", "true").lower()
            == "true",
            # Message validation
            max_message_length=int(os.environ.get("MAX_MESSAGE_LENGTH", "30720")),
            min_message_length=int(os.environ.get("MIN_MESSAGE_LENGTH", "1")),
            # LLM inference
            enable_llm_infer=os.environ.get("ENABLE_LLM_INFER", "false").lower()
            == "true",
            # Reranker engine selection
            reranker_engine=reranker_engine,
            # LLM reranking settings
            llm_model=os.environ.get("LLM_MODEL", "x-ai/grok-4.1-fast"),
            search_score_threshold=float(
                os.environ.get("SEARCH_SCORE_THRESHOLD", "0.5")
            ),
            rerank_max_concurrency=max(
                1, int(os.environ.get("RERANK_MAX_CONCURRENCY", "10"))
            ),
            # Cross-encoder reranking settings (FlagReranker)
            cross_encoder_model=os.environ.get(
                "CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3"
            ),
            cross_encoder_top_k=int(os.environ.get("CROSS_ENCODER_TOP_K", "20")),
            cross_encoder_device=os.environ.get("CROSS_ENCODER_DEVICE", "cpu"),
            cross_encoder_batch_size=int(
                os.environ.get("CROSS_ENCODER_BATCH_SIZE", "32")
            ),
            cross_encoder_score_threshold=float(
                os.environ.get("CROSS_ENCODER_SCORE_THRESHOLD", "0.5")
            ),
            cross_encoder_use_fp16=os.environ.get(
                "CROSS_ENCODER_USE_FP16", "true"
            ).lower()
            == "true",
            cross_encoder_normalize=os.environ.get(
                "CROSS_ENCODER_NORMALIZE", "true"
            ).lower()
            == "true",
            cross_encoder_max_length=int(
                os.environ.get("CROSS_ENCODER_MAX_LENGTH", "512")
            ),
            # Unified reranker settings
            reranker_min_results=max(
                0, int(os.environ.get("RERANKER_MIN_RESULTS", "0"))
            ),
            reranker_batch_normalize=os.environ.get(
                "RERANKER_BATCH_NORMALIZE", "true"
            ).lower()
            == "true",
            # Temporal-aware reranking settings (recency boost for memories)
            enable_recency_boost=os.environ.get("ENABLE_RECENCY_BOOST", "true").lower()
            == "true",
            recency_decay_rate=max(
                0.0, float(os.environ.get("RECENCY_DECAY_RATE", "0.01"))
            ),
            recency_max_boost=max(
                0.0, float(os.environ.get("RECENCY_MAX_BOOST", "0.2"))
            ),
            # Memory behavior
            deduplicate_messages=os.environ.get("DEDUPLICATE_MESSAGES", "true").lower()
            == "true",
            # Smart memory replacement settings
            enable_smart_replace=os.environ.get("ENABLE_SMART_REPLACE", "true").lower()
            == "true",
            smart_replace_threshold=float(
                os.environ.get("SMART_REPLACE_THRESHOLD", "0.7")
            ),
            smart_replace_min_similarity=float(
                os.environ.get("SMART_REPLACE_MIN_SIMILARITY", "0.9")
            ),
            smart_replace_candidate_limit=max(
                1, int(os.environ.get("SMART_REPLACE_CANDIDATE_LIMIT", "3"))
            ),
            smart_replace_archive_ttl_days=max(
                0, int(os.environ.get("SMART_REPLACE_ARCHIVE_TTL_DAYS", "30"))
            ),
            smart_replace_max_retries=max(
                1, int(os.environ.get("SMART_REPLACE_MAX_RETRIES", "3"))
            ),
            smart_replace_retry_delay=max(
                0.1, float(os.environ.get("SMART_REPLACE_RETRY_DELAY", "1.0"))
            ),
            llm_provider=llm_provider,
            # Concurrency settings
            add_max_concurrency=max(1, int(os.environ.get("ADD_MAX_CONCURRENCY", "4"))),
            # Initialization settings
            eager_initialization=os.environ.get("EAGER_INITIALIZATION", "true").lower()
            == "true",
            # Logging
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            log_search_results_verbose=os.environ.get(
                "LOG_SEARCH_RESULTS_VERBOSE", "false"
            ).lower()
            == "true",
            log_search_result_limit=int(os.environ.get("LOG_SEARCH_RESULT_LIMIT", "3")),
            # Tools
            allowed_tools=allowed_tools,
        )


# Thread-safe lazy singleton pattern - config is only created when first accessed
_config: Config | None = None
_config_lock = threading.Lock()


def get_config() -> Config:
    """Get the configuration singleton (lazy initialization).

    This delays environment variable parsing until first access,
    avoiding startup overhead for imports that don't need config.

    Thread-safe: Uses double-checked locking pattern to ensure only
    one Config instance is created even under concurrent access.

    Returns:
        The Config singleton instance.
    """
    global _config

    # Fast path: already initialized (use local var for type narrowing)
    local_config = _config
    if local_config is not None:
        return local_config

    # Slow path: need to initialize with lock
    with _config_lock:
        # Double-check inside lock to prevent race condition
        local_config = _config
        if local_config is not None:
            return local_config

        # Create and cache the singleton
        new_config = Config.from_environment()
        _config = new_config
        return new_config


# Backward-compatible alias (for existing code using `from config import config`)
# Note: This will trigger lazy init on attribute access, not import
class _LazyConfig:
    """Lazy proxy for backward compatibility with `from config import config`.

    This proxy class forwards all attribute access to the lazily-initialized
    Config singleton, enabling imports like `from config import config` without
    triggering immediate environment variable parsing.
    """

    def __getattr__(self, name: str):
        return getattr(get_config(), name)

    def __repr__(self) -> str:
        return f"_LazyConfig(initialized={_config is not None})"


# Note: This is typed as Any to allow type checkers to accept it where Config is expected
# The actual Config object is accessed lazily through __getattr__
config: Config = _LazyConfig()  # type: ignore[assignment]
