"""Configuration management for ReflectLog Server."""

from dataclasses import dataclass
import os
import re
import threading
from typing import Literal, TypedDict

from reflectlog.core.exceptions import ConfigurationError

from ..utils.security import SecretString
from .presets import apply_preset_to_env, get_active_preset

# Note: LangchainQwenEmbeddings is imported lazily in MemoryManager
# to avoid unnecessary initialization when not using langchain provider

# Type definitions
type TransportMode = Literal["stdio", "http", "sse", "streamable-http"]


class TransportConfigDict(TypedDict):
    transport: TransportMode
    port: int
    host: str
    path: str
    openrouter_base_url: str


class EmbeddingConfigDict(TypedDict):
    embedder_provider: str
    embedding_model: str
    embedding_dims: int
    qwen_embedding_dims: int
    embedding_batch_size: int
    embedding_max_concurrent_batches: int
    embedding_cache_enabled: bool
    embedding_cache_size: int


class SearchConfigDict(TypedDict):
    search_limit: int
    remove_search_limit: int
    enable_hybrid_search: bool
    tantivy_index_path_template: str
    overfetch_multiplier: int
    overfetch_adaptive: bool
    overfetch_min_multiplier: float
    overfetch_max_multiplier: float
    usearch_exact_search: bool
    usearch_exact_search_threshold: int
    fusion_method: str
    fusion_normalization: str | None
    fusion_rrf_k: int
    fusion_ranking_threshold: float
    enable_rrf_fusion: bool
    fusion_weights: list[float] | None


class RerankerConfigDict(TypedDict):
    reranker_engine: str
    llm_provider: str
    llm_model: str
    search_score_threshold: float
    rerank_max_concurrency: int
    cross_encoder_model: str
    cross_encoder_top_k: int
    cross_encoder_device: str
    cross_encoder_batch_size: int
    cross_encoder_score_threshold: float
    cross_encoder_use_fp16: bool
    cross_encoder_normalize: bool
    cross_encoder_max_length: int
    reranker_min_results: int
    reranker_batch_normalize: bool
    enable_recency_boost: bool
    recency_decay_rate: float


class StorageConfigDict(TypedDict):
    tantivy_soft_delete_enabled: bool
    tantivy_compaction_threshold_ratio: float
    tantivy_compaction_max_tombstones: int
    tantivy_tombstone_ttl_days: int
    tantivy_normalize_scores: bool


class SecurityConfigDict(TypedDict):
    max_memory_length: int
    min_memory_length: int
    deduplicate_memories: bool


class LoggingConfigDict(TypedDict):
    log_level: str
    log_search_results_verbose: bool
    log_search_result_limit: int


class MetricsConfigDict(TypedDict):
    enable_smart_replace: bool
    smart_replace_threshold: float
    smart_replace_min_similarity: float
    smart_replace_candidate_limit: int
    smart_replace_archive_ttl_days: int
    smart_replace_max_retries: int
    smart_replace_retry_delay: float


class ServerConfigDict(TypedDict):
    add_max_concurrency: int
    eager_initialization: bool
    eager_initialize_search_engines: bool | None
    eager_initialize_reranker: bool | None
    eager_initialize_smart_replacer: bool | None


def _parse_optional_bool(value: str | None) -> bool | None:
    """Parse an optional boolean environment variable.

    Returns:
        True if value is "true" or "1", False if value is "false" or "0",
        None if value is None or empty string.

    Examples:
        _parse_optional_bool("true") -> True
        _parse_optional_bool("false") -> False
        _parse_optional_bool(None) -> None
        _parse_optional_bool("") -> None
    """
    if value is None or value.strip() == "":
        return None
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes", "on"):
        return True
    if normalized in ("false", "0", "no", "off"):
        return False
    return None


@dataclass(frozen=True)
class Config:
    """Configuration settings for ReflectLog Server.

    All settings are read from environment variables with sensible defaults.
    This class is immutable (frozen) to prevent accidental modifications at runtime.
    """

    # Core settings (required)
    workspace_id: str
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
    tantivy_index_path_template: str = "indexes/{workspace_id}/tantivy"
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
    usearch_exact_search: bool = False  # HNSW approximate search (exact is opt-in)
    usearch_exact_search_threshold: int = (
        256  # Auto-switch to exact when index is smaller than this
    )

    # Fusion settings (ranx-based)
    fusion_method: str = "rrf"  # rrf, sum, mnz, max, bordafuse
    fusion_normalization: str | None = None  # min-max, max, sum, zmuv, rank, borda
    fusion_rrf_k: int = 60  # RRF k parameter (lower = more weight to top ranks)
    fusion_ranking_threshold: float = 0.8  # Min normalized RRF score to keep (0-1)
    enable_rrf_fusion: bool = True  # Enable RRF fusion (false = concatenate results)
    fusion_weights: list[float] | None = None  # Per-run weights for weighted RRF

    # Memory validation settings
    max_memory_length: int = 30720
    min_memory_length: int = 1

    # Reranker engine selection
    reranker_engine: str = "cross_encoder"  # "llm", "cross_encoder", or "none"

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

    # Memory behavior
    deduplicate_memories: bool = True

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
    add_max_concurrency: int = 4  # Max concurrent memory additions

    # Initialization settings
    eager_initialization: bool = True  # Pre-warm engines during MemoryManager init
    enable_llm_infer: bool = False  # Enable LLM memory inference

    # Granular eager initialization settings (for fine-grained control)
    # These override eager_initialization when set
    eager_initialize_search_engines: bool | None = (
        None  # Pre-warm USearch/Tantivy (default: true if eager_initialization)
    )
    eager_initialize_reranker: bool | None = (
        None  # Pre-load reranker (default: false - lazy load on first search)
    )
    eager_initialize_smart_replacer: bool | None = (
        None  # Pre-load SmartReplacer (default: false - lazy load on first add)
    )

    # Logging settings
    log_level: str = "INFO"
    log_search_results_verbose: bool = False  # Log individual search results
    log_search_result_limit: int = 3  # Max results to log when verbose enabled

    # Tool registration settings
    allowed_tools: tuple[str, ...] | None = None

    # Static helper methods for parsing configuration sections
    # These methods improve testability and maintainability by extracting
    # the large from_environment method into focused, single-responsibility functions.

    @staticmethod
    def _parse_transport_config() -> TransportConfigDict:
        """Parse transport-related configuration from environment variables."""
        transport_raw = os.environ.get("MCP_TRANSPORT", "stdio")
        if transport_raw == "http":
            transport: TransportMode = "http"
        elif transport_raw == "sse":
            transport = "sse"
        elif transport_raw == "streamable-http":
            transport = "streamable-http"
        else:
            transport = "stdio"

        return {
            "transport": transport,
            "port": int(os.environ.get("PORT", os.environ.get("MCP_PORT", "9103"))),
            "host": os.environ.get("MCP_HOST", "127.0.0.1"),
            "path": os.environ.get("MCP_PATH", "/mcp"),
            "openrouter_base_url": os.environ.get(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
        }

    @staticmethod
    def _parse_embedding_config() -> EmbeddingConfigDict:
        """Parse embedding-related configuration from environment variables."""
        return {
            "embedder_provider": os.environ.get("EMBEDDER_PROVIDER", "openai"),
            "embedding_model": os.environ.get(
                "EMBEDDING_MODEL", "openai/text-embedding-3-large"
            ),
            "embedding_dims": int(os.environ.get("EMBEDDING_DIMS", "3072")),
            "qwen_embedding_dims": int(os.environ.get("QWEN_EMBEDDING_DIMS", "4096")),
            "embedding_batch_size": max(
                1, int(os.environ.get("EMBEDDING_BATCH_SIZE", "512"))
            ),
            "embedding_max_concurrent_batches": max(
                1, int(os.environ.get("EMBEDDING_MAX_CONCURRENT_BATCHES", "4"))
            ),
            "embedding_cache_enabled": os.environ.get(
                "EMBEDDING_CACHE_ENABLED", "true"
            ).lower()
            == "true",
            "embedding_cache_size": max(
                1, int(os.environ.get("EMBEDDING_CACHE_SIZE", "100"))
            ),
        }

    @staticmethod
    def _parse_search_config() -> SearchConfigDict:
        """Parse search-related configuration from environment variables."""
        return {
            "search_limit": int(os.environ.get("SEARCH_LIMIT", "5")),
            "remove_search_limit": int(os.environ.get("REMOVE_SEARCH_LIMIT", "5")),
            "enable_hybrid_search": os.environ.get(
                "ENABLE_HYBRID_SEARCH", "true"
            ).lower()
            == "true",
            "tantivy_index_path_template": "indexes/{workspace_id}/tantivy",
            "overfetch_multiplier": max(
                1, int(os.environ.get("OVERFETCH_MULTIPLIER", "3"))
            ),
            "overfetch_adaptive": os.environ.get("OVERFETCH_ADAPTIVE", "true").lower()
            == "true",
            "overfetch_min_multiplier": max(
                1.0, float(os.environ.get("OVERFETCH_MIN_MULTIPLIER", "1.5"))
            ),
            "overfetch_max_multiplier": max(
                1.0, float(os.environ.get("OVERFETCH_MAX_MULTIPLIER", "3.0"))
            ),
            "usearch_exact_search": os.environ.get(
                "USEARCH_EXACT_SEARCH", "false"
            ).lower()
            == "true",
            "usearch_exact_search_threshold": max(
                0, int(os.environ.get("USEARCH_EXACT_SEARCH_THRESHOLD", "256"))
            ),
            "fusion_method": os.environ.get("FUSION_METHOD", "rrf").lower(),
            "fusion_normalization": os.environ.get("FUSION_NORMALIZATION") or None,
            "fusion_rrf_k": int(os.environ.get("FUSION_RRF_K", "60")),
            "fusion_ranking_threshold": float(
                os.environ.get("FUSION_RANKING_THRESHOLD", "0.8")
            ),
            "enable_rrf_fusion": os.environ.get("ENABLE_RRF_FUSION", "true").lower()
            == "true",
            "fusion_weights": (
                [
                    float(w.strip())
                    for w in os.environ.get("FUSION_WEIGHTS", "").split(",")
                    if w.strip()
                ]
                or None
            ),
        }

    @staticmethod
    def _parse_tantivy_config() -> StorageConfigDict:
        """Parse Tantivy-specific configuration from environment variables."""
        return {
            "tantivy_soft_delete_enabled": os.environ.get(
                "TANTIVY_SOFT_DELETE_ENABLED", "true"
            ).lower()
            == "true",
            "tantivy_compaction_threshold_ratio": max(
                0.01,
                min(
                    1.0,
                    float(os.environ.get("TANTIVY_COMPACTION_THRESHOLD_RATIO", "0.2")),
                ),
            ),
            "tantivy_compaction_max_tombstones": max(
                100, int(os.environ.get("TANTIVY_COMPACTION_MAX_TOMBSTONES", "10000"))
            ),
            "tantivy_tombstone_ttl_days": max(
                0, int(os.environ.get("TANTIVY_TOMBSTONE_TTL_DAYS", "7"))
            ),
            "tantivy_normalize_scores": os.environ.get(
                "TANTIVY_NORMALIZE_SCORES", "true"
            ).lower()
            == "true",
        }

    @staticmethod
    def _parse_reranker_config() -> RerankerConfigDict:
        """Parse reranker-related configuration from environment variables."""
        # Determine reranker engine
        reranker_engine_raw = os.environ.get("RERANKER_ENGINE", "cross_encoder")
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

        return {
            "reranker_engine": reranker_engine,
            "llm_provider": llm_provider,
            "llm_model": os.environ.get("LLM_MODEL", "x-ai/grok-4.1-fast"),
            "search_score_threshold": float(
                os.environ.get("SEARCH_SCORE_THRESHOLD", "0.5")
            ),
            "rerank_max_concurrency": max(
                1, int(os.environ.get("RERANK_MAX_CONCURRENCY", "10"))
            ),
            "cross_encoder_model": os.environ.get(
                "CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3"
            ),
            "cross_encoder_top_k": int(os.environ.get("CROSS_ENCODER_TOP_K", "20")),
            "cross_encoder_device": os.environ.get("CROSS_ENCODER_DEVICE", "cpu"),
            "cross_encoder_batch_size": int(
                os.environ.get("CROSS_ENCODER_BATCH_SIZE", "32")
            ),
            "cross_encoder_score_threshold": float(
                os.environ.get("CROSS_ENCODER_SCORE_THRESHOLD", "0.5")
            ),
            "cross_encoder_use_fp16": os.environ.get(
                "CROSS_ENCODER_USE_FP16", "true"
            ).lower()
            == "true",
            "cross_encoder_normalize": os.environ.get(
                "CROSS_ENCODER_NORMALIZE", "true"
            ).lower()
            == "true",
            "cross_encoder_max_length": int(
                os.environ.get("CROSS_ENCODER_MAX_LENGTH", "512")
            ),
            "reranker_min_results": max(
                0, int(os.environ.get("RERANKER_MIN_RESULTS", "0"))
            ),
            "reranker_batch_normalize": os.environ.get(
                "RERANKER_BATCH_NORMALIZE", "true"
            ).lower()
            == "true",
            "enable_recency_boost": os.environ.get(
                "ENABLE_RECENCY_BOOST", "true"
            ).lower()
            == "true",
            "recency_decay_rate": max(
                0.0, float(os.environ.get("RECENCY_DECAY_RATE", "0.01"))
            ),
        }

    @staticmethod
    def _parse_memory_config() -> SecurityConfigDict:
        """Parse memory-related configuration from environment variables."""
        return {
            "max_memory_length": int(os.environ.get("MAX_MEMORY_LENGTH", "30720")),
            "min_memory_length": int(os.environ.get("MIN_MEMORY_LENGTH", "1")),
            "deduplicate_memories": os.environ.get(
                "DEDUPLICATE_MEMORIES", "true"
            ).lower()
            == "true",
        }

    @staticmethod
    def _parse_smart_replace_config() -> MetricsConfigDict:
        """Parse smart replacement configuration from environment variables."""
        return {
            "enable_smart_replace": os.environ.get(
                "ENABLE_SMART_REPLACE", "true"
            ).lower()
            == "true",
            "smart_replace_threshold": float(
                os.environ.get("SMART_REPLACE_THRESHOLD", "0.7")
            ),
            "smart_replace_min_similarity": float(
                os.environ.get("SMART_REPLACE_MIN_SIMILARITY", "0.9")
            ),
            "smart_replace_candidate_limit": max(
                1, int(os.environ.get("SMART_REPLACE_CANDIDATE_LIMIT", "3"))
            ),
            "smart_replace_archive_ttl_days": max(
                0, int(os.environ.get("SMART_REPLACE_ARCHIVE_TTL_DAYS", "30"))
            ),
            "smart_replace_max_retries": max(
                1, int(os.environ.get("SMART_REPLACE_MAX_RETRIES", "3"))
            ),
            "smart_replace_retry_delay": max(
                0.1, float(os.environ.get("SMART_REPLACE_RETRY_DELAY", "1.0"))
            ),
        }

    @staticmethod
    def _parse_init_config() -> ServerConfigDict:
        """Parse initialization-related configuration from environment variables."""
        return {
            "add_max_concurrency": max(
                1, int(os.environ.get("ADD_MAX_CONCURRENCY", "4"))
            ),
            "eager_initialization": os.environ.get(
                "EAGER_INITIALIZATION", "true"
            ).lower()
            == "true",
            "eager_initialize_search_engines": _parse_optional_bool(
                os.environ.get("EAGER_INITIALIZE_SEARCH_ENGINES")
            ),
            "eager_initialize_reranker": _parse_optional_bool(
                os.environ.get("EAGER_INITIALIZE_RERANKER")
            ),
            "eager_initialize_smart_replacer": _parse_optional_bool(
                os.environ.get("EAGER_INITIALIZE_SMART_REPLACER")
            ),
        }

    @staticmethod
    def _parse_logging_config() -> LoggingConfigDict:
        """Parse logging-related configuration from environment variables."""
        return {
            "log_level": os.environ.get("LOG_LEVEL", "INFO"),
            "log_search_results_verbose": os.environ.get(
                "LOG_SEARCH_RESULTS_VERBOSE", "false"
            ).lower()
            == "true",
            "log_search_result_limit": int(
                os.environ.get("LOG_SEARCH_RESULT_LIMIT", "3")
            ),
        }

    @staticmethod
    def _parse_allowed_tools() -> tuple[str, ...] | None:
        """Parse allowed tools configuration from environment variables."""

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
                    allowed_tools = ()
                else:
                    # Preserve order while removing duplicates
                    seen: dict[str, None] = {}
                    for token in normalized:
                        if token not in seen:
                            seen[token] = None
                    allowed_tools = tuple(seen.keys())

        # If ALLOWED_TOOLS was set to an empty/blank value, respect it
        if allowed_tools_env is not None and allowed_tools_env.strip() == "":
            allowed_tools = ()

        return allowed_tools

    @classmethod
    def from_environment(cls) -> Config:
        """Create configuration from environment variables.

        This method centralizes environment parsing and performs strict validation
        for all critical settings. It fails fast with clear, non-secret-leaking
        errors when configuration is invalid.

        Raises:
            ConfigurationError: If required environment variables are missing or invalid.
        """
        # Validate required environment variables
        workspace_id = os.environ.get("WORKSPACE_ID")
        if not workspace_id:
            raise ConfigurationError(
                "The WORKSPACE_ID environment variable has not been configured."
            )

        # Enforce a safe WORKSPACE_ID format to avoid path traversal and
        # filesystem issues. Keep the rule intentionally strict.
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", workspace_id):
            raise ConfigurationError(
                "Invalid WORKSPACE_ID: only A-Za-z0-9_.- allowed, max length 64."
            )

        # Check for path traversal patterns (not caught by regex)
        if ".." in workspace_id or workspace_id.startswith("/"):
            raise ConfigurationError(
                f"Invalid WORKSPACE_ID: path traversal patterns not allowed: {workspace_id}"
            )

        preset = get_active_preset()
        if preset:
            apply_preset_to_env(preset)

        openrouter_api_key_raw = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_api_key_raw:
            raise ConfigurationError(
                "The OPENROUTER_API_KEY environment variable has not been configured."
            )
        # Wrap in SecretString to prevent accidental logging
        openrouter_api_key = SecretString(openrouter_api_key_raw)

        # Parse configuration sections using helper methods
        transport_config = cls._parse_transport_config()
        embedding_config = cls._parse_embedding_config()
        search_config = cls._parse_search_config()
        tantivy_config = cls._parse_tantivy_config()
        reranker_config = cls._parse_reranker_config()
        memory_config = cls._parse_memory_config()
        smart_replace_config = cls._parse_smart_replace_config()
        init_config = cls._parse_init_config()
        logging_config = cls._parse_logging_config()
        allowed_tools = cls._parse_allowed_tools()

        # Create Config instance with all parsed settings
        config = cls(
            # Required settings
            workspace_id=workspace_id,
            openrouter_api_key=openrouter_api_key,
            # Parsed configuration sections
            **transport_config,
            **embedding_config,
            **search_config,
            **tantivy_config,
            **reranker_config,
            **memory_config,
            **smart_replace_config,
            **init_config,
            **logging_config,
            allowed_tools=allowed_tools,
        )

        return config


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


# Lazy config proxy for deferred initialization
# Note: This will trigger lazy init on attribute access, not import
class _LazyConfig:
    """Lazy proxy for deferred Config initialization.

    This proxy class forwards all attribute access to the lazily-initialized
    Config singleton, enabling imports like `from config import config` without
    triggering immediate environment variable parsing.
    """

    def __getattr__(self, name: str) -> object:
        return getattr(get_config(), name)

    def __repr__(self) -> str:
        return f"_LazyConfig(initialized={_config is not None})"


def setup_config_reload() -> Config:
    """Setup runtime configuration reload via SIGHUP.

    Must be called after Config singleton is initialized.
    """
    from ..utils.config_reload import setup_signal_handler

    config = Config.from_environment()
    _ = setup_signal_handler(lambda: Config.from_environment())

    return config


# Note: This is typed as Any to allow type checkers to accept it
# where Config is expected
# The actual Config object is accessed lazily through __getattr__
config: Config | _LazyConfig = _LazyConfig()
