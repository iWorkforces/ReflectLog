from dataclasses import dataclass
from typing import Any
import os
import random
import time
import warnings
import anyio
from openai import OpenAI, AsyncOpenAI, DefaultAioHttpClient, DefaultHttpxClient
from pydantic import BaseModel, ConfigDict, PrivateAttr


@dataclass
class EmbedderConfig:
    """Configuration for embedding provider.

    Provides the configuration fields needed by LangchainQwenEmbeddings
    for connecting to OpenRouter embedding APIs.
    """

    model: str = ""
    embedding_dims: int = 1536
    api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    timeout: float = 60.0  # 60 seconds default for embedding requests
    batch_size: Optional[int] = None
    max_concurrent_batches: Optional[int] = None


class LangchainQwenEmbeddings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: Any = None
    _client: OpenAI | None = PrivateAttr(default=None)
    _async_client: AsyncOpenAI | None = PrivateAttr(default=None)

    def __init__(self, config: Optional[EmbedderConfig | dict[str, Any]] = None):
        super().__init__(config=config or {})

        if isinstance(self.config, dict):
            self.config = EmbedderConfig(**self.config)

        self.config.embedding_dims = self.config.embedding_dims or 1536

        api_key = self.config.api_key or os.getenv("OPENROUTER_API_KEY")
        base_url = (
            self.config.openai_base_url
            or os.getenv("OPENROUTER_BASE_URL")
            or "https://openrouter.ai/api/v1"
        )
        self.config.api_key = api_key
        self.config.openai_base_url = base_url
        if os.environ.get("OPENAI_API_BASE"):
            warnings.warn(
                "The environment variable 'OPENAI_API_BASE' is deprecated and will be removed in the 0.1.80. "
                "Please use 'OPENROUTER_BASE_URL' instead.",
                DeprecationWarning,
            )

        # Initialize synchronous client for sync methods
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=DefaultHttpxClient(http2=True),
            timeout=self.config.timeout,
        )

        # Async client is initialized lazily to avoid unused aiohttp sessions

    def _get_async_client(self) -> AsyncOpenAI:
        """Get or initialize the async OpenAI client."""
        if self._async_client is None:
            # Note: aiohttp does not support HTTP/2, so we don't enable it
            self._async_client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.openai_base_url,
                http_client=DefaultAioHttpClient(),
                timeout=self.config.timeout,
            )
        return self._async_client

    def _sync_embed_with_retry(self, **kwargs: Any) -> list[list[float]]:
        """Call sync embeddings API with exponential backoff retry.

        Uses exponential backoff with jitter to avoid overwhelming the API
        during recovery from transient failures.

        Args:
            **kwargs: Arguments passed to embeddings.create()

        Returns:
            List of embedding vectors.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        max_attempts = 3
        base_delay = 1.0  # seconds
        last_exc: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                if self._client is None:
                    raise RuntimeError("Qwen embeddings client is not initialized.")
                response = self._client.embeddings.create(**kwargs)
                return [d.embedding for d in response.data]
            except Exception as exc:
                last_exc = exc
                if attempt < max_attempts:
                    # Exponential backoff: 1s, 2s, 4s, ...
                    delay = base_delay * (2 ** (attempt - 1))
                    # Add jitter (10% of delay) to prevent thundering herd
                    jitter = random.uniform(0, delay * 0.1)
                    time.sleep(delay + jitter)

        raise RuntimeError("Embedding request failed after retries") from last_exc

    def embed_query(self, text: str) -> list[float]:
        """Embed query text using synchronous client with retries.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        text = text.replace("\n", " ")
        embeddings = self._sync_embed_with_retry(
            input=[text],
            model=self.config.model,
            dimensions=self.config.embedding_dims,
            encoding_format="float",
        )
        return embeddings[0] if embeddings else []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs using synchronous client with retries and batching.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        if not texts:
            return []

        cleaned_texts = [text.replace("\n", " ") for text in texts]

        # Batch size configurable via config or env var (default 512, was 64)
        # OpenRouter/OpenAI APIs support up to 2048 items per request
        batch_size = self.config.batch_size or int(
            os.environ.get("EMBEDDING_BATCH_SIZE", "512")
        )
        batch_size = max(1, batch_size)
        results: list[list[float]] = []
        for i in range(0, len(cleaned_texts), batch_size):
            batch = cleaned_texts[i : i + batch_size]
            batch_embeddings = self._sync_embed_with_retry(
                input=batch,
                model=self.config.model,
                dimensions=self.config.embedding_dims,
                encoding_format="float",
            )
            results.extend(batch_embeddings)
        return results

    async def _async_embed_with_retry(self, **kwargs: Any) -> list[list[float]]:
        """Call async embeddings API with exponential backoff retry.

        Uses exponential backoff with jitter to avoid overwhelming the API
        during recovery from transient failures.

        Args:
            **kwargs: Arguments passed to embeddings.create()

        Returns:
            List of embedding vectors.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        max_attempts = 3
        base_delay = 1.0  # seconds
        last_exc: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                client = self._get_async_client()
                response = await client.embeddings.create(**kwargs)
                return [d.embedding for d in response.data]
            except Exception as exc:  # pragma: no cover - network/env dependent
                last_exc = exc
                if attempt < max_attempts:
                    # Exponential backoff: 1s, 2s, 4s, ...
                    delay = base_delay * (2 ** (attempt - 1))
                    # Add jitter (10% of delay) to prevent thundering herd
                    jitter = random.uniform(0, delay * 0.1)
                    await anyio.sleep(delay + jitter)

        raise RuntimeError("Async embedding request failed after retries") from last_exc

    async def aembed_query(self, text: str) -> list[float]:
        """Async version: Embed query text with retries."""
        text = text.replace("\n", " ")
        embeddings = await self._async_embed_with_retry(
            input=[text],
            model=self.config.model,
            dimensions=self.config.embedding_dims,
            encoding_format="float",
        )
        return embeddings[0] if embeddings else []

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Async version: Embed search docs with batching and concurrent execution.

        Batches texts for efficient API usage (batch_size texts per request),
        then processes batches concurrently with semaphore control.

        This is ~100x more efficient than per-text API calls:
        - 1000 texts with batch_size=512 = 2 API calls (parallel)
        - vs 1000 texts without batching = 1000 API calls

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings in the same order as input texts.
        """
        if not texts:
            return []

        cleaned_texts = [text.replace("\n", " ") for text in texts]

        # Batch size configurable via config or env var (default 512)
        # OpenRouter/OpenAI APIs support up to 2048 items per request
        batch_size = self.config.batch_size or int(
            os.environ.get("EMBEDDING_BATCH_SIZE", "512")
        )
        batch_size = max(1, batch_size)

        # Max concurrent batches configurable via config or env var (default 4)
        max_concurrent_batches = self.config.max_concurrent_batches or int(
            os.environ.get("EMBEDDING_MAX_CONCURRENT_BATCHES", "4")
        )
        max_concurrent_batches = max(1, max_concurrent_batches)

        # Split into batches
        batches: list[tuple[int, list[str]]] = []
        for i in range(0, len(cleaned_texts), batch_size):
            batch = cleaned_texts[i : i + batch_size]
            batches.append((i, batch))  # (start_index, batch_texts)

        # Initialize results array
        results: list[list[float]] = [[] for _ in texts]

        # Semaphore to limit concurrent batch requests
        semaphore = anyio.Semaphore(max_concurrent_batches)

        async def embed_batch(start_idx: int, batch_texts: list[str]) -> None:
            """Embed a batch of texts with semaphore control.

            Wraps the embedding call in try-except to continue on error
            instead of cancelling all batches when one fails.
            """
            try:
                async with semaphore:
                    batch_embeddings = await self._async_embed_with_retry(
                        input=batch_texts,
                        model=self.config.model,
                        dimensions=self.config.embedding_dims,
                        encoding_format="float",
                    )
                    # Assign results to correct indices
                    for j, embedding in enumerate(batch_embeddings):
                        results[start_idx + j] = embedding
            except Exception as e:
                # Log error but don't cancel other batches
                # Empty results remain empty for this batch
                # Caller can detect by checking for empty lists
                import warnings

                warnings.warn(
                    f"Embedding batch {start_idx} failed: {e}",
                    RuntimeWarning,
                )
                # Results at failed batch indices remain as empty lists

        # Process all batches concurrently (limited by semaphore)
        # Note: If one batch fails, others continue
        async with anyio.create_task_group() as tg:
            for start_idx, batch_texts in batches:
                tg.start_soon(embed_batch, start_idx, batch_texts)

        return results

    async def aclose(self) -> None:
        """Close the async client and cleanup resources.

        This method should be called when done with the embedder to properly
        close the async HTTP client and release resources.
        """
        if self._async_client is not None:
            await self._async_client.close()
            self._async_client = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.aclose()
