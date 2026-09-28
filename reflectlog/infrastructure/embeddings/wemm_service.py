from threading import RLock
from typing import final

from asyncer import asyncify

from reflectlog.infrastructure.embeddings.wemm_embedding import (
    WeMMEmbeddingConfig,
    WeMMEmbeddings,
)


class _WeMMService:
    def __init__(self) -> None:
        self._lock = RLock()
        self._config: WeMMEmbeddingConfig | None = None
        self._provider: WeMMEmbeddings | None = None
        self._owners = 0

    def acquire(self, config: WeMMEmbeddingConfig) -> WeMMBorrow:
        with self._lock:
            if self._config is not None and self._config != config:
                raise ValueError("Incompatible concurrent WeMM configuration")
            if self._provider is None:
                self._provider = WeMMEmbeddings(config)
                self._config = config
            self._owners += 1
            return WeMMBorrow(self, self._provider)

    def release(self) -> None:
        with self._lock:
            self._owners -= 1
            if self._owners == 0:
                provider = self._provider
                try:
                    if provider is not None:
                        provider.close()
                finally:
                    self._provider = None
                    self._config = None


@final
class WeMMBorrow:
    def __init__(self, service: _WeMMService, provider: WeMMEmbeddings) -> None:
        self._service = service
        self._provider = provider
        self._lock = RLock()
        self._closed = False

    @property
    def config(self) -> WeMMEmbeddingConfig:
        return self._provider.config

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            if self._closed:
                raise RuntimeError("WeMM borrow is closed")
            return self._provider.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            if self._closed:
                raise RuntimeError("WeMM borrow is closed")
            return self._provider.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncify(self.embed_query)(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncify(self.embed_documents)(texts)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._service.release()


_SERVICE = _WeMMService()


def acquire_wemm(config: WeMMEmbeddingConfig) -> WeMMBorrow:
    return _SERVICE.acquire(config)
