from dataclasses import dataclass
from datetime import datetime
import math
from typing import TYPE_CHECKING

from pydantic import BaseModel, StrictFloat, StrictInt

from reflectlog.infrastructure.reranker_post_processor import RerankerPostProcessor
from reflectlog.utility.http import HttpClientFactory

if TYPE_CHECKING:
    from reflectlog.core.config import IAppConfig


@dataclass(frozen=True)
class OpenRouterRerankerConfig:
    model: str
    base_url: str
    api_key: str
    enable_recency_boost: bool
    recency_decay_rate: float

    @classmethod
    def from_config(cls, config: IAppConfig) -> OpenRouterRerankerConfig:
        return cls(
            model=config.openrouter_rerank_model,
            base_url=config.openrouter_base_url,
            api_key=config.openrouter_api_key,
            enable_recency_boost=config.enable_recency_boost,
            recency_decay_rate=config.recency_decay_rate,
        )


class _RerankResult(BaseModel):
    index: StrictInt
    relevance_score: StrictInt | StrictFloat


class _RerankResponse(BaseModel):
    results: list[_RerankResult]


class OpenRouterReranker:
    def __init__(self, config: OpenRouterRerankerConfig) -> None:
        self.config = config
        self._post_processor = RerankerPostProcessor(
            min_results=0, batch_normalize=False
        )

    async def rerank_async(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        timestamp_map: dict[str, str] | None = None,
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []
        response = await HttpClientFactory.get_async_httpx_client().post(
            f"{self.config.base_url.rstrip('/')}/rerank",
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            json={
                "model": self.config.model,
                "query": query,
                "documents": [document for document, _ in candidates],
                "top_n": len(candidates),
            },
        )
        _ = response.raise_for_status()
        results = _RerankResponse.model_validate(response.json()).results
        indices = [result.index for result in results]
        if (
            len(indices) != len(candidates)
            or set(indices) != set(range(len(candidates)))
            or any(not math.isfinite(result.relevance_score) for result in results)
        ):
            raise ValueError("Invalid OpenRouter rerank results")

        scored = [
            (candidates[result.index][0], float(result.relevance_score))
            for result in results
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        decay_enabled = (
            self.config.enable_recency_boost
            and self.config.recency_decay_rate > 0
            and timestamp_map is not None
            and all(
                _valid_timestamp(timestamp_map.get(document)) for document, _ in scored
            )
        )
        scored = self._post_processor.apply_decay(
            scored,
            timestamp_map,
            self.config.recency_decay_rate,
            enabled=decay_enabled,
        )
        return scored if top_k is None else scored[:top_k]


def _valid_timestamp(value: str | None) -> bool:
    if not value:
        return False
    try:
        _ = datetime.fromisoformat(value)
    except ValueError:
        return False
    return True
