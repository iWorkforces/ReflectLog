from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
from unittest.mock import patch

import httpx
import pytest

from reflectlog.infrastructure.openrouter_reranker import (
    OpenRouterReranker,
    OpenRouterRerankerConfig,
)
from reflectlog.utility.http import HttpClientFactory


@pytest.fixture
def reranker() -> OpenRouterReranker:
    return OpenRouterReranker(
        OpenRouterRerankerConfig(
            model="voyageai/rerank-2.5-lite",
            base_url="https://openrouter.ai/api/v1/",
            api_key="test-only-secret",
            enable_recency_boost=False,
            recency_decay_rate=0.001,
        )
    )


@pytest.mark.parametrize(
    "results",
    [
        [{"index": 0, "relevance_score": 0.9}],
        [
            {"index": 0, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.8},
        ],
        [
            {"index": 0, "relevance_score": 0.9},
            {"index": 2, "relevance_score": 0.8},
        ],
        [
            {"index": 0, "relevance_score": "NaN"},
            {"index": 1, "relevance_score": 0.8},
        ],
        [
            {"index": True, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.8},
        ],
        [
            {"index": 0, "relevance_score": float("inf")},
            {"index": 1, "relevance_score": 0.8},
        ],
    ],
)
async def test_rejects_incomplete_or_invalid_results(
    reranker: OpenRouterReranker, results: list[dict[str, object]]
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"results": results})
        )
    ) as client:
        with patch.object(
            HttpClientFactory, "get_async_httpx_client", return_value=client
        ):
            with pytest.raises(ValueError):
                await reranker.rerank_async("query", [("first", 0.1), ("second", 0.2)])


async def test_posts_indexed_documents_and_ranks_after_decay(
    reranker: OpenRouterReranker,
) -> None:
    captured: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "id": "rerank-1",
                "model": "voyageai/rerank-2.5-lite",
                "results": [
                    {
                        "index": 1,
                        "relevance_score": 0.8,
                        "document": {"text": "second"},
                    },
                    {"index": 0, "relevance_score": 1, "document": {"text": "first"}},
                ],
                "usage": {"total_tokens": 10},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        with patch.object(
            HttpClientFactory, "get_async_httpx_client", return_value=client
        ):
            ranked = await reranker.rerank_async(
                "query", [("first", 0.01), ("second", 0.02)], top_k=2
            )
    assert ranked == [("first", 1.0), ("second", 0.8)]
    assert str(captured[0].url) == "https://openrouter.ai/api/v1/rerank"
    assert captured[0].headers["Authorization"] == "Bearer test-only-secret"
    assert json.loads(captured[0].content) == {
        "model": "voyageai/rerank-2.5-lite",
        "query": "query",
        "documents": ["first", "second"],
        "top_n": 2,
    }


async def test_recency_applied_after_scores_before_top_k(
    reranker: OpenRouterReranker,
) -> None:
    reranker = OpenRouterReranker(
        replace(reranker.config, enable_recency_boost=True, recency_decay_rate=0.1)
    )
    now = datetime.now(UTC)
    stamps = {
        "old": (now - timedelta(hours=100)).isoformat(),
        "new": now.isoformat(),
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={
                    "results": [
                        {"index": 0, "relevance_score": 0.9},
                        {"index": 1, "relevance_score": 0.8},
                    ]
                },
            )
        )
    ) as client:
        with patch.object(
            HttpClientFactory, "get_async_httpx_client", return_value=client
        ):
            ranked = await reranker.rerank_async(
                "query", [("old", 0.1), ("new", 0.2)], stamps, top_k=1
            )
    assert len(ranked) == 1
    assert ranked[0][0] == "new"
