import asyncio
from dataclasses import replace
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from reflectlog.core.enums import WeMMDevice, WeMMModel
from reflectlog.infrastructure.embeddings.wemm_embedding import WeMMEmbeddingConfig
from reflectlog.infrastructure.embeddings.wemm_service import acquire_wemm


@pytest.fixture
def config() -> WeMMEmbeddingConfig:
    return WeMMEmbeddingConfig(WeMMModel.EMBEDDING_4B, 1024, WeMMDevice.CPU, 3)


def test_compatible_borrows_share_provider_and_final_close(
    config: WeMMEmbeddingConfig,
) -> None:
    with patch(
        "reflectlog.infrastructure.embeddings.wemm_service.WeMMEmbeddings"
    ) as provider_class:
        first = acquire_wemm(config)
        second = acquire_wemm(config)
        try:
            provider_class.assert_called_once_with(config)
            assert (
                first.embed_query("first")
                is provider_class.return_value.embed_query.return_value
            )
            first.close()
            first.close()
            provider_class.return_value.close.assert_not_called()
            assert (
                second.embed_documents(["second"])
                is provider_class.return_value.embed_documents.return_value
            )
        finally:
            first.close()
            second.close()
        provider_class.return_value.close.assert_called_once_with()
        with pytest.raises(RuntimeError, match="closed"):
            first.embed_query("closed")
        third = acquire_wemm(config)
        assert third is not first
        third.close()
        assert provider_class.call_count == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("device", WeMMDevice.MPS),
        ("model", WeMMModel.EMBEDDING_2B),
        ("batch_size", 4),
        ("dimensions", 512),
    ],
)
def test_incompatible_live_key_is_rejected(
    config: WeMMEmbeddingConfig, field: str, value: WeMMDevice | WeMMModel | int
) -> None:
    with patch(
        "reflectlog.infrastructure.embeddings.wemm_service.WeMMEmbeddings"
    ) as provider_class:
        first = acquire_wemm(config)
        try:
            with pytest.raises(ValueError, match="WeMM"):
                acquire_wemm(replace(config, **{field: value}))
            provider_class.assert_called_once()
        finally:
            first.close()
        next_owner = acquire_wemm(replace(config, **{field: value}))
        next_owner.close()


def test_failed_provider_construction_does_not_poison_service(
    config: WeMMEmbeddingConfig,
) -> None:
    with patch(
        "reflectlog.infrastructure.embeddings.wemm_service.WeMMEmbeddings"
    ) as provider_class:
        provider_class.side_effect = [RuntimeError("construction"), MagicMock()]
        with pytest.raises(RuntimeError, match="construction"):
            acquire_wemm(config)
        owner = acquire_wemm(config)
        owner.close()
        assert provider_class.call_count == 2


def test_failed_first_model_load_retries_on_same_borrow(
    config: WeMMEmbeddingConfig,
) -> None:
    model = MagicMock()
    model.encode_query.return_value = np.ones((1, config.dimensions), dtype=np.float32)
    with patch(
        "reflectlog.infrastructure.embeddings.wemm_embedding.SentenceTransformer",
        side_effect=[OSError("local"), OSError("remote"), model],
    ) as constructor:
        owner = acquire_wemm(config)
        try:
            with pytest.raises(RuntimeError, match="model load failed"):
                owner.embed_query("retry")
            assert len(owner.embed_query("retry")) == config.dimensions
            assert constructor.call_count == 3
        finally:
            owner.close()


def test_final_close_waits_for_inflight_inference(config: WeMMEmbeddingConfig) -> None:
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    provider = MagicMock()

    def infer(text: str) -> list[float]:
        _ = text
        entered.set()
        assert release.wait(timeout=5)
        return [1.0]

    provider.embed_query.side_effect = infer
    with patch(
        "reflectlog.infrastructure.embeddings.wemm_service.WeMMEmbeddings",
        return_value=provider,
    ):
        owner = acquire_wemm(config)
        worker = threading.Thread(target=lambda: owner.embed_query("busy"))
        closer = threading.Thread(target=lambda: (owner.close(), finished.set()))
        worker.start()
        try:
            assert entered.wait(timeout=5)
            closer.start()
            assert not finished.wait(timeout=0.05)
            provider.close.assert_not_called()
        finally:
            release.set()
            worker.join(timeout=5)
            closer.join(timeout=5)
        assert finished.is_set()
        provider.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_async_embeddings_use_owned_provider(config: WeMMEmbeddingConfig) -> None:
    with patch(
        "reflectlog.infrastructure.embeddings.wemm_service.WeMMEmbeddings"
    ) as provider_class:
        provider_class.return_value.embed_query.return_value = [1.0]
        provider_class.return_value.embed_documents.return_value = [[1.0]]
        owner = acquire_wemm(config)
        try:
            assert await owner.aembed_query("query") == [1.0]
            assert await owner.aembed_documents(["document"]) == [[1.0]]
        finally:
            await asyncio.to_thread(owner.close)
