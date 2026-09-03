"""Unit tests for CrossEncoderReranker (using FlagReranker)."""

import sys
from typing import Any, Generator, cast
from unittest.mock import MagicMock, patch

import pytest

from reflectlog.application.utils.logging import StructuredLogger
from reflectlog.core.logging import IStructuredLogger
from reflectlog.infrastructure.cross_encoder_reranker import (
    CrossEncoderConfig,
    CrossEncoderReranker,
    FlagRerankerProtocol,
)


def create_mock_logger() -> tuple[IStructuredLogger, MagicMock]:
    """Create a properly typed mock logger and its mock reference for testing."""
    mock = MagicMock(spec=StructuredLogger)
    return cast(IStructuredLogger, mock), mock


@pytest.fixture(autouse=True)
def mock_flag_embedding_module() -> Generator[None, Any, None]:
    """Mock FlagEmbedding module in sys.modules to avoid heavy imports."""
    mock_flag_embedding = MagicMock()
    mock_flag_reranker_class = MagicMock()
    mock_flag_embedding.FlagReranker = mock_flag_reranker_class
    sys.modules["FlagEmbedding"] = mock_flag_embedding
    yield
    sys.modules.pop("FlagEmbedding", None)


class TestCrossEncoderConfig:
    """Test CrossEncoderConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = CrossEncoderConfig()

        assert config.model_name == "BAAI/bge-reranker-v2-m3"
        assert config.enabled is True
        assert config.top_k == 20
        assert config.device == "cpu"
        assert config.batch_size == 32
        assert config.score_threshold == 0.5
        # FlagReranker-specific defaults
        assert config.use_fp16 is True
        assert config.normalize is True
        assert config.max_length == 512

    def test_custom_values(self) -> None:
        """Test configuration with custom values."""
        config = CrossEncoderConfig(
            model_name="BAAI/bge-reranker-large",
            enabled=True,
            top_k=10,
            device="cuda",
            batch_size=16,
            score_threshold=0.5,
            use_fp16=False,
            normalize=False,
            max_length=256,
        )

        assert config.model_name == "BAAI/bge-reranker-large"
        assert config.enabled is True
        assert config.top_k == 10
        assert config.device == "cuda"
        assert config.batch_size == 16
        assert config.score_threshold == 0.5
        assert config.use_fp16 is False
        assert config.normalize is False
        assert config.max_length == 256

    def test_default_batch_normalize_and_min_results(self) -> None:
        """Test default values for batch_normalize and min_results."""
        config = CrossEncoderConfig()

        assert config.batch_normalize is False
        assert config.min_results == 1

    def test_from_app_config_enabled(self) -> None:
        """Test factory method from application config when enabled."""
        mock_app_config = MagicMock()
        mock_app_config.reranker_engine = "cross_encoder"
        mock_app_config.cross_encoder_model = "BAAI/bge-reranker-v2-m3"
        mock_app_config.cross_encoder_top_k = 15
        mock_app_config.cross_encoder_device = "mps"
        mock_app_config.cross_encoder_batch_size = 64
        mock_app_config.cross_encoder_score_threshold = 0.3
        mock_app_config.cross_encoder_use_fp16 = True
        mock_app_config.cross_encoder_normalize = True
        mock_app_config.cross_encoder_max_length = 512
        mock_app_config.reranker_min_results = 2
        mock_app_config.reranker_batch_normalize = True

        config = CrossEncoderConfig.from_config(mock_app_config)

        assert config.enabled is True
        assert config.model_name == "BAAI/bge-reranker-v2-m3"
        assert config.top_k == 15
        assert config.device == "mps"
        assert config.batch_size == 64
        assert config.score_threshold == 0.3
        assert config.use_fp16 is True
        assert config.normalize is True
        assert config.max_length == 512
        assert config.min_results == 2
        assert config.batch_normalize is False

    def test_from_app_config_disabled(self) -> None:
        """Test factory method from application config when disabled."""
        mock_app_config = MagicMock()
        mock_app_config.reranker_engine = "none"  # Not cross_encoder
        mock_app_config.cross_encoder_model = "BAAI/bge-reranker-v2-m3"
        mock_app_config.cross_encoder_top_k = 20
        mock_app_config.cross_encoder_device = "cpu"
        mock_app_config.cross_encoder_batch_size = 32
        mock_app_config.cross_encoder_score_threshold = 0.0
        mock_app_config.cross_encoder_use_fp16 = True
        mock_app_config.cross_encoder_normalize = True
        mock_app_config.cross_encoder_max_length = 512
        mock_app_config.reranker_min_results = 0
        mock_app_config.reranker_batch_normalize = True

        config = CrossEncoderConfig.from_config(mock_app_config)

        assert config.enabled is False


class TestCrossEncoderRerankerInitialization:
    """Test CrossEncoderReranker initialization."""

    def test_initialization_does_not_load_model(self) -> None:
        """Test that initialization does not load the model (lazy loading)."""
        config = CrossEncoderConfig()

        with patch("FlagEmbedding.FlagReranker") as mock_flag_reranker:
            reranker = CrossEncoderReranker(config=config)

            # Model should not be loaded during initialization
            mock_flag_reranker.assert_not_called()
            assert reranker._model is None

    def test_model_property_lazy_loads(self) -> None:
        """Test that accessing model property triggers lazy loading."""
        config = CrossEncoderConfig(
            model_name="test-model", device="cpu", use_fp16=True
        )

        with patch("FlagEmbedding.FlagReranker") as mock_flag_reranker:
            mock_model = MagicMock()
            mock_flag_reranker.return_value = mock_model

            reranker = CrossEncoderReranker(config=config)

            # Access model property
            model = reranker.model

            # Model should now be loaded with FlagReranker API
            mock_flag_reranker.assert_called_once_with(
                "test-model", use_fp16=True, devices=["cpu"]
            )
            assert model is mock_model

    def test_model_property_caches_instance(self) -> None:
        """Test that model property returns cached instance on subsequent calls."""
        config = CrossEncoderConfig()

        with patch("FlagEmbedding.FlagReranker") as mock_flag_reranker:
            mock_model = MagicMock()
            mock_flag_reranker.return_value = mock_model

            reranker = CrossEncoderReranker(config=config)

            # Access model property multiple times
            model1 = reranker.model
            model2 = reranker.model
            model3 = reranker.model

            # Model should only be loaded once
            mock_flag_reranker.assert_called_once()
            assert model1 is model2 is model3


class TestRerank:
    """Test rerank method."""

    @pytest.fixture
    def mock_reranker(self) -> CrossEncoderReranker:
        """Create a CrossEncoderReranker with mocked FlagReranker model."""
        config = CrossEncoderConfig(
            enabled=True,
            top_k=10,
            batch_size=32,
            score_threshold=0.0,
            normalize=True,
            max_length=512,
            batch_normalize=False,  # Disable batch normalization for raw score tests
        )

        with patch("FlagEmbedding.FlagReranker"):
            reranker = CrossEncoderReranker(config=config)
            # Pre-set the model to avoid lazy loading
            reranker._model = MagicMock()

            # Assert to narrow the type from BaseReranker | None to MagicMock

            assert reranker._model is not None
            # Assert to narrow the type from BaseReranker | None to MagicMock
            assert reranker._model is not None
            return reranker

    def test_rerank_empty_candidates(self, mock_reranker: CrossEncoderReranker) -> None:
        """Test reranking empty candidate list."""
        result = mock_reranker.rerank("test query", [])
        assert result == []

    def test_rerank_disabled_returns_candidates_unchanged(self) -> None:
        """Test that disabled reranker returns candidates unchanged."""
        config = CrossEncoderConfig(enabled=False)

        with patch("FlagEmbedding.FlagReranker"):
            reranker = CrossEncoderReranker(config=config)

            candidates = [("doc1", 0.8), ("doc2", 0.6)]
            result = reranker.rerank("test query", candidates)

            # Should return original candidates unchanged
            assert result == candidates

    def test_rerank_scores_and_sorts(self, mock_reranker: CrossEncoderReranker) -> None:
        """Test that rerank scores candidates and sorts by score descending."""
        # Mock model.compute_score to return scores (FlagReranker API)
        mock_reranker._model.compute_score.return_value = [0.3, 0.9, 0.6]  # type: ignore

        candidates = [
            ("doc1", 0.8),
            ("doc2", 0.7),
            ("doc3", 0.6),
        ]

        result = mock_reranker.rerank("test query", candidates)

        # Should be sorted by cross-encoder score descending
        assert len(result) == 3
        assert result[0] == ("doc2", 0.9)
        assert result[1] == ("doc3", 0.6)
        assert result[2] == ("doc1", 0.3)

    def test_rerank_respects_top_k(self) -> None:
        """Test that rerank limits results to top_k."""
        config = CrossEncoderConfig(enabled=True, top_k=2, batch_normalize=False)

        with patch("FlagEmbedding.FlagReranker"):
            reranker = CrossEncoderReranker(config=config)
            reranker._model = MagicMock()

            # Assert to narrow the type from BaseReranker | None to MagicMock

            assert reranker._model is not None
            reranker._model.compute_score.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]

            candidates = [(f"doc{i}", 0.5) for i in range(5)]
            result = reranker.rerank("test query", candidates)

            # Should only return top 2
            assert len(result) == 2

    def test_rerank_filters_by_score_threshold(self) -> None:
        """Test that results below score threshold are filtered out."""
        config = CrossEncoderConfig(
            enabled=True, top_k=10, score_threshold=0.5, batch_normalize=False
        )

        with patch("FlagEmbedding.FlagReranker"):
            reranker = CrossEncoderReranker(config=config)
            reranker._model = MagicMock()

            # Assert to narrow the type from BaseReranker | None to MagicMock

            assert reranker._model is not None
            # Scores: 0.9, 0.4, 0.6 - only 0.9 and 0.6 pass threshold
            reranker._model.compute_score.return_value = [0.9, 0.4, 0.6]

            candidates = [("doc1", 0.8), ("doc2", 0.7), ("doc3", 0.6)]
            result = reranker.rerank("test query", candidates)

            # Only doc1 (0.9) and doc3 (0.6) should pass threshold
            assert len(result) == 2
            assert result[0] == ("doc1", 0.9)
            assert result[1] == ("doc3", 0.6)

    def test_rerank_builds_correct_pairs(
        self, mock_reranker: CrossEncoderReranker
    ) -> None:
        """Test that query-document pairs are built correctly for scoring."""

        class RecordingReranker:
            def __init__(self) -> None:
                self.pairs: list[tuple[str, str]] = []

            def compute_score(
                self,
                sentence_pairs: list[tuple[str, str]],
                *,
                batch_size: int,
                max_length: int,
                normalize: bool,
            ) -> list[float]:
                self.pairs = sentence_pairs
                return [0.8, 0.7]

        model = RecordingReranker()
        mock_reranker._model = model

        candidates = [("Python guide", 0.5), ("JavaScript guide", 0.5)]
        _ = mock_reranker.rerank("Python tutorials", candidates)

        assert model.pairs == [
            ("Python tutorials", "Python guide"),
            ("Python tutorials", "JavaScript guide"),
        ]

    def test_rerank_uses_batch_size_and_max_length(self) -> None:
        """Test that compute_score is called with configured batch_size and max_length."""
        config = CrossEncoderConfig(
            enabled=True,
            batch_size=16,
            max_length=256,
            normalize=True,
            batch_normalize=False,
        )

        with patch("FlagEmbedding.FlagReranker"):
            reranker = CrossEncoderReranker(config=config)
            reranker._model = MagicMock()

            # Assert to narrow the type from BaseReranker | None to MagicMock

            assert reranker._model is not None
            reranker._model.compute_score.return_value = [0.8]

            candidates = [("doc1", 0.5)]
            reranker.rerank("query", candidates)

            # Verify FlagReranker API parameters were passed
            call_kwargs = reranker._model.compute_score.call_args[1]
            assert call_kwargs["batch_size"] == 16
            assert call_kwargs["max_length"] == 256
            assert call_kwargs["normalize"] is True

    def test_rerank_with_logger_logs_debug(self) -> None:
        """Test that reranking logs debug information when logger is provided."""
        config = CrossEncoderConfig(enabled=True, batch_normalize=False)

        with patch("FlagEmbedding.FlagReranker"):
            logger, mock_logger = create_mock_logger()
            reranker = CrossEncoderReranker(config=config, logger=logger)
            reranker._model = MagicMock()

            # Assert to narrow the type from BaseReranker | None to MagicMock

            assert reranker._model is not None
            reranker._model.compute_score.return_value = [0.8, 0.7]

            candidates = [("doc1", 0.5), ("doc2", 0.5)]
            reranker.rerank("query", candidates)

            # Should log info messages about scoring (verbose logging)
            mock_logger.info.assert_called()

    def test_rerank_handles_single_pair_float_return(self) -> None:
        """Test that rerank handles single-pair case where compute_score returns float."""
        config = CrossEncoderConfig(
            enabled=True, top_k=10, score_threshold=0.0, batch_normalize=False
        )

        with patch("FlagEmbedding.FlagReranker"):
            reranker = CrossEncoderReranker(config=config)
            reranker._model = MagicMock()

            # Assert to narrow the type from BaseReranker | None to MagicMock

            assert reranker._model is not None
            # FlagReranker returns float for single pair, not list
            reranker._model.compute_score.return_value = 0.85

            candidates = [("single doc", 0.5)]
            result = reranker.rerank("query", candidates)

            # Should handle single float correctly
            assert len(result) == 1
            assert result[0] == ("single doc", 0.85)


class TestRerankAsync:
    """Test rerank_async method."""

    @pytest.fixture
    def mock_reranker(self) -> CrossEncoderReranker:
        """Create a CrossEncoderReranker with mocked FlagReranker model."""
        config = CrossEncoderConfig(enabled=True, batch_normalize=False)

        with patch("FlagEmbedding.FlagReranker"):
            reranker = CrossEncoderReranker(config=config)
            reranker._model = MagicMock()

            # Assert to narrow the type from BaseReranker | None to MagicMock

            assert reranker._model is not None
            return reranker

    @pytest.mark.asyncio
    async def test_rerank_async_calls_sync_rerank(
        self, mock_reranker: CrossEncoderReranker
    ) -> None:
        """Test that rerank_async wraps the sync rerank method."""
        mock_reranker._model.compute_score.return_value = [0.9, 0.7]  # type: ignore

        candidates = [("doc1", 0.8), ("doc2", 0.6)]
        result = await mock_reranker.rerank_async("test query", candidates)

        # Should return same results as sync version
        assert len(result) == 2
        assert result[0] == ("doc1", 0.9)
        assert result[1] == ("doc2", 0.7)

    @pytest.mark.asyncio
    async def test_rerank_async_empty_candidates(
        self, mock_reranker: CrossEncoderReranker
    ) -> None:
        """Test rerank_async with empty candidate list."""
        result = await mock_reranker.rerank_async("test query", [])
        assert result == []


class TestCrossEncoderRerankerIntegration:
    """Integration-style tests for full reranking flow."""

    def test_full_reranking_flow(self) -> None:
        """Test complete reranking flow with mocked FlagReranker model."""
        config = CrossEncoderConfig(
            enabled=True,
            top_k=5,
            score_threshold=0.3,
            batch_normalize=False,  # Test raw score behavior
        )

        with patch("FlagEmbedding.FlagReranker"):
            logger, _ = create_mock_logger()
            reranker = CrossEncoderReranker(config=config, logger=logger)
            reranker._model = MagicMock()

            # Assert to narrow the type from BaseReranker | None to MagicMock

            assert reranker._model is not None

            # Simulate realistic cross-encoder scores
            reranker._model.compute_score.return_value = [
                0.95,
                0.25,
                0.78,
                0.15,
                0.65,
            ]

            candidates = [
                ("Python programming guide", 0.8),
                ("JavaScript basics", 0.75),
                ("Data science with Python", 0.7),
                ("Web development intro", 0.65),
                ("Machine learning tutorial", 0.6),
            ]

            result = reranker.rerank("Python tutorials", candidates)

            # Should filter out scores < 0.3 and sort descending
            # Passed: 0.95, 0.78, 0.65 (indices 0, 2, 4)
            # Filtered: 0.25, 0.15 (indices 1, 3)
            assert len(result) == 3
            assert result[0] == ("Python programming guide", 0.95)
            assert result[1] == ("Data science with Python", 0.78)
            assert result[2] == ("Machine learning tutorial", 0.65)

    def test_disabled_reranker_passthrough(self) -> None:
        """Test that disabled reranker passes through candidates unchanged."""
        config = CrossEncoderConfig(enabled=False)

        with patch("FlagEmbedding.FlagReranker") as mock_flag_reranker:
            reranker = CrossEncoderReranker(config=config)

            candidates = [
                ("doc1", 0.8),
                ("doc2", 0.7),
                ("doc3", 0.6),
            ]

            result = reranker.rerank("query", candidates)

            # Model should not be accessed
            mock_flag_reranker.assert_not_called()

            # Candidates should be returned unchanged
            assert result == candidates


class TestThreadSafety:
    """Test thread-safety of FlagReranker model initialization."""

    def test_double_checked_locking_pattern(self) -> None:
        """Test that model initialization uses double-checked locking."""
        config = CrossEncoderConfig()

        with patch("FlagEmbedding.FlagReranker") as mock_flag_reranker:
            mock_model = MagicMock()
            mock_flag_reranker.return_value = mock_model

            reranker = CrossEncoderReranker(config=config)

            # Simulate concurrent access
            import threading

            results: list[FlagRerankerProtocol] = []

            def access_model() -> None:
                model = reranker.model
                results.append(model)

            threads = [threading.Thread(target=access_model) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All threads should get the same model instance
            assert len(results) == 10
            assert all(r is mock_model for r in results)

            # Model should only be loaded once
            mock_flag_reranker.assert_called_once()


class TestBatchNormalization:
    """Test batch normalization feature."""

    def test_batch_normalize_enabled_normalizes_scores(self) -> None:
        """Test that batch_normalize=True normalizes scores to 0-1 range."""
        config = CrossEncoderConfig(
            enabled=True,
            top_k=10,
            score_threshold=0.0,  # No threshold filtering
            normalize=False,
            batch_normalize=True,
        )

        with patch("FlagEmbedding.FlagReranker"):
            reranker = CrossEncoderReranker(config=config)
            reranker._model = MagicMock()

            # Assert to narrow the type from BaseReranker | None to MagicMock

            assert reranker._model is not None
            # CrossEncoder typical low scores (0.001-0.17)
            reranker._model.compute_score.return_value = [0.17, 0.05, 0.001]

            candidates = [("doc1", 0.8), ("doc2", 0.7), ("doc3", 0.6)]
            result = reranker.rerank("query", candidates)

            # After normalization: best=1.0, worst=0.0
            assert len(result) == 3
            # Best score (0.17) should become 1.0
            assert result[0][0] == "doc1"
            assert result[0][1] == pytest.approx(1.0)
            # Worst score (0.001) should become 0.0
            assert result[2][0] == "doc3"
            assert result[2][1] == pytest.approx(0.0)

    def test_batch_normalize_disabled_uses_raw_scores(self) -> None:
        """Test that batch_normalize=False uses raw scores."""
        config = CrossEncoderConfig(
            enabled=True,
            top_k=10,
            score_threshold=0.0,
            batch_normalize=False,
        )

        with patch("FlagEmbedding.FlagReranker"):
            reranker = CrossEncoderReranker(config=config)
            reranker._model = MagicMock()

            # Assert to narrow the type from BaseReranker | None to MagicMock

            assert reranker._model is not None
            reranker._model.compute_score.return_value = [0.17, 0.05, 0.001]

            candidates = [("doc1", 0.8), ("doc2", 0.7), ("doc3", 0.6)]
            result = reranker.rerank("query", candidates)

            # Scores should remain as raw values
            assert len(result) == 3
            assert result[0][0] == "doc1"
            assert result[0][1] == 0.17  # Raw score preserved
            assert result[1][0] == "doc2"
            assert result[1][1] == 0.05

    def test_batch_normalize_with_threshold_filters_correctly(self) -> None:
        """Test batch normalization with threshold filtering."""
        config = CrossEncoderConfig(
            enabled=True,
            top_k=10,
            score_threshold=0.5,  # 50% threshold
            normalize=False,
            batch_normalize=True,
            min_results=0,
        )

        with patch("FlagEmbedding.FlagReranker"):
            reranker = CrossEncoderReranker(config=config)
            reranker._model = MagicMock()

            # Assert to narrow the type from BaseReranker | None to MagicMock

            assert reranker._model is not None
            # After normalization: 0.17->1.0, 0.05->~0.29, 0.001->0.0
            reranker._model.compute_score.return_value = [0.17, 0.05, 0.001]

            candidates = [("doc1", 0.8), ("doc2", 0.7), ("doc3", 0.6)]
            result = reranker.rerank("query", candidates)

            # Only doc1 (1.0) passes threshold 0.5
            assert len(result) == 1
            assert result[0][0] == "doc1"


class TestModelPropertyWithLogger:
    """Test model property logging behavior during initialization."""

    def test_model_property_logs_loading_and_success(self) -> None:
        """Test that model property logs loading start and success when logger is provided."""
        config = CrossEncoderConfig(
            model_name="test-model", device="cpu", use_fp16=True
        )

        with patch("FlagEmbedding.FlagReranker") as mock_flag_reranker:
            mock_model = MagicMock()
            mock_flag_reranker.return_value = mock_model

            logger, mock_logger = create_mock_logger()
            reranker = CrossEncoderReranker(config=config, logger=logger)

            # Access model property to trigger lazy loading
            _ = reranker.model

            # Should log loading start (line 140)
            loading_call = mock_logger.info.call_args_list[0]
            assert "Loading FlagReranker model" in loading_call[0][0]
            assert loading_call[1]["extra"]["device"] == "cpu"
            assert loading_call[1]["extra"]["use_fp16"] is True

            # Should log success (line 174)
            success_call = mock_logger.info.call_args_list[1]
            assert "loaded successfully" in success_call[0][0]
            assert success_call[1]["extra"]["model"] == "test-model"


class TestSuppressFastTokenizerWarning:
    """Test _suppress_fast_tokenizer_warning method."""

    def test_installs_warnings_filter(self) -> None:
        """Suppression uses a warnings filter, not tokenizer attribute probes."""
        config = CrossEncoderConfig()

        with patch("FlagEmbedding.FlagReranker"):
            reranker = CrossEncoderReranker(config=config)
            reranker._model = MagicMock()

            with patch(
                "reflectlog.infrastructure.cross_encoder_reranker.warnings"
            ) as mock_warnings:
                reranker._suppress_fast_tokenizer_warning()

                mock_warnings.filterwarnings.assert_called_once_with(
                    "ignore",
                    message=r"You're using a \w+TokenizerFast tokenizer.*using the `__call__` method is faster",
                    category=UserWarning,
                    module=r"transformers\.tokenization_utils_base",
                )


class TestRerankDisabledWithLogger:
    """Test disabled reranker with logger present."""

    def test_disabled_reranker_logs_debug_with_logger(self) -> None:
        """Test that disabled reranker logs debug message when logger is provided (line 302)."""
        config = CrossEncoderConfig(enabled=False)

        with patch("FlagEmbedding.FlagReranker"):
            logger, mock_logger = create_mock_logger()
            reranker = CrossEncoderReranker(config=config, logger=logger)

            candidates = [("doc1", 0.8), ("doc2", 0.6)]
            result = reranker.rerank("test query", candidates)

            # Should return unchanged candidates
            assert result == candidates

            # Should log debug about being disabled
            mock_logger.debug.assert_called_once()
            call_args = mock_logger.debug.call_args
            assert "disabled" in call_args[0][0].lower()
            assert call_args[1]["extra"]["candidate_count"] == 2


class TestBatchNormalizationWithLogger:
    """Test batch normalization logging."""

    def test_batch_normalize_logs_range_info(self) -> None:
        """Test that batch normalization logs raw and normalized ranges (line 338)."""
        config = CrossEncoderConfig(
            enabled=True,
            top_k=10,
            score_threshold=0.0,
            normalize=False,
            batch_normalize=True,
        )

        with patch("FlagEmbedding.FlagReranker"):
            logger, mock_logger = create_mock_logger()
            reranker = CrossEncoderReranker(config=config, logger=logger)
            mock_model = MagicMock()
            reranker._model = mock_model
            assert reranker._model is not None

            mock_model.compute_score.return_value = [0.17, 0.05, 0.001]

            candidates = [("doc1", 0.8), ("doc2", 0.7), ("doc3", 0.6)]
            reranker.rerank("query", candidates)

            # Find the batch normalization log call
            info_calls = mock_logger.info.call_args_list
            batch_norm_calls = [
                c for c in info_calls if "Batch normalization" in c[0][0]
            ]
            assert len(batch_norm_calls) == 1
            batch_call = batch_norm_calls[0]
            assert batch_call[1]["extra"]["batch_normalize"] is True
            assert batch_call[1]["extra"]["raw_min"] == pytest.approx(0.001)
            assert batch_call[1]["extra"]["raw_max"] == pytest.approx(0.17)


class TestRecencyDecay:
    """Test recency decay integration in rerank."""

    def test_recency_decay_applied_with_timestamp_map(self) -> None:
        """Test recency decay is applied when enabled and timestamp_map provided (lines 357-368)."""
        config = CrossEncoderConfig(
            enabled=True,
            top_k=10,
            score_threshold=0.0,
            batch_normalize=False,
            enable_recency_boost=True,
            recency_decay_rate=0.01,
        )

        with patch("FlagEmbedding.FlagReranker"):
            reranker = CrossEncoderReranker(config=config)
            reranker._model = MagicMock()
            assert reranker._model is not None
            reranker._model.compute_score.return_value = [0.9, 0.8]

            candidates = [("doc1", 0.5), ("doc2", 0.5)]
            timestamp_map = {
                "doc1": "2020-01-01T00:00:00+00:00",
                "doc2": "2026-02-13T00:00:00+00:00",
            }

            result = reranker.rerank("query", candidates, timestamp_map=timestamp_map)

            # Both should be present (no threshold filtering)
            assert len(result) == 2
            # doc2 is much newer so should have higher decayed score
            doc_scores = {doc: score for doc, score in result}
            assert doc_scores["doc2"] > doc_scores["doc1"]

    def test_recency_does_not_gate_old_docs_at_default_threshold(self) -> None:
        """Age may reorder but must not empty a batch that passed CE quality."""
        config = CrossEncoderConfig(
            enabled=True,
            top_k=10,
            score_threshold=0.5,
            batch_normalize=False,
            enable_recency_boost=True,
            recency_decay_rate=0.01,
        )
        with patch("FlagEmbedding.FlagReranker"):
            reranker = CrossEncoderReranker(config=config)
            reranker._model = MagicMock()
            assert reranker._model is not None
            reranker._model.compute_score.return_value = [0.9, 0.8]
            week_ago = "2026-08-22T00:00:00+00:00"
            result = reranker.rerank(
                "query",
                [("old-a", 0.7), ("old-b", 0.6)],
                timestamp_map={"old-a": week_ago, "old-b": week_ago},
            )
        assert [name for name, _score in result] == ["old-a", "old-b"]

    def test_recency_does_not_gate_equal_scores_with_batch_normalize(self) -> None:
        """Default batch min-max plus 0.5 must keep equal-score week-old docs."""
        config = CrossEncoderConfig(
            enabled=True,
            top_k=10,
            score_threshold=0.5,
            normalize=False,
            batch_normalize=True,
            enable_recency_boost=True,
            recency_decay_rate=0.01,
        )
        with patch("FlagEmbedding.FlagReranker"):
            reranker = CrossEncoderReranker(config=config)
            reranker._model = MagicMock()
            assert reranker._model is not None
            reranker._model.compute_score.return_value = [0.8, 0.8]
            week_ago = "2026-08-22T00:00:00+00:00"
            result = reranker.rerank(
                "query",
                [("old-a", 0.7), ("old-b", 0.6)],
                timestamp_map={"old-a": week_ago, "old-b": week_ago},
            )
        assert [name for name, _score in result] == ["old-a", "old-b"]

    def test_top_k_is_applied_after_recency(self) -> None:
        """Recency can promote a later CE rank into the top_k window."""
        config = CrossEncoderConfig(
            enabled=True,
            top_k=1,
            score_threshold=0.0,
            batch_normalize=False,
            enable_recency_boost=True,
            recency_decay_rate=0.01,
        )
        with patch("FlagEmbedding.FlagReranker"):
            reranker = CrossEncoderReranker(config=config)
            reranker._model = MagicMock()
            assert reranker._model is not None
            reranker._model.compute_score.return_value = [0.9, 0.8]
            result = reranker.rerank(
                "query",
                [("old", 0.7), ("new", 0.6)],
                timestamp_map={
                    "old": "2026-08-01T00:00:00+00:00",
                    "new": "2026-08-29T00:00:00+00:00",
                },
            )
        assert [name for name, _score in result] == ["new"]

    def test_recency_decay_logs_when_logger_present(self) -> None:
        """Test recency decay logs pre/post decay scores when logger is present (lines 367-368)."""
        config = CrossEncoderConfig(
            enabled=True,
            top_k=10,
            score_threshold=0.0,
            batch_normalize=False,
            enable_recency_boost=True,
            recency_decay_rate=0.01,
        )

        with patch("FlagEmbedding.FlagReranker"):
            logger, mock_logger = create_mock_logger()
            reranker = CrossEncoderReranker(config=config, logger=logger)
            mock_model = MagicMock()
            reranker._model = mock_model
            assert reranker._model is not None
            mock_model.compute_score.return_value = [0.9, 0.8]

            candidates = [("doc1", 0.5), ("doc2", 0.5)]
            timestamp_map = {
                "doc1": "2020-01-01T00:00:00+00:00",
                "doc2": "2026-02-13T00:00:00+00:00",
            }

            reranker.rerank("query", candidates, timestamp_map=timestamp_map)

            # Should log recency decay debug info
            debug_calls = mock_logger.debug.call_args_list
            decay_calls = [c for c in debug_calls if "Recency decay" in c[0][0]]
            assert len(decay_calls) == 1
            decay_call = decay_calls[0]
            assert decay_call[1]["extra"]["recency_decay"] is True
            assert decay_call[1]["extra"]["decay_rate"] == 0.01

    def test_recency_decay_skipped_when_disabled(self) -> None:
        """Test recency decay is skipped when enable_recency_boost=False."""
        config = CrossEncoderConfig(
            enabled=True,
            top_k=10,
            score_threshold=0.0,
            batch_normalize=False,
            enable_recency_boost=False,
            recency_decay_rate=0.01,
        )

        with patch("FlagEmbedding.FlagReranker"):
            reranker = CrossEncoderReranker(config=config)
            reranker._model = MagicMock()
            assert reranker._model is not None
            reranker._model.compute_score.return_value = [0.9, 0.8]

            candidates = [("doc1", 0.5), ("doc2", 0.5)]
            timestamp_map = {
                "doc1": "2020-01-01T00:00:00+00:00",
                "doc2": "2026-02-13T00:00:00+00:00",
            }

            result = reranker.rerank("query", candidates, timestamp_map=timestamp_map)

            # Scores should remain raw (no decay applied)
            assert result[0] == ("doc1", 0.9)
            assert result[1] == ("doc2", 0.8)

    def test_partial_timestamps_do_not_reverse_post_ce_order(self) -> None:
        """A stronger old hit must not lose to a weaker hit that lacks a stamp."""
        config = CrossEncoderConfig(
            enabled=True,
            top_k=10,
            score_threshold=0.0,
            batch_normalize=False,
            enable_recency_boost=True,
            recency_decay_rate=0.01,
        )
        with patch("FlagEmbedding.FlagReranker"):
            reranker = CrossEncoderReranker(config=config)
            reranker._model = MagicMock()
            assert reranker._model is not None
            reranker._model.compute_score.return_value = [0.9, 0.8]
            result = reranker.rerank(
                "query",
                [("old-strong", 0.7), ("new-weak", 0.6)],
                timestamp_map={"old-strong": "2020-01-01T00:00:00+00:00"},
            )
        assert result[0] == ("old-strong", 0.9)
        assert result[1] == ("new-weak", 0.8)

    def test_invalid_timestamp_disables_recency_for_batch(self) -> None:
        config = CrossEncoderConfig(
            enabled=True,
            top_k=10,
            score_threshold=0.0,
            batch_normalize=False,
            enable_recency_boost=True,
            recency_decay_rate=0.01,
        )
        with patch("FlagEmbedding.FlagReranker"):
            reranker = CrossEncoderReranker(config=config)
            reranker._model = MagicMock()
            assert reranker._model is not None
            reranker._model.compute_score.return_value = [0.9, 0.8]
            result = reranker.rerank(
                "query",
                [("a", 0.7), ("b", 0.6)],
                timestamp_map={"a": "2026-08-29T00:00:00+00:00", "b": "not-a-time"},
            )
        assert result[0] == ("a", 0.9)
        assert result[1] == ("b", 0.8)

    def test_recency_decay_skipped_when_no_timestamp_map(self) -> None:
        """Test recency decay is skipped when timestamp_map is None."""
        config = CrossEncoderConfig(
            enabled=True,
            top_k=10,
            score_threshold=0.0,
            batch_normalize=False,
            enable_recency_boost=True,
            recency_decay_rate=0.01,
        )

        with patch("FlagEmbedding.FlagReranker"):
            reranker = CrossEncoderReranker(config=config)
            reranker._model = MagicMock()
            assert reranker._model is not None
            reranker._model.compute_score.return_value = [0.9, 0.8]

            candidates = [("doc1", 0.5), ("doc2", 0.5)]

            result = reranker.rerank("query", candidates, timestamp_map=None)

            # Scores should remain raw (no decay applied)
            assert result[0] == ("doc1", 0.9)
            assert result[1] == ("doc2", 0.8)


class TestMinResultsSafetyNet:
    """Test min_results safety net feature."""

    def test_safety_net_returns_min_results_when_all_filtered(self) -> None:
        """Test that safety net returns min_results when all below threshold."""
        config = CrossEncoderConfig(
            enabled=True,
            top_k=10,
            score_threshold=0.9,  # Very high threshold
            normalize=False,
            batch_normalize=True,
            min_results=2,  # Safety net: return at least 2
        )

        with patch("FlagEmbedding.FlagReranker"):
            reranker = CrossEncoderReranker(config=config)
            reranker._model = MagicMock()

            # Assert to narrow the type from BaseReranker | None to MagicMock

            assert reranker._model is not None
            # After normalization: 0.6->1.0, 0.4->0.5, 0.2->0.0
            # None would pass 0.9 threshold without safety net
            reranker._model.compute_score.return_value = [0.6, 0.4, 0.2]

            candidates = [("doc1", 0.8), ("doc2", 0.7), ("doc3", 0.6)]
            result = reranker.rerank("query", candidates)

            # Safety net should return top 2
            assert len(result) == 2
            assert result[0][0] == "doc1"  # Best score (1.0)
            assert result[1][0] == "doc2"  # Second best (0.5)

    def test_no_safety_net_when_min_results_zero(self) -> None:
        """Test that min_results=0 allows empty results."""
        config = CrossEncoderConfig(
            enabled=True,
            top_k=10,
            score_threshold=0.9,  # Very high threshold
            normalize=False,
            batch_normalize=True,
            min_results=0,  # No safety net
        )

        with patch("FlagEmbedding.FlagReranker"):
            reranker = CrossEncoderReranker(config=config)
            reranker._model = MagicMock()

            # Assert to narrow the type from BaseReranker | None to MagicMock

            assert reranker._model is not None
            # After normalization: max is 1.0, but no scores pass 0.9 threshold
            # because we need normalized score >= 0.9
            reranker._model.compute_score.return_value = [0.6, 0.4, 0.2]

            candidates = [("doc1", 0.8), ("doc2", 0.7), ("doc3", 0.6)]
            result = reranker.rerank("query", candidates)

            # Only best (1.0) passes 0.9 threshold, rest filtered
            assert len(result) == 1
            assert result[0][0] == "doc1"

    def test_safety_net_not_needed_when_enough_pass(self) -> None:
        """Test that safety net doesn't interfere when enough results pass."""
        config = CrossEncoderConfig(
            enabled=True,
            top_k=10,
            score_threshold=0.3,  # Low threshold
            normalize=False,
            batch_normalize=True,
            min_results=1,
        )

        with patch("FlagEmbedding.FlagReranker"):
            reranker = CrossEncoderReranker(config=config)
            reranker._model = MagicMock()

            # Assert to narrow the type from BaseReranker | None to MagicMock

            assert reranker._model is not None
            # After normalization: 0.9->1.0, 0.7->0.5, 0.5->0.0
            # Only 1.0 and 0.5 pass 0.3 threshold (0.0 doesn't pass)
            reranker._model.compute_score.return_value = [0.9, 0.7, 0.5]

            candidates = [("doc1", 0.8), ("doc2", 0.7), ("doc3", 0.6)]
            result = reranker.rerank("query", candidates)

            # doc1 (1.0) and doc2 (0.5) pass 0.3 threshold, doc3 (0.0) doesn't
            assert len(result) == 2
            assert result[0][0] == "doc1"
            assert result[1][0] == "doc2"
